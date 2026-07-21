"""FastAPI web UI for Immich custom AI description generator."""

import asyncio
import base64
import json
import logging
import os
import random
import time as _time
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from engine import (
    DEFAULT_CONFIG,
    ImmichClient,
    LLMPool,
    ProcessingEngine,
    PromptBuilder,
    StateDB,
    auth_headers,
    build_chat_body,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("custom-ai-desc")

# Bootstrap env vars (required to boot — rest lives in SQLite)
IMMICH_API_URL = os.environ.get("IMMICH_API_URL", "http://immich-server:2283/api")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "")
DB_URL = os.environ.get("DB_URL", "")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
PORT = int(os.environ.get("PORT", "8095"))

# Globals
state_db: StateDB = None
immich: ImmichClient = None
engine: ProcessingEngine = None
_running_engine: ProcessingEngine = None  # preserve reference during active run
_task: asyncio.Task = None


def build_engine():
    """Build/rebuild processing engine from current config."""
    global engine
    config = {**DEFAULT_CONFIG, **state_db.get_all_config()}

    providers = state_db.get_providers(enabled_only=True, include_secrets=True)
    llm_pool = LLMPool(providers=providers)

    prompt_builder = PromptBuilder(template=config.get("prompt_template"))
    engine = ProcessingEngine(state_db, immich, llm_pool, prompt_builder)


def _get_active_engine() -> ProcessingEngine:
    """Return the running engine if a batch is active, otherwise the current engine."""
    if _running_engine and _task and not _task.done():
        return _running_engine
    return engine


def _migrate_old_config():
    """On first boot after upgrade: migrate old llm_endpoints config to providers table."""
    if state_db.get_providers():
        return  # already have providers
    config = state_db.get_all_config()
    endpoints_str = config.get("llm_endpoints", "")
    if not endpoints_str:
        return
    endpoints = [e.strip() for e in endpoints_str.split(",") if e.strip()]
    if not endpoints:
        return

    model = config.get("llm_model", "qwen3.5-4b")
    concurrency = int(config.get("llm_concurrency", "4"))
    timeout = int(config.get("llm_timeout", "120"))
    max_tokens = int(config.get("max_tokens", "500"))
    temperature = float(config.get("temperature", "0.7"))

    for ep in endpoints:
        state_db.add_provider({
            "url": ep,
            "model": model,
            "concurrency": concurrency,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_params": DEFAULT_EXTRA_PARAMS,
        })
    logger.info(f"Migrated {len(endpoints)} old endpoint(s) to providers table")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state_db, immich
    db_path = os.path.join(DATA_DIR, "state.db")
    state_db = StateDB(db_path)

    if not state_db.has_config():
        state_db.save_defaults()

    _migrate_old_config()

    immich = ImmichClient(IMMICH_API_URL, IMMICH_API_KEY, db_url=DB_URL or None)
    build_engine()

    logger.info(f"Started — Immich: {IMMICH_API_URL}, Data: {DATA_DIR}")
    scheduler = asyncio.create_task(_scheduler())
    yield

    scheduler.cancel()
    try:
        await scheduler
    except asyncio.CancelledError:
        pass
    if _task and not _task.done():
        engine.stop()
        await asyncio.sleep(1)
    if _audit_task and not _audit_task.done():
        if engine:
            engine._stop_event.set()
        await asyncio.sleep(1)
    if immich:
        await immich.close()
    if engine and engine.llm:
        await engine.llm.close()
    if state_db:
        state_db.close()


app = FastAPI(title="Immich AI Descriptions", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    config = {**DEFAULT_CONFIG, **(state_db.get_all_config() if state_db else {})}
    stats = state_db.get_stats() if state_db else {}
    runs = state_db.get_runs(5) if state_db else []
    recent = state_db.get_recent(20) if state_db else []
    progress = engine.progress if engine else None

    return templates.TemplateResponse(request, "index.html", {
        "config": config,
        "stats": stats,
        "runs": runs,
        "recent": recent,
        "progress": progress,
        "has_config": state_db.has_config() if state_db else False,
    })


# --- Stats & Recent polling endpoints ---


@app.get("/stats", response_class=HTMLResponse)
async def stats_partial():
    stats = state_db.get_stats() if state_db else {"done": 0, "failed": 0, "skipped": 0, "total_processed": 0}

    # Use library_count from last batch run (ground truth from Immich search API)
    row = state_db.conn.execute(
        "SELECT value FROM metadata WHERE key = 'library_count'"
    ).fetchone() if state_db else None
    library = int(row["value"]) if row else 0
    remaining = max(0, library - stats.get("total_processed", 0))

    return HTMLResponse(f"""
        <div class="glass p-5">
            <div class="font-display text-3xl font-bold text-ok stat-glow-ok">{stats.get('done', 0)}</div>
            <div class="text-txt-3 text-xs mt-2 font-mono uppercase tracking-widest">Done</div>
        </div>
        <div class="glass p-5">
            <div class="font-display text-3xl font-bold text-err stat-glow-err">{stats.get('failed', 0)}</div>
            <div class="text-txt-3 text-xs mt-2 font-mono uppercase tracking-widest">Failed</div>
        </div>
        <div class="glass p-5">
            <div class="font-display text-3xl font-bold text-warn stat-glow-warn">{stats.get('skipped', 0)}</div>
            <div class="text-txt-3 text-xs mt-2 font-mono uppercase tracking-widest">Skipped</div>
        </div>
        <div class="glass p-5">
            <div class="font-display text-3xl font-bold text-cool">{remaining}</div>
            <div class="text-txt-3 text-xs mt-2 font-mono uppercase tracking-widest">Remaining</div>
        </div>
    """)


@app.get("/recent", response_class=HTMLResponse)
async def recent_partial():
    recent = state_db.get_recent(20) if state_db else []
    if not recent:
        return HTMLResponse('<p class="text-txt-3 text-xs">No activity yet.</p>')
    rows = ""
    for a in recent:
        dot = {"done": "dot-ok", "failed": "dot-err"}.get(a["status"], "dot-warn")
        preview = escape((a.get("description_preview") or a.get("error") or "")[:80])
        dur = f'{a.get("duration_ms", 0)}ms' if a.get("duration_ms") else "-"
        model = escape(a.get("model_used") or "-")

        rows += f"""<tr>
            <td><span class="dot {dot}"></span></td>
            <td class="font-mono text-txt-2 text-xs">{a['asset_id'][:12]}...</td>
            <td class="text-xs max-w-xs truncate text-txt-2">{preview}</td>
            <td class="font-mono text-xs text-txt-3">{model}</td>
            <td class="font-mono text-xs text-txt-3">{dur}</td>
        </tr>"""

    return HTMLResponse(f"""<table class="data-table">
        <thead><tr>
            <th>Status</th><th>Asset</th><th>Preview</th><th>Model</th><th>Time</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>""")


# --- Settings ---


@app.post("/settings", response_class=HTMLResponse)
async def save_settings(request: Request):
    form = await request.form()
    for key in DEFAULT_CONFIG:
        if key in form:
            state_db.set_config(key, form[key])
    # Handle checkbox (unchecked = absent from form)
    if "overwrite_existing" not in form:
        state_db.set_config("overwrite_existing", "false")
    if "quality_gate_enabled" not in form:
        state_db.set_config("quality_gate_enabled", "false")
    build_engine()
    return HTMLResponse(
        '<span class="text-ok text-xs font-mono" style="animation:fadeUp 0.3s ease-out">Saved</span>'
    )


# --- Provider CRUD ---

DEFAULT_EXTRA_PARAMS = json.dumps(
    {"top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.1, "reasoning_format": "none"},
    indent=2,
)

# 1x1 transparent PNG — enough for a "does this provider/key/params combo work" test call.
TEST_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _pretty_json(raw: str) -> str:
    """Pretty-print stored extra_params JSON; falls back to the raw string if it's malformed
    so a bad edit is never silently discarded — the user sees exactly what they typed."""
    try:
        return json.dumps(json.loads(raw), indent=2)
    except (TypeError, ValueError):
        return raw or "{}"


def _render_provider_card(p: dict, models_html: str = "", error: str = "") -> str:
    """Render a single provider card form."""
    pid = p.get("id", "new")
    is_new = pid == "new"
    checked = "checked" if p.get("enabled", 1) else ""
    enabled_dot = "dot-ok" if p.get("enabled", 1) else "dot-err"
    extra_params = DEFAULT_EXTRA_PARAMS if is_new else _pretty_json(p.get("extra_params") or "{}")
    last4 = p.get("api_key_last4")
    key_placeholder = (
        f"•••• saved (…{last4}) · leave blank to keep" if last4
        else "sk-... (optional, for hosted APIs)"
    )
    # Existing cards auto-save themselves in place; the "new" card (client-side only,
    # see addProviderCard() in index.html) still targets the full list on its one-time Save.
    target = "#provider-list" if is_new else f"#provider-{pid}"
    swap = "innerHTML" if is_new else "outerHTML"
    error_html = f'<p class="text-err text-xs font-mono">{escape(error)}</p>' if error else ""
    return f"""
    <div class="glass p-5 provider-card" id="provider-{pid}">
      <form hx-post="/providers" hx-target="{target}" hx-swap="{swap}"
            {'hx-trigger="change delay:500ms"' if not is_new else ''} class="space-y-4">
        <input type="hidden" name="id" value="{pid if not is_new else ''}">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="dot {enabled_dot}"></span>
            <span class="font-display text-sm font-semibold text-txt">Provider #{pid}</span>
          </div>
          <div class="flex items-center gap-4">
            <label class="flex items-center gap-2 text-xs text-txt-3 cursor-pointer">
              <input type="checkbox" name="enabled" value="1" {checked}> Enabled
            </label>
            {f'<button type="button" hx-delete="/providers/{pid}" hx-target="#provider-list" hx-swap="innerHTML" hx-confirm="Delete this provider?" class="text-err hover:text-err text-xs opacity-60 hover:opacity-100 transition-opacity">Delete</button>' if not is_new else ''}
          </div>
        </div>

        {error_html}

        <div class="flex gap-2">
          <input name="url" value="{escape(p.get('url', ''))}" placeholder="http://your-llm-server:1234/v1"
                 class="input-field flex-1">
          <button type="button" hx-post="/providers/validate" hx-include="closest form"
                  hx-target="#models-{pid}" hx-swap="innerHTML"
                  class="btn btn-ghost btn-sm whitespace-nowrap">Validate</button>
        </div>

        <div>
          <label class="block text-xs text-txt-3 mb-1.5">Model</label>
          <div id="models-{pid}" class="flex gap-2">
            {models_html or f'<input name="model" value="{escape(p.get("model", ""))}" placeholder="model name" class="input-field flex-1">'}
          </div>
        </div>

        <div>
          <label class="block text-xs text-txt-3 mb-1.5">API Key</label>
          <div class="flex gap-2 items-center">
            <input type="password" name="api_key" value="" placeholder="{escape(key_placeholder)}"
                   autocomplete="off" class="input-field flex-1 font-mono">
            <label class="flex items-center gap-1.5 text-xs text-txt-3 cursor-pointer whitespace-nowrap">
              <input type="checkbox" name="clear_api_key" value="1"> Remove key
            </label>
          </div>
        </div>

        <details class="text-xs">
          <summary class="text-txt-3 cursor-pointer hover:text-txt-2 text-xs">Parameters</summary>
          <div class="grid grid-cols-4 gap-3 mt-3">
            <div><label class="block text-txt-3 mb-1 text-xs">Concurrency</label>
              <input name="concurrency" type="number" value="{p.get('concurrency', 4)}" class="input-field text-xs"></div>
            <div><label class="block text-txt-3 mb-1 text-xs">Timeout (s)</label>
              <input name="timeout" type="number" value="{p.get('timeout', 120)}" class="input-field text-xs"></div>
            <div><label class="block text-txt-3 mb-1 text-xs">Max Tokens</label>
              <input name="max_tokens" type="number" value="{p.get('max_tokens', 500)}" class="input-field text-xs"></div>
            <div><label class="block text-txt-3 mb-1 text-xs">Temperature</label>
              <input name="temperature" type="number" step="0.05" value="{p.get('temperature', 0.7)}" class="input-field text-xs"></div>
          </div>
          <div class="mt-3">
            <label class="block text-txt-3 mb-1 text-xs">Extra request body params (JSON, merged into every request)</label>
            <textarea name="extra_params" rows="6" spellcheck="false"
                      class="input-field text-xs font-mono w-full">{escape(extra_params)}</textarea>
          </div>
        </details>

        <div id="test-result-{pid}" class="text-xs"></div>

        <div class="flex gap-2 items-center">
          <button type="button" hx-post="/providers/test" hx-include="closest form"
                  hx-target="#test-result-{pid}" hx-swap="innerHTML"
                  class="btn btn-ghost btn-sm">Test</button>
          {'' if not is_new else '<button type="submit" class="btn btn-warm btn-sm">Save</button>'}
        </div>
      </form>
    </div>"""


@app.get("/providers", response_class=HTMLResponse)
async def list_providers():
    providers = state_db.get_providers()
    if not providers:
        return HTMLResponse(
            '<p class="text-txt-3 text-xs py-2">No providers configured. Add one below.</p>'
        )
    cards = "".join(_render_provider_card(p) for p in providers)
    return HTMLResponse(cards)


def _resolve_test_api_key(form) -> Optional[str]:
    """Typed key wins; otherwise fall back to the stored (decrypted) key for an
    existing provider, so Validate/Test work without retyping an already-saved key."""
    api_key = (form.get("api_key") or "").strip()
    if api_key:
        return api_key
    pid = (form.get("id") or "").strip()
    if not pid:
        return None
    stored = state_db.get_provider(int(pid), include_secrets=True)
    return stored.get("api_key") if stored else None


@app.post("/providers/validate", response_class=HTMLResponse)
async def validate_provider(request: Request):
    form = await request.form()
    url = (form.get("url") or "").strip().rstrip("/")
    current_model = (form.get("model") or "").strip()
    if not url:
        return HTMLResponse('<span class="text-err text-xs">No URL</span>')

    try:
        api_key = _resolve_test_api_key(form)
    except Exception:
        return HTMLResponse('<span class="text-err text-xs">Could not read stored key</span>')

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/models", headers=headers)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                names = [m.get("id", "?") for m in models]
                if not names:
                    return HTMLResponse('<span class="text-warn text-xs">No models found</span>')
                options = "".join(
                    f'<option value="{escape(n)}" {"selected" if n == current_model else ""}>{escape(n)}</option>'
                    for n in names
                )
                return HTMLResponse(
                    f'<select name="model" class="input-field flex-1">'
                    f'{options}</select>'
                    f'<span class="text-ok text-xs whitespace-nowrap font-mono">{len(names)} models</span>'
                )
            return HTMLResponse(f'<span class="text-warn text-xs">Status {resp.status_code}</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="text-err text-xs">{escape(str(e))}</span>')


@app.post("/providers/test", response_class=HTMLResponse)
async def test_provider(request: Request):
    """Fires one real chat-completion request with the form's current url/model/key/
    extra_params so a hosted provider's rejection (bad param, bad key, ...) shows up
    verbatim before the provider is trusted with a full batch run."""
    form = await request.form()
    url = (form.get("url") or "").strip().rstrip("/")
    model = (form.get("model") or "").strip()
    if not url or not model:
        return HTMLResponse('<span class="text-err text-xs">URL and model required to test</span>')

    extra_params_raw = (form.get("extra_params") or "{}").strip()
    try:
        json.loads(extra_params_raw or "{}")
    except json.JSONDecodeError as e:
        return HTMLResponse(f'<span class="text-err text-xs font-mono">Invalid extra params JSON: {escape(str(e))}</span>')

    try:
        api_key = _resolve_test_api_key(form)
    except Exception:
        return HTMLResponse('<span class="text-err text-xs">Could not read stored key — re-enter it to test</span>')

    provider = {
        "id": (form.get("id") or "").strip() or "test",
        "url": url,
        "model": model,
        "api_key": api_key,
        "max_tokens": 10,
        "temperature": float(form.get("temperature", 0.7)),
        "extra_params": extra_params_raw,
    }
    body = build_chat_body(provider, "Reply with just OK.", TEST_IMAGE_B64)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{url}/chat/completions", json=body, headers=auth_headers(provider))
        if resp.status_code >= 400:
            return HTMLResponse(
                f'<span class="text-err text-xs font-mono">HTTP {resp.status_code}: {escape(resp.text[:500])}</span>'
            )
        data = resp.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return HTMLResponse(f'<span class="text-ok text-xs">OK — provider responded: {escape(reply[:200])}</span>')
    except httpx.TimeoutException:
        return HTMLResponse('<span class="text-err text-xs">Timed out</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="text-err text-xs">{escape(str(e))}</span>')


@app.post("/providers", response_class=HTMLResponse)
async def save_provider(request: Request):
    form = await request.form()

    extra_params_raw = (form.get("extra_params") or "{}").strip()
    try:
        json.loads(extra_params_raw or "{}")
    except json.JSONDecodeError as e:
        return await _provider_save_error(form, f"Invalid extra params JSON: {e}")

    data = {
        "url": (form.get("url") or "").strip().rstrip("/"),
        "model": (form.get("model") or "").strip(),
        "concurrency": int(form.get("concurrency", 4)),
        "timeout": int(form.get("timeout", 120)),
        "max_tokens": int(form.get("max_tokens", 500)),
        "temperature": float(form.get("temperature", 0.7)),
        "enabled": 1 if form.get("enabled") else 0,
        "extra_params": extra_params_raw,
    }
    api_key = (form.get("api_key") or "").strip()
    if api_key:
        data["api_key"] = api_key
    elif form.get("clear_api_key"):
        data["clear_api_key"] = True

    if not data["url"] or not data["model"]:
        return await _provider_save_error(form, "URL and model required")

    pid = (form.get("id") or "").strip()
    if pid:
        state_db.update_provider(int(pid), data)
        build_engine()
        updated = state_db.get_provider(int(pid))
        return HTMLResponse(_render_provider_card(updated))
    else:
        state_db.add_provider(data)
        build_engine()
        return await list_providers()


async def _provider_save_error(form, message: str) -> HTMLResponse:
    """On validation failure, re-render the card (not just an error span) so an
    existing provider's outerHTML auto-save target doesn't get replaced with a
    dead-end error — the form has to survive so the user can fix and resubmit."""
    pid = (form.get("id") or "").strip()
    if not pid:
        return HTMLResponse(f'<span class="text-err text-xs font-mono">{escape(message)}</span>')
    display = state_db.get_provider(int(pid)) or {"id": int(pid)}
    display.update({
        "url": form.get("url", display.get("url", "")),
        "model": form.get("model", display.get("model", "")),
        "concurrency": form.get("concurrency", display.get("concurrency")),
        "timeout": form.get("timeout", display.get("timeout")),
        "max_tokens": form.get("max_tokens", display.get("max_tokens")),
        "temperature": form.get("temperature", display.get("temperature")),
        "enabled": 1 if form.get("enabled") else 0,
        "extra_params": form.get("extra_params", display.get("extra_params")),
    })
    return HTMLResponse(_render_provider_card(display, error=message))


@app.delete("/providers/{provider_id}", response_class=HTMLResponse)
async def delete_provider(provider_id: int):
    state_db.delete_provider(provider_id)
    build_engine()
    return await list_providers()


# --- Endpoints health ---


@app.get("/endpoints", response_class=HTMLResponse)
async def endpoints_status():
    if not engine or not engine.llm.providers:
        return HTMLResponse("<p class='text-txt-3 text-xs'>No providers configured</p>")
    health = await engine.llm.health_check()
    rows = ""
    for pid, info in health.items():
        dot = "dot-ok" if info["online"] else "dot-err"
        rows += (
            f'<tr>'
            f'<td class="font-mono text-txt-3 text-xs">#{pid}</td>'
            f'<td class="font-mono text-txt-2 text-xs">{escape(info["url"])}</td>'
            f'<td class="font-mono text-xs text-txt-3">{escape(info["model"])}</td>'
            f'<td><span class="dot {dot}"></span></td>'
            f'</tr>'
        )
    return HTMLResponse(
        f'<table class="data-table"><thead><tr>'
        f'<th>ID</th><th>URL</th><th>Model</th><th>Status</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


# --- Run controls ---


def _task_done_callback(task: asyncio.Task):
    """Log unhandled exceptions from background batch tasks."""
    try:
        exc = task.exception()
        if exc:
            logger.error(f"Batch task failed: {exc}")
            if engine:
                engine.progress.message = f"Error: {exc}"
    except asyncio.CancelledError:
        pass


async def _scheduler():
    """Background scheduler: runs a batch every watch_interval_hours."""
    while True:
        hours = int(state_db.get_config("watch_interval_hours", "24"))
        logger.info(f"Scheduler: next run in {hours}h")
        await asyncio.sleep(hours * 3600)
        # Skip if a batch or audit is already running
        if (_task and not _task.done()) or (_audit_task and not _audit_task.done()):
            logger.info("Scheduler: skipped — batch or audit already running")
            continue
        logger.info("Scheduler: starting scheduled batch")
        build_engine()
        _task_ref = asyncio.create_task(engine.run_batch(resume=True))
        _task_ref.add_done_callback(_task_done_callback)
        # Update globals so UI tracks it
        globals()["_task"] = _task_ref
        globals()["_running_engine"] = engine


@app.post("/run/start")
async def start_run(request: Request):
    global _task, _running_engine
    if _task and not _task.done():
        return JSONResponse({"error": "Already running"}, status_code=409)
    fresh = request.query_params.get("fresh") == "true"
    build_engine()
    if fresh:
        state_db.clear_all()
        state_db.conn.execute("DELETE FROM metadata WHERE key = 'last_run_timestamp'")
        state_db.conn.commit()
        logger.info("Fresh start: cleared all processed assets and last_run_timestamp")
    _running_engine = engine
    _task = asyncio.create_task(engine.run_batch(resume=True))
    _task.add_done_callback(_task_done_callback)
    return JSONResponse({"status": "started"})


@app.post("/run/stop")
async def stop_run():
    active = _get_active_engine()
    if active:
        active.stop()
    return JSONResponse({"status": "stopping"})


@app.get("/run/status", response_class=HTMLResponse)
async def run_status():
    active = _get_active_engine()
    if not active:
        return HTMLResponse("<p>Not initialized</p>")
    p = active.progress

    total = p.total or 1
    completed = p.done + p.failed + p.skipped
    pct = min(100, int((completed / total) * 100))

    if p.running:
        bar_class = "progress-active"
        html = f"""
    <div class="space-y-4">
        <div class="flex items-center gap-3">
            <button onclick="fetch('/run/stop',{{method:'POST'}})" class="btn btn-danger btn-sm">Stop</button>
            <div class="dot dot-warn dot-pulse"></div>
            <span class="text-txt-2 text-sm">Processing...</span>
        </div>
        <div class="flex justify-between text-sm">
            <span class="text-txt-2">{escape(p.message)}</span>
            <span class="font-mono text-warm text-xs">{completed}/{p.total}</span>
        </div>
        <div class="w-full bg-base-deep rounded-full h-2 overflow-hidden">
            <div class="{bar_class} rounded-full h-2 transition-all duration-500"
                 style="width:{pct}%"></div>
        </div>
        <div class="flex gap-6 text-xs font-mono">
            <span class="text-ok">{p.done} done</span>
            <span class="text-err">{p.failed} failed</span>
            <span class="text-warn">{p.skipped} skip</span>
            <span class="text-txt-3">{p.rate:.1f}/min</span>
        </div>"""

        # Per-provider breakdown
        provider_rows = ""
        for pid, ps in p.provider_stats.items():
            model = escape(ps.get("model", "?"))
            status = ps.get("status", "active")
            status_dot = {
                "active": "dot-ok",
                "backoff": "dot-warn dot-pulse",
                "disabled": "dot-err",
            }.get(status, "dot-ok")
            provider_rows += (
                f'<div class="flex justify-between items-center text-xs">'
                f'<span class="text-txt-3 font-mono">'
                f'<span class="dot {status_dot}" style="margin-right:6px"></span>'
                f'#{pid} '
                f'<span class="text-txt-2">{model}</span></span>'
                f'<span class="font-mono">'
                f'<span class="text-ok">{ps["done"]}</span> / '
                f'<span class="text-err">{ps["failed"]}</span>'
                f' &middot; <span class="text-txt-3">{ps["rate"]:.1f}/min</span>'
                f'</span></div>'
            )
        if provider_rows:
            html += f"""
        <div class="mt-3 pt-3 border-t border-edge space-y-1.5">
            <div class="text-txt-3 text-xs font-mono uppercase tracking-widest mb-1">Per Provider</div>
            {provider_rows}
        </div>"""

        html += "\n    </div>"
    else:
        html = f"""
    <div class="space-y-4">
        <div class="flex items-center gap-3 flex-wrap">
            <button onclick="fetch('/run/start',{{method:'POST'}})" class="btn btn-warm btn-sm">Start Processing</button>
            <button onclick="fetch('/run/reset-failed',{{method:'POST'}})" class="btn btn-ghost btn-sm">Reset Failed</button>
            <button onclick="if(confirm('WARNING: This will DELETE all {state_db.get_stats().get('done',0)} tracked records from the database and re-generate descriptions for ALL photos from scratch.\\n\\nExisting descriptions in Immich will be OVERWRITTEN.\\n\\nThis action cannot be undone. Continue?'))fetch('/run/start?fresh=true',{{method:'POST'}})"
                class="btn btn-ghost btn-sm text-err">Start Fresh</button>
        </div>
        <div class="mt-2 space-y-1 text-txt-3" style="font-size:11px">
            <p><span class="text-warm">Start Processing</span> — scans your library, skips already-described photos, processes the rest</p>
            <p><span class="text-txt-2">Reset Failed</span> — clears failed records so they get retried on next run</p>
            <p><span class="text-err">Start Fresh</span> — wipes all tracking and re-processes every photo from scratch</p>
        </div>
    </div>"""
    return HTMLResponse(html)


# --- Assets ---


@app.get("/assets", response_class=HTMLResponse)
async def list_assets(status: str = None, limit: int = 50):
    recent = state_db.get_recent(limit, status if status != "all" else None)
    if not recent:
        return HTMLResponse('<p class="text-txt-3 text-xs py-4">No assets processed yet</p>')

    rows = ""
    for a in recent:
        dot = {"done": "dot-ok", "failed": "dot-err"}.get(a["status"], "dot-warn")
        preview = escape((a.get("description_preview") or a.get("error") or "")[:100])
        dur = f'{a.get("duration_ms", 0)}ms' if a.get("duration_ms") else "-"
        score = a.get("quality_score")
        score_html = f'{score:.2f}' if score is not None else "-"
        score_class = "text-err" if score is not None and score > float(state_db.get_config("repetition_threshold", "0.3")) else "text-txt-3"
        ts = (a.get("processed_at") or "")[:16]
        model = escape(a.get("model_used") or "-")
        aid = a["asset_id"]

        retry = ""
        if a["status"] == "failed":
            retry = (
                f'<button hx-post="/assets/{aid}/retry" '
                f'hx-target="closest tr" hx-swap="outerHTML" '
                f'class="text-warm hover:text-warm text-xs opacity-60 hover:opacity-100 transition-opacity"'
                f' onclick="event.stopPropagation()">retry</button>'
            )

        rows += f"""<tr class="cursor-pointer"
                        hx-get="/assets/{aid}" hx-target="next .asset-detail" hx-swap="innerHTML"
                        onclick="this.nextElementSibling.classList.toggle('hidden')">
            <td><span class="dot {dot}"></span></td>
            <td class="font-mono text-txt-2 text-xs">{aid[:12]}...</td>
            <td class="text-xs max-w-md truncate text-txt-2">{preview}</td>
            <td class="font-mono text-xs text-txt-3">{model}</td>
            <td class="font-mono text-xs text-txt-3">{dur}</td>
            <td class="font-mono text-xs {score_class}">{score_html}</td>
            <td class="font-mono text-xs text-txt-3">{ts}</td>
            <td>{retry}</td>
        </tr>
        <tr class="asset-detail hidden"><td colspan="8" class="p-0">
            <div class="px-4 py-2 text-txt-3 text-xs">Loading...</div>
        </td></tr>"""

    return HTMLResponse(f"""<table class="data-table">
        <thead><tr>
            <th>Status</th><th>Asset</th><th>Preview</th><th>Model</th><th>Time</th><th>Score</th><th>Date</th><th></th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>""")


@app.get("/assets/{asset_id}", response_class=HTMLResponse)
async def asset_detail(asset_id: str):
    """Returns detail HTML for an asset: thumbnail, prompt, description, provider info."""
    row = state_db.get_asset(asset_id)

    if not row:
        return HTMLResponse('<td colspan="8" class="p-4 text-txt-3 text-xs">Not found</td>')

    # Fetch full description from Immich (DB preview is truncated to 200 chars)
    full_desc = row.get("description_preview") or ""
    if immich and row.get("status") in ("done", "failed"):
        try:
            detail = await immich.get_asset(asset_id)
            full_desc = detail.get("exifInfo", {}).get("description") or full_desc
        except Exception:
            pass
    preview = escape(full_desc)
    prompt = escape(row.get("prompt_used") or "N/A")
    model = escape(row.get("model_used") or "-")
    endpoint = escape(row.get("endpoint_used") or "-")
    pid = row.get("provider_id") or "-"
    dur = f'{row.get("duration_ms", 0)}ms' if row.get("duration_ms") else "-"
    score = row.get("quality_score")
    score_str = f'{score:.2f}' if score is not None else "N/A"
    error = escape(row.get("error") or "")

    immich_link = f"https://photos.maheidem.com/photos/{asset_id}"

    return HTMLResponse(f"""<td colspan="8" style="padding:0">
    <div style="background:#161925;border:1px solid #2a2d3a;border-radius:10px;padding:16px;margin:4px 8px">
        <div style="display:flex;gap:16px">
            <img src="/assets/{asset_id}/thumbnail" alt=""
                 style="width:120px;height:120px;object-fit:cover;border-radius:8px;border:1px solid #2a2d3a;flex-shrink:0"
                 onerror="this.style.display='none'">
            <div style="flex:1;min-width:0;font-size:13px">
                <div style="margin-bottom:8px">
                    <a href="{immich_link}" target="_blank" style="color:#d4915c;font-family:monospace;font-size:12px;margin-right:16px">Open in Immich</a>
                    <span style="color:#888;font-family:monospace;font-size:11px;margin-right:12px">Provider #{pid}</span>
                    <span style="color:#888;font-family:monospace;font-size:11px;margin-right:12px">{model}</span>
                    <span style="color:#888;font-family:monospace;font-size:11px">{dur}</span>
                    <span style="color:#888;font-family:monospace;font-size:11px">Score: {score_str}</span>
                </div>
                {f'<div style="color:#f87171;font-size:12px;margin-bottom:6px">Error: {error}</div>' if error else ''}
                <div style="color:#777;font-size:10px;font-family:monospace;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Description</div>
                <p style="color:#c8c6c3;line-height:1.6;margin:0 0 10px 0">{preview}</p>
                <details>
                    <summary style="color:#777;cursor:pointer;font-size:11px;font-family:monospace">Full Prompt</summary>
                    <pre style="background:#0f1119;border:1px solid #2a2d3a;padding:12px;border-radius:8px;margin-top:6px;
                                white-space:pre-wrap;line-height:1.5;color:#b0aead;font-family:monospace;font-size:11px;
                                max-height:200px;overflow-y:auto">{prompt}</pre>
                </details>
            </div>
        </div>
    </div>
    </td>""")


@app.get("/assets/{asset_id}/thumbnail")
async def asset_thumbnail(asset_id: str):
    """Proxy Immich thumbnail."""
    if not immich:
        return Response(status_code=503)
    thumb = await immich.get_thumbnail(asset_id)
    if not thumb:
        return Response(status_code=404)
    return Response(content=thumb, media_type="image/jpeg")


@app.post("/assets/{asset_id}/retry", response_class=HTMLResponse)
async def retry_asset(asset_id: str):
    state_db.conn.execute(
        "DELETE FROM processed WHERE asset_id = ?", (asset_id,)
    )
    state_db.conn.commit()
    return HTMLResponse(
        '<tr><td colspan="8" class="text-ok text-xs font-mono py-2 px-3">'
        "Queued for retry on next run</td></tr>"
    )


@app.get("/preview", response_class=HTMLResponse)
async def preview_playground():
    """Preview playground: 5 random asset cards with context and Generate buttons."""
    if not immich:
        return HTMLResponse("<p class='text-err text-xs'>Not connected to Immich</p>")
    try:
        resp = await immich.client.post(
            f"{immich.api_url}/search/metadata",
            json={"type": "IMAGE", "size": 100, "page": 1, "withExif": True},
        )
        resp.raise_for_status()
        items = resp.json().get("assets", {}).get("items", [])
        if not items:
            return HTMLResponse("<p class='text-txt-3 text-xs'>No assets found</p>")

        samples = random.sample(items, min(5, len(items)))
        providers = state_db.get_providers(enabled_only=True)
        cards = ""

        for asset in samples:
            aid = asset["id"]
            detail = await immich.get_asset(aid)
            exif = detail.get("exifInfo") or {}

            people = [p["name"] for p in detail.get("people", []) if p.get("name")]
            ocr_min = float(state_db.get_config("ocr_min_score", "0.5"))
            ocr = immich.get_ocr_text(aid, ocr_min)
            nearby = await immich.get_nearby_count(detail)

            prompt = engine.prompt.build(
                detail, ocr_text=ocr, nearby_count=nearby, people=people,
            )

            fname = escape(detail.get("originalFileName", "unknown"))
            cam_parts = [p for p in [exif.get("make"), exif.get("model")] if p]
            cam = escape(" ".join(cam_parts)) if cam_parts else ""
            dt_str = (exif.get("dateTimeOriginal") or "")[:16]
            city = exif.get("city") or ""
            state_name = exif.get("state") or ""
            loc_parts = [p for p in [city, state_name] if p]
            loc = escape(", ".join(loc_parts))

            people_html = ""
            if people:
                names = escape(", ".join(people[:6]))
                extra = f" +{len(people) - 6}" if len(people) > 6 else ""
                people_html = f'<div style="font-size:12px;color:#8b8990;margin-top:2px">People: {names}{extra}</div>'

            ocr_html = ""
            if ocr:
                ocr_preview = escape(ocr[:80]) + ("..." if len(ocr) > 80 else "")
                ocr_html = f'<div style="font-size:11px;color:#5a5862;margin-top:2px;font-style:italic">OCR: "{ocr_preview}"</div>'

            # Build per-provider Generate buttons
            provider_buttons = ""
            for prov in providers:
                pid = prov["id"]
                pmodel = escape(prov.get("model", "?"))
                short_model = escape(pmodel.split("/")[-1][:20])
                provider_buttons += (
                    f'<button hx-post="/preview/generate/{aid}?provider_id={pid}" '
                    f'hx-target="#preview-resp-{aid[:8]}-{pid}" '
                    f'hx-swap="innerHTML" '
                    f'class="btn btn-ghost btn-sm" style="font-size:11px">'
                    f'#{pid} {short_model}</button>'
                )
            # "All" button to fire all providers at once
            all_pids = ",".join(str(prov["id"]) for prov in providers)
            provider_buttons += (
                f'<button onclick="this.parentElement.querySelectorAll(\'[hx-post]\')'
                f'.forEach(b=>htmx.trigger(b,\'click\'))" '
                f'class="btn btn-warm btn-sm" style="font-size:11px">All</button>'
            )

            # Build response slots for each provider
            response_slots = ""
            for prov in providers:
                pid = prov["id"]
                response_slots += f'<div id="preview-resp-{aid[:8]}-{pid}"></div>'

            cards += f"""
            <div class="glass p-4 fade-up" style="margin-bottom:12px">
              <div style="display:flex;gap:14px">
                <img src="/assets/{aid}/thumbnail" alt=""
                     style="width:100px;height:100px;object-fit:cover;border-radius:8px;border:1px solid #252836;flex-shrink:0"
                     onerror="this.style.display='none'">
                <div style="flex:1;min-width:0">
                  <div style="font-size:13px;color:#e8e6e3;font-weight:500">{fname}</div>
                  <div style="font-size:11px;color:#5a5862;margin-top:2px;font-family:'JetBrains Mono',monospace">
                    {f'{cam} &middot; ' if cam else ''}{dt_str}{f' &middot; {loc}' if loc else ''}
                  </div>
                  {people_html}
                  {ocr_html}
                  <details style="margin-top:8px">
                    <summary style="color:#5a5862;cursor:pointer;font-size:11px;font-family:'JetBrains Mono',monospace">Prompt</summary>
                    <pre style="background:#0f1119;border:1px solid #252836;padding:10px;border-radius:8px;margin-top:6px;
                                white-space:pre-wrap;line-height:1.5;color:#b0aead;font-family:'JetBrains Mono',monospace;font-size:11px;
                                max-height:160px;overflow-y:auto">{escape(prompt)}</pre>
                  </details>
                  <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
                    {provider_buttons}
                  </div>
                  <div style="margin-top:8px">{response_slots}</div>
                </div>
              </div>
            </div>"""

        return HTMLResponse(cards)
    except Exception as e:
        return HTMLResponse(f"<p class='text-err text-xs'>Error: {escape(str(e))}</p>")


@app.post("/preview/generate/{asset_id}", response_class=HTMLResponse)
async def preview_generate(asset_id: str, request: Request):
    """Run full LLM pipeline on one asset WITHOUT saving to Immich."""
    if not immich or not engine:
        return HTMLResponse('<span class="text-err text-xs">Not initialized</span>')

    pid_str = request.query_params.get("provider_id")
    target_provider_id = int(pid_str) if pid_str else None

    try:
        start = _time.monotonic()

        detail = await immich.get_asset(asset_id)
        people = [p["name"] for p in detail.get("people", []) if p.get("name")]
        ocr_min = float(state_db.get_config("ocr_min_score", "0.5"))
        ocr = await asyncio.to_thread(immich.get_ocr_text, asset_id, ocr_min)
        window = int(state_db.get_config("nearby_window_minutes", "10"))
        nearby = await immich.get_nearby_count(detail, window)

        prompt_text = engine.prompt.build(
            detail, ocr_text=ocr, nearby_count=nearby, people=people,
        )

        thumb = await immich.get_thumbnail(asset_id)
        if not thumb:
            return HTMLResponse('<span class="text-err text-xs">Thumbnail unavailable</span>')

        image_b64 = base64.b64encode(thumb).decode()

        if target_provider_id is not None:
            description, used_pid, model_used = await engine.llm.generate_with(
                target_provider_id, prompt_text, image_b64,
            )
        else:
            description, used_pid, model_used = await engine.llm.generate(
                prompt_text, image_b64,
            )

        from engine import compute_repetition_score
        quality_score = compute_repetition_score(description)
        threshold = float(state_db.get_config("repetition_threshold", "0.3"))
        score_color = "#f87171" if quality_score > threshold else "#5a5862"

        duration = _time.monotonic() - start
        dur_str = f"{duration:.1f}s"
        model_str = escape(model_used or "?")

        return HTMLResponse(f"""
        <div style="background:#161925;border:1px solid #252836;border-radius:8px;padding:12px;margin-top:4px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="color:#5a5862;font-size:10px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:0.08em">#{used_pid} {model_str}</span>
                <span style="color:{score_color};font-size:11px;font-family:'JetBrains Mono',monospace">score: {quality_score:.2f}</span>
                <span style="color:#5a5862;font-size:11px;font-family:'JetBrains Mono',monospace">{dur_str}</span>
            </div>
            <p style="color:#c8c6c3;font-size:13px;line-height:1.6;margin:0">{escape(description)}</p>
        </div>""")

    except Exception as e:
        return HTMLResponse(f'<span class="text-err text-xs">Error: {escape(str(e))}</span>')


@app.post("/run/reset-failed")
async def reset_failed():
    state_db.reset_failed()
    return JSONResponse({"status": "ok"})


_audit_task: asyncio.Task = None
_running_audit_engine: ProcessingEngine = None


@app.post("/audit/start")
async def start_audit():
    global _audit_task, _running_audit_engine
    if _audit_task and not _audit_task.done():
        return JSONResponse({"error": "Audit already running"}, status_code=409)
    if _task and not _task.done():
        return JSONResponse({"error": "Batch running — wait for it to finish"}, status_code=409)
    build_engine()
    _running_audit_engine = engine
    _audit_task = asyncio.create_task(engine.run_audit())
    return JSONResponse({"status": "started"})


@app.get("/audit/status", response_class=HTMLResponse)
async def audit_status():
    if _audit_task and not _audit_task.done():
        active = _running_audit_engine or engine
        msg = active.progress.message if active else "Running..."
        return HTMLResponse(
            f'<div class="flex items-center gap-2">'
            f'<div class="dot dot-warn dot-pulse"></div>'
            f'<span class="text-txt-2 text-xs">{escape(msg)}</span>'
            f'</div>'
        )
    if _audit_task and _audit_task.done():
        try:
            result = _audit_task.result()
            return HTMLResponse(
                f'<span class="text-ok text-xs font-mono">'
                f'Audit done: {result["scored"]} scored, {result["flagged"]} flagged</span>'
            )
        except Exception as e:
            return HTMLResponse(
                f'<span class="text-err text-xs">Error: {escape(str(e))}</span>'
            )
    return HTMLResponse("")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
