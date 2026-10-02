"""Processing engine for Immich custom AI descriptions.

Components:
- StateDB: SQLite state tracking (config, processed, runs, metadata)
- ImmichClient: Immich API + postgres OCR queries
- LLMPool: Multi-endpoint round-robin with semaphores and health checks
- PromptBuilder: Context-enriched prompt assembly (EXIF, people, OCR, nearby)
- ProcessingEngine: Orchestrates asset processing with progress tracking
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("custom-ai-desc")

MONTHS_PT = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril",
    5: "maio", 6: "junho", 7: "julho", 8: "agosto",
    9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro",
}

DEFAULT_PROMPT = (
    "Crie uma descrição detalhada da imagem em português brasileiro para "
    "funcionalidade de busca. Na resposta, forneça apenas a descrição sem "
    "palavras introdutórias. Também especifique o formato da imagem "
    "(Papel de parede, Captura de tela, Desenho, Foto de cidade, Selfie, "
    "Foto, etc.) no final da descrição."
)

DEFAULT_CONFIG = {
    "prompt_template": DEFAULT_PROMPT,
    "overwrite_existing": "false",
    "quality_gate_enabled": "true",
    "repetition_threshold": "0.3",
    "min_words": "10",
    "max_words": "300",
    "watch_interval_hours": "24",
    "ocr_min_score": "0.5",
    "max_people": "8",
    "nearby_window_minutes": "10",
}


class ProviderDisabledError(Exception):
    """Raised when a provider should be permanently disabled (model not found)."""
    pass


class ProviderTransientError(Exception):
    """Raised when a provider is temporarily unavailable (connection, timeout, 5xx).

    Attributes:
        server_error: True if the server responded with 5xx (up but can't serve).
                      False for connection/timeout errors (server unreachable).
    """

    def __init__(self, message: str, server_error: bool = False):
        super().__init__(message)
        self.server_error = server_error


# --- Helpers ---

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from qwen3 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def desc_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sanitize(text: str) -> str:
    """Strip null bytes and control characters."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def compute_repetition_score(text: str) -> float:
    """Score text repetition 0.0 (unique) to 1.0 (fully repetitive).

    Uses trigram (3-word sliding window) duplicate ratio.
    Short texts (< 6 words) return 0.0 — not enough data to judge.
    """
    words = text.lower().split()
    if len(words) < 6:
        return 0.0
    n = 3
    trigrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    unique = len(set(trigrams))
    return 1.0 - (unique / len(trigrams))


# --- StateDB ---

class StateDB:
    """SQLite state tracking for processed assets, config, and run history."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._fernet = self._load_or_create_secret(Path(db_path).parent / "secret.key")
        self._init_schema()

    @staticmethod
    def _load_or_create_secret(path: Path) -> Fernet:
        """Load the Fernet key used to encrypt provider API keys, generating it on first run.

        O_EXCL ensures two racing startups can't both "win" the create — the loser just
        falls through to the read. Failing here (e.g. read-only volume) is preferable to
        failing later mid-request.
        """
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, Fernet.generate_key())
            finally:
                os.close(fd)
        except FileExistsError:
            pass
        return Fernet(path.read_bytes())

    def _encrypt_api_key(self, plain: str) -> str:
        return self._fernet.encrypt(plain.encode()).decode()

    def _decrypt_api_key(self, ciphertext: str) -> Optional[str]:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken:
            logger.error("Could not decrypt stored provider API key (secret.key changed?) — treating as unset")
            return None

    def _encode_api_key(self, plain: Optional[str]) -> tuple:
        """Returns (encrypted, last4) for storage, or (None, None) if no key given."""
        if not plain:
            return None, None
        return self._encrypt_api_key(plain), plain[-4:]

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processed (
                asset_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                description_hash TEXT,
                description_preview TEXT,
                processed_at TEXT,
                endpoint_used TEXT,
                duration_ms INTEGER,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                finished_at TEXT,
                mode TEXT,
                total INTEGER,
                done INTEGER,
                failed INTEGER,
                skipped INTEGER
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                model TEXT NOT NULL,
                concurrency INTEGER DEFAULT 4,
                timeout INTEGER DEFAULT 120,
                max_tokens INTEGER DEFAULT 500,
                temperature REAL DEFAULT 0.7,
                top_k INTEGER DEFAULT 40,
                top_p REAL DEFAULT 0.95,
                min_p REAL DEFAULT 0.05,
                repeat_penalty REAL DEFAULT 1.1,
                enabled INTEGER DEFAULT 1
            );
        """)
        self.conn.execute(
            "INSERT OR IGNORE INTO metadata (key, value) VALUES ('schema_version', '1')"
        )
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        version = int(row["value"]) if row else 1

        if version < 2:
            for col in ("provider_id INTEGER", "model_used TEXT", "prompt_used TEXT"):
                try:
                    self.conn.execute(f"ALTER TABLE processed ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            self.conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', '2')"
            )
            self.conn.commit()
            logger.info("Migrated schema to v2 (provider columns in processed)")

        if version < 3:
            try:
                self.conn.execute(
                    "ALTER TABLE processed ADD COLUMN quality_score REAL"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
            self.conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) "
                "VALUES ('schema_version', '3')"
            )
            self.conn.commit()
            logger.info("Migrated schema to v3 (quality_score column)")

        if version < 4:
            for col in ("api_key_encrypted TEXT", "api_key_last4 TEXT", "extra_params TEXT"):
                try:
                    self.conn.execute(f"ALTER TABLE providers ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass  # column already exists

            # Backfill extra_params from the old dedicated sampling columns so
            # existing providers keep sending identical request bodies.
            rows = self.conn.execute(
                "SELECT id, top_k, top_p, min_p, repeat_penalty FROM providers "
                "WHERE extra_params IS NULL"
            ).fetchall()
            for row in rows:
                params = {
                    "top_k": row["top_k"],
                    "top_p": row["top_p"],
                    "min_p": row["min_p"],
                    "repeat_penalty": row["repeat_penalty"],
                    "reasoning_format": "none",
                }
                self.conn.execute(
                    "UPDATE providers SET extra_params = ? WHERE id = ?",
                    (json.dumps(params), row["id"]),
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) "
                "VALUES ('schema_version', '4')"
            )
            self.conn.commit()
            logger.info("Migrated schema to v4 (api_key + extra_params columns)")

    def get_config(self, key: str, default: str = None) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_config(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_all_config(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM config").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def has_config(self) -> bool:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM config").fetchone()
        return row["cnt"] > 0

    def save_defaults(self):
        for key, value in DEFAULT_CONFIG.items():
            self.conn.execute(
                "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
                (key, value),
            )
        self.conn.commit()

    def is_processed(self, asset_id: str) -> Optional[str]:
        """Returns status if processed, None otherwise."""
        row = self.conn.execute(
            "SELECT status FROM processed WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        return row["status"] if row else None

    def get_processed_ids(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT asset_id FROM processed WHERE status IN ('done', 'skipped')"
        ).fetchall()
        return {r["asset_id"] for r in rows}

    def mark_done(self, asset_id: str, d_hash: str, preview: str,
                  endpoint: str, duration_ms: int,
                  provider_id: int = None, model_used: str = None,
                  prompt_used: str = None, quality_score: float = None):
        self.conn.execute(
            """INSERT OR REPLACE INTO processed
               (asset_id, status, description_hash, description_preview,
                processed_at, endpoint_used, duration_ms, error,
                provider_id, model_used, prompt_used, quality_score)
               VALUES (?, 'done', ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)""",
            (asset_id, d_hash, preview[:200],
             datetime.now().isoformat(), endpoint, duration_ms,
             provider_id, model_used, prompt_used, quality_score),
        )
        self.conn.commit()

    def mark_failed(self, asset_id: str, error: str,
                    endpoint: str = None, duration_ms: int = None,
                    provider_id: int = None, model_used: str = None):
        self.conn.execute(
            """INSERT OR REPLACE INTO processed
               (asset_id, status, description_hash, description_preview,
                processed_at, endpoint_used, duration_ms, error,
                provider_id, model_used, prompt_used)
               VALUES (?, 'failed', NULL, NULL, ?, ?, ?, ?, ?, ?, NULL)""",
            (asset_id, datetime.now().isoformat(), endpoint, duration_ms, error,
             provider_id, model_used),
        )
        self.conn.commit()

    def mark_skipped(self, asset_id: str, reason: str):
        self.conn.execute(
            """INSERT OR REPLACE INTO processed
               (asset_id, status, description_hash, description_preview,
                processed_at, endpoint_used, duration_ms, error)
               VALUES (?, 'skipped', NULL, NULL, ?, NULL, NULL, ?)""",
            (asset_id, datetime.now().isoformat(), reason),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM processed GROUP BY status"
        ).fetchall()
        stats = {"done": 0, "failed": 0, "skipped": 0}
        for r in rows:
            stats[r["status"]] = r["cnt"]
        stats["total_processed"] = sum(stats.values())
        return stats

    def get_recent(self, limit: int = 50, status: str = None) -> list[dict]:
        if status:
            rows = self.conn.execute(
                """SELECT * FROM processed WHERE status = ?
                   ORDER BY processed_at DESC LIMIT ?""",
                (status, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM processed ORDER BY processed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def start_run(self, mode: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO runs (started_at, mode) VALUES (?, ?)",
            (datetime.now().isoformat(), mode),
        )
        self.conn.commit()
        return cursor.lastrowid

    def finish_run(self, run_id: int, total: int, done: int,
                   failed: int, skipped: int):
        self.conn.execute(
            """UPDATE runs SET finished_at = ?, total = ?, done = ?,
               failed = ?, skipped = ? WHERE id = ?""",
            (datetime.now().isoformat(), total, done, failed, skipped, run_id),
        )
        self.conn.commit()

    def get_runs(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_last_run_timestamp(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_run_timestamp'"
        ).fetchone()
        return row["value"] if row else None

    def set_last_run_timestamp(self, ts: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) "
            "VALUES ('last_run_timestamp', ?)",
            (ts,),
        )
        self.conn.commit()

    def reset_failed(self):
        self.conn.execute("DELETE FROM processed WHERE status = 'failed'")
        self.conn.commit()

    def get_asset(self, asset_id: str) -> Optional[dict]:
        """Get a single processed asset by ID."""
        row = self.conn.execute(
            "SELECT * FROM processed WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_unscored_done_ids(self) -> list[str]:
        """Get asset IDs that are done but have no quality score."""
        rows = self.conn.execute(
            "SELECT asset_id FROM processed "
            "WHERE status = 'done' AND quality_score IS NULL"
        ).fetchall()
        return [r["asset_id"] for r in rows]

    def update_quality_score(self, asset_id: str, score: float):
        """Set quality score for an already-processed asset."""
        self.conn.execute(
            "UPDATE processed SET quality_score = ? WHERE asset_id = ?",
            (score, asset_id),
        )
        self.conn.commit()

    def fail_from_audit(self, asset_id: str, score: float,
                        reason: str = None):
        """Mark a done asset as failed due to quality audit."""
        error_msg = reason or f"quality_audit: score={score:.2f}"
        self.conn.execute(
            """UPDATE processed SET status = 'failed',
               quality_score = ?,
               error = ?
               WHERE asset_id = ?""",
            (score, error_msg,
             asset_id),
        )
        self.conn.commit()

    def clear_all(self):
        self.conn.execute("DELETE FROM processed")
        self.conn.commit()

    # --- Provider CRUD ---

    def _prepare_provider(self, p: dict, include_secrets: bool) -> dict:
        """Strip ciphertext from the dict; decrypt into plaintext `api_key` only when asked."""
        p["api_key"] = None
        if include_secrets and p.get("api_key_encrypted"):
            p["api_key"] = self._decrypt_api_key(p["api_key_encrypted"])
        p.pop("api_key_encrypted", None)
        return p

    def get_providers(self, enabled_only: bool = False, include_secrets: bool = False) -> list[dict]:
        if enabled_only:
            rows = self.conn.execute(
                "SELECT * FROM providers WHERE enabled = 1"
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM providers").fetchall()
        return [self._prepare_provider(dict(r), include_secrets) for r in rows]

    def get_provider(self, provider_id: int, include_secrets: bool = False) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM providers WHERE id = ?", (provider_id,)
        ).fetchone()
        return self._prepare_provider(dict(row), include_secrets) if row else None

    def add_provider(self, data: dict) -> int:
        api_key_encrypted, api_key_last4 = self._encode_api_key(data.get("api_key"))
        cursor = self.conn.execute(
            """INSERT INTO providers (url, model, concurrency, timeout, max_tokens,
               temperature, enabled, api_key_encrypted, api_key_last4, extra_params)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["url"], data["model"],
                data.get("concurrency", 4), data.get("timeout", 120),
                data.get("max_tokens", 500), data.get("temperature", 0.7),
                data.get("enabled", 1),
                api_key_encrypted, api_key_last4,
                data.get("extra_params", "{}"),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_provider(self, provider_id: int, data: dict):
        fields = []
        values = []
        for key in ("url", "model", "concurrency", "timeout", "max_tokens",
                     "temperature", "enabled", "extra_params"):
            if key in data:
                fields.append(f"{key} = ?")
                values.append(data[key])

        # api_key handling: blank field (key absent from `data`) keeps the existing
        # key untouched. A typed key always wins over "clear_api_key" if both are set.
        if data.get("api_key"):
            encrypted, last4 = self._encode_api_key(data["api_key"])
            fields += ["api_key_encrypted = ?", "api_key_last4 = ?"]
            values += [encrypted, last4]
        elif data.get("clear_api_key"):
            fields += ["api_key_encrypted = ?", "api_key_last4 = ?"]
            values += [None, None]

        if fields:
            values.append(provider_id)
            self.conn.execute(
                f"UPDATE providers SET {', '.join(fields)} WHERE id = ?", values
            )
            self.conn.commit()

    def delete_provider(self, provider_id: int):
        self.conn.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
        self.conn.commit()

    # Async-safe write methods (for concurrent batch processing)

    async def async_mark_done(self, asset_id: str, d_hash: str, preview: str,
                              endpoint: str, duration_ms: int,
                              provider_id: int = None, model_used: str = None,
                              prompt_used: str = None, quality_score: float = None):
        async with self._lock:
            self.mark_done(asset_id, d_hash, preview, endpoint, duration_ms,
                           provider_id, model_used, prompt_used, quality_score)

    async def async_mark_failed(self, asset_id: str, error: str,
                                endpoint: str = None, duration_ms: int = None,
                                provider_id: int = None, model_used: str = None):
        async with self._lock:
            self.mark_failed(asset_id, error, endpoint, duration_ms,
                             provider_id, model_used)

    async def async_mark_skipped(self, asset_id: str, reason: str):
        async with self._lock:
            self.mark_skipped(asset_id, reason)

    def close(self):
        self.conn.close()


# --- ImmichClient ---

class ImmichClient:
    """Async client for Immich API and database."""

    def __init__(self, api_url: str, api_key: str, db_url: str = None):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.db_url = db_url
        self.client = httpx.AsyncClient(
            headers={"x-api-key": api_key},
            timeout=httpx.Timeout(30.0, read=60.0),
        )
        self._db_conn = None

    def _get_db(self):
        """Lazy single postgres connection for OCR queries."""
        if self._db_conn is None and self.db_url:
            try:
                import psycopg
                self._db_conn = psycopg.connect(self.db_url)
                logger.info("Connected to Immich postgres for OCR queries")
            except Exception as e:
                logger.warning(f"Postgres connection failed (OCR disabled): {e}")
        return self._db_conn

    async def get_all_image_assets(self, since: str = None) -> list[dict]:
        """Paginate through all image assets."""
        all_assets = []
        page = 1
        while True:
            body: dict = {
                "type": "IMAGE",
                "size": 1000,
                "page": page,
                "withExif": True,
            }
            if since:
                body["createdAfter"] = since

            resp = await self.client.post(
                f"{self.api_url}/search/metadata", json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("assets", {}).get("items", [])
            all_assets.extend(items)

            next_page = data.get("assets", {}).get("nextPage")
            if not next_page:
                break
            page = int(next_page)
            logger.info(f"Fetched page {page - 1}, {len(all_assets)} assets so far")

        return all_assets

    async def get_asset(self, asset_id: str) -> dict:
        resp = await self.client.get(f"{self.api_url}/assets/{asset_id}")
        resp.raise_for_status()
        return resp.json()

    async def get_thumbnail(self, asset_id: str) -> Optional[bytes]:
        try:
            resp = await self.client.get(
                f"{self.api_url}/assets/{asset_id}/thumbnail",
                params={"size": "preview"},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.warning(f"Thumbnail failed for {asset_id}: {e}")
            return None

    async def update_description(self, asset_id: str, description: str):
        resp = await self.client.put(
            f"{self.api_url}/assets/{asset_id}",
            json={"description": description},
        )
        resp.raise_for_status()

    async def get_nearby_count(self, asset: dict,
                               window_minutes: int = 10) -> int:
        """Count assets taken within +/-window_minutes of this asset."""
        exif = asset.get("exifInfo") or {}
        dt_str = exif.get("dateTimeOriginal")
        if not dt_str:
            return 0
        # Skip for screenshots (no camera make)
        if not exif.get("make"):
            return 0
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            after = (dt - timedelta(minutes=window_minutes)).isoformat()
            before = (dt + timedelta(minutes=window_minutes)).isoformat()

            resp = await self.client.post(
                f"{self.api_url}/search/metadata",
                json={
                    "type": "IMAGE",
                    "takenAfter": after,
                    "takenBefore": before,
                    "size": 100,
                    "page": 1,
                },
            )
            resp.raise_for_status()
            items = resp.json().get("assets", {}).get("items", [])
            return max(0, len(items) - 1)
        except Exception as e:
            logger.debug(f"Nearby count failed: {e}")
            return 0

    def get_ocr_text(self, asset_id: str, min_score: float = 0.5) -> Optional[str]:
        """Query OCR text from Immich postgres via asset_ocr table."""
        conn = self._get_db()
        if not conn:
            return None
        try:
            cur = conn.execute(
                """SELECT string_agg(text, ' ' ORDER BY y1, x1) as combined
                   FROM asset_ocr
                   WHERE "assetId" = %s::uuid AND "textScore" > %s
                   GROUP BY "assetId" """,
                (asset_id, min_score),
            )
            row = cur.fetchone()
            if row and row[0]:
                text = sanitize(row[0].strip())
                return text[:500] if text else None
            return None
        except Exception as e:
            logger.debug(f"OCR query failed for {asset_id}: {e}")
            return None

    async def close(self):
        await self.client.aclose()
        if self._db_conn:
            self._db_conn.close()


# --- LLMPool ---

def build_chat_body(provider: dict, prompt: str, image_b64: str) -> dict:
    """Builds the OpenAI-compatible chat completion request body for a provider.

    Only `model`/`messages`/`max_tokens`/`temperature`/`stream` are fixed. Everything
    else (top_k, top_p, min_p, repeat_penalty, reasoning_format, or any other
    provider-specific field) comes from the provider's freeform `extra_params` JSON
    blob, merged in last — this is what lets a provider that rejects a given param
    (e.g. OpenAI rejecting `repeat_penalty`) simply omit it instead of the app
    hardcoding a one-size-fits-all param set.
    """
    body = {
        "model": provider["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                }},
            ],
        }],
        "max_tokens": provider.get("max_tokens", 500),
        "temperature": provider.get("temperature", 0.7),
        "stream": False,
    }
    extra = provider.get("extra_params")
    if extra:
        try:
            body.update(json.loads(extra))
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                f"Provider {provider.get('id')}: invalid extra_params JSON, ignoring"
            )
    return body


def auth_headers(provider: dict) -> dict:
    """Authorization header for a provider's API key, read fresh each call.

    Not baked into the httpx client at construction time — that would mean a
    key added/changed via Settings never takes effect until the client is
    rebuilt, which nothing currently does for an already-known provider id.
    """
    key = provider.get("api_key")
    return {"Authorization": f"Bearer {key}"} if key else {}


class LLMPool:
    """Per-provider LLM pool with round-robin, semaphores, and health checks."""

    def __init__(self, providers: list[dict]):
        self.providers = providers
        self._semaphores = {
            p["id"]: asyncio.Semaphore(p.get("concurrency", 4))
            for p in providers
        }
        self._disabled: set[int] = set()
        self._counter = 0
        self._clients = {
            p["id"]: httpx.AsyncClient(
                timeout=httpx.Timeout(float(p.get("timeout", 120)), connect=10.0),
            )
            for p in providers
        }

    def _next_provider(self) -> Optional[dict]:
        available = [p for p in self.providers if p["id"] not in self._disabled]
        if not available:
            return None
        p = available[self._counter % len(available)]
        self._counter += 1
        return p

    async def generate(self, prompt: str, image_b64: str) -> tuple[str, int, str]:
        """Send vision request. Returns (response_text, provider_id, model_used).

        Tries providers round-robin. Disables on model-not-found.
        Raises RuntimeError if all exhausted.
        """
        tried: set[int] = set()
        last_error = None

        while True:
            provider = self._next_provider()
            if provider is None or provider["id"] in tried:
                raise RuntimeError(
                    f"All LLM providers failed. Last error: {last_error}"
                )
            tried.add(provider["id"])

            try:
                async with self._semaphores[provider["id"]]:
                    text = await self._call(provider, prompt, image_b64)
                    return text, provider["id"], provider["model"]
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Provider {provider['id']} ({provider['url']}) failed: {e}")
                err_lower = str(e).lower()
                if ("model" in err_lower and "not found" in err_lower) or \
                   ("404" in err_lower and "model" in err_lower):
                    logger.error(f"Disabling provider {provider['id']}: model not found")
                    self._disabled.add(provider["id"])
                elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403):
                    logger.error(f"Disabling provider {provider['id']}: auth error ({e.response.status_code})")
                    self._disabled.add(provider["id"])

    async def generate_with(self, provider_id: int, prompt: str,
                            image_b64: str) -> tuple[str, int, str]:
        """Generate using a specific provider. Used by per-provider workers.

        Raises:
            ProviderDisabledError: model not found (permanent)
            ProviderTransientError: connection/timeout/5xx (temporary)
            Other exceptions: propagate normally (asset-level errors)
        """
        provider = next(p for p in self.providers if p["id"] == provider_id)
        try:
            async with self._semaphores[provider_id]:
                text = await self._call(provider, prompt, image_b64)
                return text, provider_id, provider["model"]
        except Exception as e:
            err_lower = str(e).lower()
            # Permanent: model not found (string match)
            if ("model" in err_lower and "not found" in err_lower) or \
               ("404" in err_lower and "model" in err_lower):
                self._disabled.add(provider_id)
                raise ProviderDisabledError(str(e)) from e
            # Permanent: any 404 from LLM API is almost always model-not-found
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
                self._disabled.add(provider_id)
                raise ProviderDisabledError(str(e)) from e
            # Permanent: bad/missing/expired API key — won't fix itself on retry
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403):
                self._disabled.add(provider_id)
                raise ProviderDisabledError(str(e)) from e
            # Transient: rate limited — back off and retry, don't disable
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                raise ProviderTransientError(str(e)) from e
            # Transient: connection errors, timeouts
            if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout,
                              httpx.ReadTimeout, httpx.WriteTimeout,
                              httpx.PoolTimeout, ConnectionError, OSError)):
                raise ProviderTransientError(str(e)) from e
            # Transient: 5xx server errors (server up but can't serve)
            if isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code >= 500:
                    raise ProviderTransientError(str(e), server_error=True) from e
            raise  # asset-level error — let process_asset handle it

    async def _call(self, provider: dict, prompt: str, image_b64: str) -> str:
        url = provider["url"].rstrip("/")
        body = build_chat_body(provider, prompt, image_b64)

        client = self._clients[provider["id"]]
        resp = await client.post(
            f"{url}/chat/completions", json=body, headers=auth_headers(provider),
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        finish = choice.get("finish_reason", "")
        text = choice.get("message", {}).get("content", "")

        if finish == "length":
            logger.warning(f"Response truncated (hit max_tokens) from provider {provider['id']}")

        text = strip_thinking(text).strip()
        if not text:
            raise ValueError("Empty response from LLM")

        return text

    async def health_check(self) -> dict[int, dict]:
        """Returns provider_id -> {url, model, online}."""
        results = {}
        for p in self.providers:
            url = p["url"].rstrip("/")
            try:
                client = self._clients[p["id"]]
                resp = await client.get(f"{url}/models", timeout=5.0, headers=auth_headers(p))
                results[p["id"]] = {"url": url, "model": p["model"], "online": resp.status_code == 200}
            except Exception:
                results[p["id"]] = {"url": url, "model": p["model"], "online": False}
        return results

    async def check_provider_health(self, provider_id: int) -> bool:
        """Quick reachability check — is the server responding at all?

        Uses GET /models (cheap, no inference).  This is appropriate for
        connection-level failures (server down, network issues).  For 5xx
        errors (server up but can't serve, e.g. GPU VRAM exhausted), the
        worker uses timed backoff instead of health-checking.
        """
        provider = next((p for p in self.providers if p["id"] == provider_id), None)
        if not provider:
            return False
        url = provider["url"].rstrip("/")
        try:
            client = self._clients[provider_id]
            resp = await client.get(f"{url}/models", timeout=5.0, headers=auth_headers(provider))
            return resp.status_code == 200
        except Exception:
            return False

    def add_provider(self, provider: dict):
        """Hot-add a provider to the pool (for supervisor use during batch)."""
        pid = provider["id"]
        if pid in self._clients:
            return  # already exists
        self.providers.append(provider)
        self._semaphores[pid] = asyncio.Semaphore(provider.get("concurrency", 4))
        self._clients[pid] = httpx.AsyncClient(
            timeout=httpx.Timeout(float(provider.get("timeout", 120)), connect=10.0),
        )

    async def close(self):
        for client in self._clients.values():
            await client.aclose()


# --- PromptBuilder ---

class PromptBuilder:
    """Builds context-enriched prompts from EXIF, people, OCR, and nearby data."""

    def __init__(self, template: str = None):
        self.template = template or DEFAULT_PROMPT

    def build(self, asset: dict, ocr_text: str = None,
              nearby_count: int = 0, people: list[str] = None) -> str:
        exif = asset.get("exifInfo") or {}
        lines = []

        loc = self._format_location(exif)
        if loc:
            lines.append(f"- {loc}")

        dt = self._format_datetime(exif)
        if dt:
            lines.append(f"- {dt}")

        cam = self._format_camera(exif)
        if cam:
            lines.append(f"- {cam}")

        res = self._format_resolution(exif)
        if res:
            lines.append(f"- {res}")

        if people:
            max_p = 8
            names = [n.split()[0] for n in people[:max_p]]
            people_str = ", ".join(names)
            if len(people) > max_p:
                people_str += " e outros"
            lines.append(f"- Pessoas reconhecidas: {people_str}")

        if ocr_text:
            lines.append(f'- Texto detectado na imagem: "{ocr_text}"')

        if nearby_count > 1:
            lines.append(
                f"- Contexto: parte de uma sequência de {nearby_count + 1} "
                f"fotos tiradas no mesmo período"
            )

        if lines:
            context = "Contexto da foto:\n" + "\n".join(lines)
            return f"{context}\n\n{self.template}"
        return self.template

    def _format_location(self, exif: dict) -> Optional[str]:
        city = exif.get("city")
        state = exif.get("state")
        country = exif.get("country")
        lat = exif.get("latitude")
        lon = exif.get("longitude")

        # Bad GPS filter (null island)
        if lat is not None and lon is not None:
            if abs(lat) < 1 and abs(lon) < 1:
                lat = lon = None

        parts = [p for p in [city, state, country] if p]
        if not parts and lat is None:
            return None

        loc = "Local: " + ", ".join(parts) if parts else "Local:"
        if lat is not None and lon is not None:
            loc += f" ({lat:.2f}, {lon:.2f})"
        return loc

    def _format_datetime(self, exif: dict) -> Optional[str]:
        dt_str = exif.get("dateTimeOriginal")
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            # Apply timezone offset if available
            tz_str = exif.get("timeZone")
            if tz_str and tz_str.startswith("UTC"):
                try:
                    offset = tz_str[3:]
                    if offset:
                        sign = 1 if offset[0] == "+" else -1
                        hours = int(offset.lstrip("+-"))
                        dt = dt.astimezone(timezone(timedelta(hours=sign * hours)))
                except (ValueError, IndexError):
                    pass

            month = MONTHS_PT.get(dt.month, str(dt.month))
            hour = dt.hour
            if 0 <= hour < 6:
                period = "madrugada"
            elif 6 <= hour < 12:
                period = "manhã"
            elif 12 <= hour < 18:
                period = "tarde"
            else:
                period = "noite"

            return (
                f"Data: {dt.day} de {month} de {dt.year}, "
                f"{dt.hour:02d}:{dt.minute:02d} ({period})"
            )
        except Exception:
            return None

    def _format_camera(self, exif: dict) -> Optional[str]:
        make = exif.get("make")
        model = exif.get("model")
        if not make and not model:
            return None

        cam_parts = [p for p in [make, model] if p]
        cam = "Camera: " + " ".join(cam_parts)

        specs = []
        if exif.get("fNumber"):
            specs.append(f"f/{exif['fNumber']}")
        if exif.get("exposureTime"):
            specs.append(f"{exif['exposureTime']}s")
        if exif.get("iso"):
            specs.append(f"ISO {exif['iso']}")
        if exif.get("focalLength"):
            specs.append(f"{exif['focalLength']}mm")

        if specs:
            cam += f" ({', '.join(specs)})"
        return cam

    def _format_resolution(self, exif: dict) -> Optional[str]:
        w = exif.get("exifImageWidth")
        h = exif.get("exifImageHeight")
        if w and h:
            return f"Resolução: {w}x{h}"
        return None


# --- ProcessingEngine ---

class ProgressState:
    """Mutable progress tracker polled by the web UI."""

    def __init__(self):
        self.total = 0
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.current_asset = ""
        self.running = False
        self.started_at: Optional[str] = None
        self.message = ""
        self.rate = 0.0  # assets/min
        self.provider_stats: dict[int, dict] = {}  # {pid: {done, failed, rate, model, url}}


class ProcessingEngine:
    """Orchestrates batch asset processing."""

    def __init__(self, state_db: StateDB, immich: ImmichClient,
                 llm_pool: LLMPool, prompt_builder: PromptBuilder):
        self.db = state_db
        self.immich = immich
        self.llm = llm_pool
        self.prompt = prompt_builder
        self.progress = ProgressState()
        self._stop_event = asyncio.Event()

    def should_skip(self, asset: dict, overwrite: bool = False) -> Optional[str]:
        """Returns skip reason or None."""
        if asset.get("isOffline"):
            return "offline"

        # Stack child — skip non-primary assets in a stack
        stack = asset.get("stack")
        if stack:
            primary_id = stack.get("primaryAssetId")
            if primary_id and primary_id != asset["id"]:
                return "stack_child"

        if not overwrite:
            if (asset.get("exifInfo") or {}).get("description"):
                return "has_description"
            status = self.db.is_processed(asset["id"])
            if status in ("done", "skipped"):
                return "already_processed"

        return None

    async def process_asset(self, asset: dict, provider_id: int = None) -> str:
        """Process a single asset. Returns 'done', 'failed', or 'skipped'.

        Args:
            provider_id: If set, use this specific provider (worker mode).
                         If None, use round-robin via generate() (preview mode).
        """
        asset_id = asset["id"]
        start = time.monotonic()
        provider_url = None
        used_provider_id = None
        model_used = None

        try:
            # Full detail (includes people)
            detail = await self.immich.get_asset(asset_id)

            # Named people only
            max_people = int(self.db.get_config("max_people", "8"))
            people_names = []
            for p in detail.get("people", []):
                name = (p.get("name") or "").strip()
                if name:
                    people_names.append(name)
                    if len(people_names) >= max_people:
                        break

            # OCR text (sync DB call — run in thread to avoid blocking)
            ocr_min = float(self.db.get_config("ocr_min_score", "0.5"))
            ocr_text = await asyncio.to_thread(
                self.immich.get_ocr_text, asset_id, ocr_min,
            )

            # Nearby count
            window = int(self.db.get_config("nearby_window_minutes", "10"))
            nearby = await self.immich.get_nearby_count(detail, window)

            # Build prompt
            prompt_text = self.prompt.build(
                detail,
                ocr_text=ocr_text,
                nearby_count=nearby,
                people=people_names,
            )

            # Check thumbnail exists before downloading (null thumbhash = corrupt/unprocessed)
            if not detail.get("thumbhash"):
                await self.db.async_mark_failed(asset_id, "no_thumbhash")
                logger.warning(f"Skipping {asset_id[:12]}...: no thumbhash (corrupt or unprocessed asset)")
                return "failed"

            # Get thumbnail
            thumb = await self.immich.get_thumbnail(asset_id)
            if not thumb:
                await self.db.async_mark_failed(asset_id, "thumbnail_unavailable")
                return "failed"

            image_b64 = base64.b64encode(thumb).decode()

            # Generate description
            if provider_id is not None:
                description, used_provider_id, model_used = await self.llm.generate_with(
                    provider_id, prompt_text, image_b64,
                )
            else:
                description, used_provider_id, model_used = await self.llm.generate(
                    prompt_text, image_b64,
                )
            provider_url = next(
                (p["url"] for p in self.llm.providers if p["id"] == used_provider_id), ""
            )

            if not description or not description.strip():
                await self.db.async_mark_failed(
                    asset_id, "empty_response", provider_url,
                    provider_id=used_provider_id, model_used=model_used,
                )
                return "failed"

            # Quality gate
            quality_score = compute_repetition_score(description)
            gate_enabled = self.db.get_config("quality_gate_enabled", "true") == "true"

            if gate_enabled:
                threshold = float(self.db.get_config("repetition_threshold", "0.3"))
                min_words = int(self.db.get_config("min_words", "10"))
                max_words = int(self.db.get_config("max_words", "300"))
                word_count = len(description.split())
                reject_reason = None

                if quality_score > threshold:
                    reject_reason = (
                        f"quality_gate: repetition_score={quality_score:.2f} "
                        f"(threshold={threshold})"
                    )
                elif word_count < min_words:
                    reject_reason = (
                        f"quality_gate: too_short={word_count} words "
                        f"(min={min_words})"
                    )
                elif word_count > max_words:
                    reject_reason = (
                        f"quality_gate: too_long={word_count} words "
                        f"(max={max_words})"
                    )

                if reject_reason:
                    duration = int((time.monotonic() - start) * 1000)
                    await self.db.async_mark_failed(
                        asset_id, reject_reason, provider_url, duration,
                        provider_id=used_provider_id, model_used=model_used,
                    )
                    logger.warning(
                        f"Quality gate blocked {asset_id[:12]}... "
                        f"{reject_reason}"
                    )
                    return "failed"

            # Write back to Immich
            await self.immich.update_description(asset_id, description)

            duration = int((time.monotonic() - start) * 1000)
            await self.db.async_mark_done(
                asset_id, desc_hash(description), description,
                provider_url, duration,
                provider_id=used_provider_id, model_used=model_used,
                prompt_used=prompt_text, quality_score=quality_score,
            )
            count = self.progress.done + self.progress.failed + self.progress.skipped + 1
            logger.info(
                f"[{count}/{self.progress.total}] {asset_id[:12]}... "
                f"OK ({duration}ms) via provider {used_provider_id} ({model_used})"
            )
            return "done"

        except (ProviderDisabledError, ProviderTransientError):
            raise  # let worker handle provider-level errors
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            await self.db.async_mark_failed(
                asset_id, str(e)[:500], provider_url, duration,
                provider_id=used_provider_id, model_used=model_used,
            )
            logger.error(f"Failed {asset_id[:12]}...: {e}")
            return "failed"

    async def run_batch(self, resume: bool = False) -> dict:
        """Process all unprocessed assets. Returns stats dict.

        Args:
            resume: If True, ignore last_run_timestamp and fetch all assets,
                    relying on processed_ids filter to skip already-done ones.
        """
        self._stop_event.clear()
        self.progress = ProgressState()
        self.progress.running = True
        self.progress.started_at = datetime.now().isoformat()
        self.progress.message = "Fetching asset list..."

        try:
            # Fetch assets
            since = None
            if not resume:
                since = self.db.get_last_run_timestamp()

            assets = await self.immich.get_all_image_assets(since=since)
            logger.info(f"Found {len(assets)} image assets")

            # Store the actual library count from Immich (ground truth)
            if resume:  # resume fetches ALL assets, so count is accurate
                self.db.conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) "
                    "VALUES ('library_count', ?)",
                    (str(len(assets)),),
                )
                self.db.conn.commit()

            # Filter already-processed
            processed_ids = self.db.get_processed_ids()
            to_process = [a for a in assets if a["id"] not in processed_ids]

            # Apply skip filters
            final = []
            for asset in to_process:
                reason = self.should_skip(asset)
                if reason and reason != "already_processed":
                    self.db.mark_skipped(asset["id"], reason)
                    self.progress.skipped += 1
                elif not reason:
                    final.append(asset)

            self.progress.total = len(final) + self.progress.skipped
            self.progress.message = f"Processing {len(final)} assets..."
            logger.info(
                f"To process: {len(final)}, skipped: {self.progress.skipped}"
            )

            run_id = self.db.start_run("batch")
            batch_start = time.monotonic()

            # Initialize per-provider stats
            for p in self.llm.providers:
                if p["id"] not in self.llm._disabled:
                    self.progress.provider_stats[p["id"]] = {
                        "done": 0, "failed": 0, "rate": 0.0,
                        "model": p["model"], "url": p["url"],
                        "status": "active",
                    }

            # Shared work queue — workers pull as they finish
            queue: asyncio.Queue = asyncio.Queue()
            for asset in final:
                queue.put_nowait(asset)

            BACKOFF_SCHEDULE = [5, 10, 30, 60]  # seconds
            MAX_BACKOFF_ROUNDS = 5  # give up on provider after this many consecutive backoff cycles
            retry_counts: dict[str, int] = {}  # asset_id -> re-queue count

            async def worker(worker_provider_id: int):
                while True:
                    if self._stop_event.is_set():
                        return
                    if worker_provider_id in self.llm._disabled:
                        if worker_provider_id in self.progress.provider_stats:
                            self.progress.provider_stats[worker_provider_id]["status"] = "disabled"
                        return

                    try:
                        asset = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                    self.progress.current_asset = asset["id"]
                    try:
                        result = await self.process_asset(
                            asset, provider_id=worker_provider_id,
                        )
                    except ProviderDisabledError:
                        logger.error(
                            f"Provider {worker_provider_id} disabled, "
                            f"stopping its workers"
                        )
                        queue.put_nowait(asset)
                        if worker_provider_id in self.progress.provider_stats:
                            self.progress.provider_stats[worker_provider_id]["status"] = "disabled"
                        return
                    except ProviderTransientError as e:
                        logger.warning(
                            f"Provider {worker_provider_id} transient error "
                            f"({'5xx' if e.server_error else 'connection'}): {e}"
                        )
                        queue.put_nowait(asset)

                        if worker_provider_id in self.progress.provider_stats:
                            self.progress.provider_stats[worker_provider_id]["status"] = "backoff"

                        if e.server_error:
                            # Server responded with 5xx — it's up but can't serve
                            # (e.g. GPU VRAM exhausted). Health-checking is useless
                            # here (GET /models returns 200 anyway). Just wait with
                            # longer delays and let the resource contention resolve.
                            SERVER_BACKOFF = [30, 60, 120, 300]  # seconds
                            attempt = 0
                            while not self._stop_event.is_set() and attempt < MAX_BACKOFF_ROUNDS:
                                delay = SERVER_BACKOFF[min(attempt, len(SERVER_BACKOFF) - 1)]
                                logger.info(
                                    f"Provider {worker_provider_id} server error, "
                                    f"waiting {delay}s (attempt {attempt + 1}/{MAX_BACKOFF_ROUNDS})"
                                )
                                try:
                                    await asyncio.wait_for(
                                        self._stop_event.wait(), timeout=delay,
                                    )
                                    return  # stop event was set
                                except asyncio.TimeoutError:
                                    pass
                                attempt += 1
                            # After waiting, just resume and let the next attempt
                            # either succeed or fail again naturally
                            if not self._stop_event.is_set():
                                if worker_provider_id in self.progress.provider_stats:
                                    self.progress.provider_stats[worker_provider_id]["status"] = "active"
                                continue
                            return
                        else:
                            # Connection error — server unreachable. Health-check
                            # until it comes back or we exhaust retries.
                            recovered = False
                            attempt = 0
                            while not self._stop_event.is_set() and attempt < MAX_BACKOFF_ROUNDS:
                                delay = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
                                try:
                                    await asyncio.wait_for(
                                        self._stop_event.wait(), timeout=delay,
                                    )
                                    return
                                except asyncio.TimeoutError:
                                    pass
                                healthy = await self.llm.check_provider_health(
                                    worker_provider_id,
                                )
                                if healthy:
                                    logger.info(
                                        f"Provider {worker_provider_id} recovered"
                                    )
                                    recovered = True
                                    break
                                attempt += 1

                            if not recovered and not self._stop_event.is_set():
                                logger.warning(
                                    f"Provider {worker_provider_id} exhausted "
                                    f"{MAX_BACKOFF_ROUNDS} backoff rounds, "
                                    f"worker exiting"
                                )
                                if worker_provider_id in self.progress.provider_stats:
                                    self.progress.provider_stats[worker_provider_id]["status"] = "exhausted"
                                return

                            if recovered:
                                if worker_provider_id in self.progress.provider_stats:
                                    self.progress.provider_stats[worker_provider_id]["status"] = "active"
                                continue
                            return

                    # Normal result handling
                    if result == "done":
                        self.progress.done += 1
                        self.progress.provider_stats[worker_provider_id]["done"] += 1
                    elif result == "failed":
                        self.progress.failed += 1
                        self.progress.provider_stats[worker_provider_id]["failed"] += 1

                    elapsed = time.monotonic() - batch_start
                    total_done = self.progress.done + self.progress.failed
                    if elapsed > 0:
                        self.progress.rate = (total_done / elapsed) * 60
                        ps = self.progress.provider_stats[worker_provider_id]
                        ps["rate"] = ((ps["done"] + ps["failed"]) / elapsed) * 60

            # Spawn N workers per provider (N = concurrency setting)
            worker_tasks = []
            for p in self.llm.providers:
                if p["id"] in self.llm._disabled:
                    continue
                for _ in range(p.get("concurrency", 4)):
                    worker_tasks.append(asyncio.create_task(worker(p["id"])))

            # Track active worker task set for supervisor to extend
            active_workers: set[asyncio.Task] = set(worker_tasks)

            async def supervisor():
                """Periodically check for provider changes and manage workers."""
                known_provider_ids = {
                    p["id"] for p in self.llm.providers
                }
                while True:
                    if self._stop_event.is_set():
                        return
                    await asyncio.sleep(30)
                    if self._stop_event.is_set():
                        return

                    try:
                        all_db_providers = self.db.get_providers(include_secrets=True)
                    except Exception:
                        continue

                    enabled_ids = {
                        p["id"] for p in all_db_providers if p.get("enabled")
                    }
                    db_by_id = {p["id"]: p for p in all_db_providers}

                    # Detect providers toggled OFF → disable their workers
                    for pid in list(known_provider_ids):
                        if pid not in enabled_ids and pid not in self.llm._disabled:
                            logger.info(
                                f"Supervisor: provider #{pid} disabled in settings, "
                                f"stopping its workers"
                            )
                            self.llm._disabled.add(pid)
                            if pid in self.progress.provider_stats:
                                self.progress.provider_stats[pid]["status"] = "disabled"

                    # Detect providers toggled ON or new → spawn workers
                    for p in all_db_providers:
                        if not p.get("enabled"):
                            continue
                        pid = p["id"]

                        # Re-enabled: was disabled, now enabled again
                        if pid in self.llm._disabled and pid in known_provider_ids:
                            logger.info(
                                f"Supervisor: provider #{pid} re-enabled, "
                                f"spawning workers"
                            )
                            self.llm._disabled.discard(pid)
                            # Update provider config in pool (URL/model may have changed)
                            for i, existing in enumerate(self.llm.providers):
                                if existing["id"] == pid:
                                    self.llm.providers[i] = p
                                    break
                            if pid in self.progress.provider_stats:
                                self.progress.provider_stats[pid]["status"] = "active"
                            else:
                                self.progress.provider_stats[pid] = {
                                    "done": 0, "failed": 0, "rate": 0.0,
                                    "model": p["model"], "url": p["url"],
                                    "status": "active",
                                }
                            concurrency = p.get("concurrency", 4)
                            for _ in range(concurrency):
                                t = asyncio.create_task(worker(pid))
                                active_workers.add(t)

                        # Brand new provider
                        elif pid not in known_provider_ids:
                            logger.info(
                                f"Supervisor: detected new provider #{pid} "
                                f"({p['model']}), spawning workers"
                            )
                            known_provider_ids.add(pid)
                            self.llm.add_provider(p)
                            self.progress.provider_stats[pid] = {
                                "done": 0, "failed": 0, "rate": 0.0,
                                "model": p["model"], "url": p["url"],
                                "status": "active",
                            }
                            concurrency = p.get("concurrency", 4)
                            for _ in range(concurrency):
                                t = asyncio.create_task(worker(pid))
                                active_workers.add(t)

                    # Update message if all non-disabled workers are in backoff
                    active_statuses = [
                        ps.get("status") for ps in self.progress.provider_stats.values()
                        if ps.get("status") != "disabled"
                    ]
                    all_backoff = bool(active_statuses) and all(
                        s == "backoff" for s in active_statuses
                    )
                    if all_backoff:
                        self.progress.message = "Waiting for providers..."
                    elif not queue.empty():
                        remaining = queue.qsize()
                        self.progress.message = f"Processing {remaining} remaining..."

            supervisor_task = asyncio.create_task(supervisor())

            # Wait for all workers + supervisor to complete
            while active_workers or not supervisor_task.done():
                done_workers = {t for t in active_workers if t.done()}
                active_workers -= done_workers
                for t in done_workers:
                    try:
                        t.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.error(f"Worker crashed: {exc}")

                if not active_workers and queue.empty():
                    break

                # All workers exited but queue still has items —
                # all providers exhausted/disabled, mark remaining as failed
                if not active_workers and not queue.empty():
                    remaining = queue.qsize()
                    logger.warning(
                        f"All workers exited with {remaining} assets "
                        f"still in queue — marking as failed"
                    )
                    while not queue.empty():
                        try:
                            stuck = queue.get_nowait()
                            await self.db.async_mark_failed(
                                stuck["id"], "all_providers_exhausted",
                            )
                            self.progress.failed += 1
                        except asyncio.QueueEmpty:
                            break
                    break

                await asyncio.sleep(1)

            # Cancel supervisor and any zombie workers
            if not supervisor_task.done():
                supervisor_task.cancel()
            for t in active_workers:
                t.cancel()
            all_remaining = [supervisor_task] + list(active_workers)
            for t in all_remaining:
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            self.db.finish_run(
                run_id, self.progress.total,
                self.progress.done, self.progress.failed, self.progress.skipped,
            )
            self.db.set_last_run_timestamp(datetime.now().isoformat())

            stats = {
                "total": self.progress.total,
                "done": self.progress.done,
                "failed": self.progress.failed,
                "skipped": self.progress.skipped,
            }
            self.progress.message = "Batch complete"
            logger.info(f"Batch complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            self.progress.message = f"Error: {e}"
            raise
        finally:
            self.progress.running = False
            self.progress.current_asset = ""

    def stop(self):
        """Signal the batch to stop gracefully."""
        self._stop_event.set()
        self.progress.message = "Stopping..."

    async def run_audit(self) -> dict:
        """Audit existing descriptions: fetch from Immich, score, fail bad ones."""
        threshold = float(self.db.get_config("repetition_threshold", "0.3"))
        min_words = int(self.db.get_config("min_words", "10"))
        max_words = int(self.db.get_config("max_words", "300"))
        unscored = self.db.get_unscored_done_ids()

        if not unscored:
            return {"total": 0, "scored": 0, "flagged": 0}

        self.progress.message = f"Auditing {len(unscored)} descriptions..."
        self.progress.running = True

        scored = 0
        flagged = 0

        try:
            for i, asset_id in enumerate(unscored):
                if self._stop_event.is_set():
                    break

                try:
                    detail = await self.immich.get_asset(asset_id)
                    description = (
                        detail.get("exifInfo", {}).get("description") or ""
                    )

                    if not description:
                        continue

                    score = compute_repetition_score(description)
                    word_count = len(description.split())
                    scored += 1

                    reject_reason = None
                    if score > threshold:
                        reject_reason = (
                            f"quality_audit: repetition_score={score:.2f} "
                            f"(threshold={threshold})"
                        )
                    elif word_count < min_words:
                        reject_reason = (
                            f"quality_audit: too_short={word_count} words "
                            f"(min={min_words})"
                        )
                    elif word_count > max_words:
                        reject_reason = (
                            f"quality_audit: too_long={word_count} words "
                            f"(max={max_words})"
                        )

                    if reject_reason:
                        self.db.fail_from_audit(asset_id, score, reject_reason)
                        flagged += 1
                        logger.info(
                            f"Audit flagged {asset_id[:12]}... "
                            f"{reject_reason}"
                        )
                    else:
                        self.db.update_quality_score(asset_id, score)

                except Exception as e:
                    logger.debug(f"Audit skip {asset_id[:12]}...: {e}")

                if (i + 1) % 50 == 0:
                    self.progress.message = (
                        f"Auditing... {i + 1}/{len(unscored)} "
                        f"({flagged} flagged)"
                    )

        finally:
            self.progress.running = False
            self.progress.message = (
                f"Audit complete: {scored} scored, {flagged} flagged"
            )

        stats = {"total": len(unscored), "scored": scored, "flagged": flagged}
        logger.info(f"Audit complete: {stats}")
        return stats
