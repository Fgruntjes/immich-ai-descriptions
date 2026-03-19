# Immich AI Descriptions

Generate rich, searchable descriptions for your [Immich](https://immich.app) photo library using vision LLMs. Processes photos in batch with context-aware prompts enriched by EXIF data, face recognition, OCR text, and temporal proximity.

![Dashboard](https://img.shields.io/badge/UI-htmx%20%2B%20FastAPI-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-green) ![License](https://img.shields.io/badge/license-MIT-orange)

## What It Does

For each photo in your Immich library, the app:

1. **Extracts context** — location, date/time, camera info, recognized people, OCR text, nearby photos in the same time window
2. **Builds an enriched prompt** — combines all context with your custom template
3. **Sends to vision LLM** — any OpenAI-compatible endpoint (LM Studio, llama.cpp, Ollama, etc.)
4. **Writes back to Immich** — the description becomes searchable in Immich's search

### Example

A photo of your family at dinner becomes:

> *Foto de grupo interior à noite em ambiente residencial, com sete pessoas sorrindo em fileira: homem com óculos de armação preta, mulher com jaqueta rosa, idosa de cabelos brancos com bengala... Formato: Foto*

Instead of just... nothing. Now you can search "family dinner" or "grandma" and find it.

## Features

- **Multi-provider worker pools** — distribute work across multiple LLM endpoints (fast GPU server gets more work automatically)
- **Resilient processing** — transient errors trigger backoff + health-check recovery, not permanent failure
- **Dynamic provider management** — add, remove, or toggle providers mid-run via the web UI
- **Supervisor pattern** — detects provider changes every 30s, spawns/stops workers on the fly
- **Preview playground** — test descriptions on random photos with per-provider comparison before committing to a full batch
- **Context enrichment** — EXIF location/datetime/camera, face names (from Immich), OCR text (from Immich postgres), photo sequence detection
- **Resume support** — stop mid-batch and pick up where you left off
- **Auto-save settings** — all configuration changes apply immediately

## Architecture

```
┌──────────────────────────────┐
│  Shared Queue (32K+ assets)  │
└─────┬──────────┬─────────┬───┘
      │          │         │
  ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
  │ Prov 1 │ │ Prov 2 │ │ Prov 3 │   ← Each pulls at its own speed
  │ 4 wkrs │ │ 4 wkrs │ │ 4 wkrs │
  │ ~4s/ea │ │ ~45s/ea│ │ ~16s/ea│
  └────────┘ └────────┘ └────────┘

  Supervisor (every 30s):
  - Detects new/toggled/re-enabled providers
  - Spawns/stops workers dynamically
  - Updates "Waiting for providers..." on all-backoff
```

## Quick Start

### Docker Compose (recommended)

```yaml
services:
  custom-ai-desc:
    build: .
    # Or use a pre-built image:
    # image: ghcr.io/maheidem/immich-ai-desc:latest
    restart: unless-stopped
    ports:
      - "8095:8095"
    volumes:
      - ./data:/app/data
    environment:
      IMMICH_API_URL: "http://immich-server:2283/api"
      IMMICH_API_KEY: "${IMMICH_API_KEY}"
      DB_URL: "postgresql://user:pass@immich-postgres:5432/immich"  # Optional, for OCR
      TZ: "America/Sao_Paulo"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8095/')"]
      interval: 30s
      timeout: 5s
      retries: 3
```

Add this to your existing Immich docker-compose or run standalone on the same Docker network.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `IMMICH_API_URL` | Yes | `http://immich-server:2283/api` | Immich API endpoint |
| `IMMICH_API_KEY` | Yes | — | Immich API key (create in Immich Admin → API Keys) |
| `DB_URL` | No | — | Immich PostgreSQL URL (enables OCR text enrichment) |
| `DATA_DIR` | No | `/app/data` | SQLite state database location |
| `PORT` | No | `8095` | Web UI port |
| `TZ` | No | `UTC` | Timezone for date formatting |

### First Run

1. Open `http://your-host:8095`
2. Go to **Settings** → add at least one LLM provider (URL + model)
3. Click **Validate** to verify the connection
4. Go to **Dashboard** → click **Start Processing**

## LLM Provider Setup

Any OpenAI-compatible vision endpoint works. Tested with:

| Backend | Example URL | Notes |
|---------|-------------|-------|
| [LM Studio](https://lmstudio.ai) | `http://192.168.x.x:1234/v1` | Easy setup, GPU-accelerated |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | `http://192.168.x.x:8090/v1` | Lightweight, CLI-based |
| [Ollama](https://ollama.com) | `http://192.168.x.x:11434/v1` | Pull-and-run models |

Recommended model: **Qwen 2.5 VL** or **Qwen 3.5 4B** (good vision + multilingual).

The app sends `"reasoning_format": "none"` to disable thinking mode on Qwen3+ models.

## Web UI

### Dashboard
- Live stats: done / failed / skipped / remaining
- Progress bar with per-provider breakdown (green = active, yellow pulsing = backoff, red = disabled)
- Run controls: Start, Resume, Reset Failed, Start Fresh
- Recent activity feed
- Provider health status

### Settings
- Per-provider configuration: URL, model, concurrency, timeout, sampling parameters, enable/disable toggle
- Global settings: prompt template, OCR threshold, people limit, nearby window
- All changes auto-save

### Assets
- Browse processed assets with expandable detail rows
- View full description, prompt used, provider info, duration
- Open in Immich link
- Retry failed assets
- Auto-refresh (pauses when detail row is open)

### Preview Playground
- 5 random photos with thumbnails and enriched context
- Per-provider Generate buttons for side-by-side comparison
- "All" button to fire all providers simultaneously
- No writes to Immich — safe to experiment

## Context Enrichment

Each photo's prompt is enriched with available metadata:

```
Contexto da foto:
- Local: Petrópolis, Rio de Janeiro, Brazil (-22.51, -43.18)
- Data: 18 de outubro de 2025, 15:46 (tarde)
- Camera: Samsung Galaxy S25 Ultra (f/2.2, 1/60s, ISO 320, 3.3mm)
- Resolução: 4000x3000
- Pessoas reconhecidas: Marcos, Mariana, Cissa
- Texto detectado na imagem: "LEGO Classic 790 peças"
- Contexto: parte de uma sequência de 3 fotos tiradas no mesmo período

Crie uma descrição detalhada da imagem em português brasileiro...
```

## Error Handling

| Error Type | Behavior |
|------------|----------|
| **Provider offline** (connection refused, timeout, 5xx) | Asset re-queued, worker enters exponential backoff (5→10→30→60s), health-checks until recovered |
| **Model not found** (404) | Provider permanently disabled, workers exit, assets re-queued for other providers |
| **Asset error** (thumbnail unavailable, empty response) | Asset marked failed, worker continues with next asset |
| **All providers down** | Batch stays running, displays "Waiting for providers...", resumes when any provider recovers |

## State Management

All state is stored in a single SQLite database (`data/state.db`):
- **Processed assets** — tracks which assets are done/failed/skipped with provider info
- **Provider config** — URL, model, sampling parameters, enabled state
- **Global config** — prompt template, thresholds
- **Run history** — batch stats and timestamps

The database auto-migrates on startup.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
IMMICH_API_URL=http://localhost:2283/api \
IMMICH_API_KEY=your-key \
python -m uvicorn app:app --host 0.0.0.0 --port 8095 --reload
```

## License

MIT
