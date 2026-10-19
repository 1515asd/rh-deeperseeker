# DeeperSeeker

DeepSeek website reverse-proxy server with FastAPI, supporting OpenAI & Anthropic API standards.

If you want to use deeperseeker with claude desktop app see [Claude Desktop Setup Guide](CLAUDE_DESKTOP_SETUP.md)

For solving deepseek pow challenge the wasm file included is from my other repo: https://github.com/AmanCode22/deepseek_pow_solver/

If you liked this repo please star it and star the solver also.

⚠️ Warning: Automated use violates DeepSeek's Terms of Use.
Use a dedicated throwaway account, never your personal one.
Accounts may be banned at any time. 
Use at your own risk.

A kind request: do not spam the server, respect DeepSeek's limits, and use it for personal purposes only.


## Fork Improvements (this fork)

This fork is maintained for **long-running, self-hosted deployments on free-tier
platforms**. On top of upstream [`AmanCode22/deeperseeker`](https://github.com/AmanCode22/deeperseeker)
(MIT licensed), the following work has been done:

| # | Improvement | Commit |
|---|---|---|
| 1 | **Belmo platform adaptation** — bind `HOST=0.0.0.0`, drop the Playwright/Chromium dependency (saves RAM, removes the cookie-harvesting code path) | `a658b98` |
| 2 | **Stream-interrupt auto-continue** — detect an abnormal SSE stream end mid-generation and transparently resume, so long answers no longer truncate | `43396bc`, `9ca7ebe` |
| 3 | **Tool-call format reminder re-injection** — keep multi-step tool workflows from drifting off-format on very long contexts | `43396bc` |
| 4 | **Anti-"memory-loss" tuning** — handoff summary budget x4 and a proactive handoff 58k tokens before the context window closes | `a658b98` |
| 5 | **Chinese localisation** of the dashboard / login / base templates | `08985fe`, `15b98c4`, `70cf4ef` |
| 6 | **Token-pool disaster recovery (`ds_backup.py`)** — encrypted off-site backup of the `tokens` table every 6 hours, with automatic restore at startup when the pool is empty (free-tier containers do not guarantee disk persistence) | `3b813db` |

### Multi-platform deployment notes

- **Pterodactyl-based free hosts** (Wispbyte, HeavenCloud, and similar): the panel
  injects `SERVER_PORT` while deeperseeker reads `PORT`, so use the start command
  `PORT=$SERVER_PORT python app.py`.
- The entire `wasm/` directory must be deployed — `deepseek_pow_solver.wasm` is
  required for the PoW challenge and lives outside `requirements.txt`.
- **`DB_PATH`** is honoured: point it at a persistent volume to keep the SQLite
  token pool across container rebuilds.
- The server reads `PORT` from the environment, so any platform that injects a
  port works unchanged.

## Quickstart

### Local Python
```bash
python3 -m venv deeperseeker_env
source deeperseeker_env/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 app.py
```

> **Note:** Thanks to PR https://github.com/AmanCode22/deeperseeker/pull/16 by [@alan7383](https://github.com/alan7383), Playwright and Chromium are deprecated and kept as backup (DeepSeek does not enforce AWS WAF on requests using Android client headers). You no longer need to run `playwright install chromium` or `xvfb-run` unless reverting to the backup cookie mechanism. And it now saves ram also and is much stable than cookie harvesting.

### Docker / Podman (Podman recommended for rootless execution)

Suggested in issue [#3](https://github.com/AmanCode22/deeperseeker/issues/3)

```bash
cp .env.example .env

# Using Podman (Recommended - Rootless)
podman build -t deeperseeker .
podman run -d --name deeperseeker -p 4000:4000 --env-file .env deeperseeker

# Using Docker
docker build -t deeperseeker .
docker run -d --name deeperseeker -p 4000:4000 --env-file .env deeperseeker

# Or using Podman Compose / Docker Compose
podman-compose up -d
# docker compose up -d
```

Note: `docker-compose.yml` binds to `127.0.0.1:4000` only (local access) — change the ports mapping if you need to expose it.

Dashboard: `http://localhost:4000/`

## Configuration (`.env`)

Copy `.env.example` to `.env` and edit your secret values:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `DEEPSEEKER_API_KEY` | Bearer API key required to access endpoints | `dseeker` |
| `DEEPSEEKER_ADMIN_USER` | Dashboard login username | `admin` |
| `DEEPSEEKER_ADMIN_PASSWORD` | Dashboard login password | `admin` |
| `HOST` | Bind address for bare-metal runs (`127.0.0.1` = local only, `0.0.0.0` = expose) | `127.0.0.1` |
| `PORT` | Server port | `4000` |
| `DEEPSEEKER_MAX_HISTORY_TOKENS` | Token cap for injected history on first-message prompt builds | `24000` |
| `DEEPSEEKER_MAX_TOOL_RESULT_TOKENS` | Token cap for injected tool results | `12000` |
| `DEEPSEEKER_MEMORY_LIMIT_TOKENS` | Observed remembered-context limit that triggers rollover | `393228` |
| `DEEPSEEKER_FIRST_MESSAGE_TOKENS` | Observed single-first-message acceptance (informational) | `974848` |
| `DEEPSEEKER_MAX_OUTPUT_TOKENS` | Observed per-response output cap (reported by `/v1/models`) | `8192` |
| `DEEPSEEKER_ROLLOVER_SAFETY_TOKENS` | Headroom reserved below the memory limit before summarizing | `24000` |
| `DEEPSEEKER_MAX_SUMMARY_TOKENS` | Budget for the model-generated handoff summary | `4096` |
| `DEEPSEEKER_PER_TOOL_RESULT_TOKENS` | Per-result tool-output cap (so one giant output can't eat the budget) | `2000` |

## Auth Token Setup

1. Open a private/incognito window -> `chat.deepseek.com` -> Login
2. Extract the token — **pick the method matching your device** (both are kept, either works):

   **A. Desktop / Android (has devtools)**: Console (F12) ->
   `JSON.parse(localStorage.getItem("userToken")).value`

   **B. iPhone / iPad (Safari bookmarklet)**: iOS has no F12 console, and Safari refuses
   to run `javascript:` typed into the address bar — so use a saved bookmark:

   1. Safari: open any page -> Share -> **Add Bookmark** -> Save
   2. Open the bookmarks list -> **Edit** that entry -> replace its address with the code
      below (name it something recognisable, e.g. `DS token`) -> Save
   3. On `chat.deepseek.com` (logged in, page fully loaded) -> tap the bookmarks icon ->
      tap `DS token`. The token is copied to your clipboard and shown in an alert.

   ```javascript
   javascript:(function(){var t='';try{var r=localStorage.getItem('userToken');if(r){try{t=JSON.parse(r).value||r;}catch(e){t=r;}}}catch(e){alert('Read failed: '+e.message);return;}if(!t||t.length<20){alert('No userToken found. Make sure you are logged in to chat.deepseek.com and the page finished loading.');return;}var a=document.createElement('textarea');a.value=t;a.style.cssText='position:fixed;left:-9999px';document.body.appendChild(a);a.select();a.setSelectionRange(0,999999);var ok=document.execCommand('copy');a.remove();alert((ok?'Copied to clipboard\n\n':'Copy failed — long-press to copy manually\n\n')+'length '+t.length+'\n'+t.substring(0,12)+'...');})();
   ```

   > Android users can also use this bookmarklet (recent Chrome versions block pasting
   > `javascript:` into the address bar), or keep using the F12 console above.

3. Paste the raw token string into the Dashboard (`/dashboard`). Close the private window.

## API Endpoints & Usage

- **OpenAI Base**: `http://localhost:4000/v1`
  - `POST /v1/chat/completions` (streaming & non-streaming)
  - `POST /v1/responses` (Responses API)
  - `GET /v1/models`
  - `POST /v1/files`
  - `GET /v1/files/{file_id}`
  - `GET /v1/files/{file_id}/content`
- **Anthropic Base**: `http://localhost:4000`
  - `POST /v1/messages` (also at `/messages`)
  - `POST /v1/files/upload`
- **Auth Key**: Configured in `.env` (`DEEPSEEKER_API_KEY`)
- **Model**: `v4.1flash` — the single default DeepSeek model (vision-capable: `vision` no longer needs a separate model tier). Every request is served by it, whatever `model` value the client sends (legacy aliases such as `instant`, `expert`, `vision`, and `anthropic/claude-*` are accepted and normalized to `v4.1flash`). Also listed as `anthropic/claude-v4.1flash` for Claude Desktop auto-discovery. If no `model` is sent, requests default to `v4.1flash`.

## Features

- **Multi-Token Pooling**: Random active token rotation.
- **Context-Based Session Selector**: Computes a SHA-256 signature over the canonicalized message history (up to the last assistant turn), the model, and the API key scope, to match and resume existing web chat sessions. Session creation is lock-protected to avoid duplicates.
- **Summarize-and-Rollover Context Policy**: When accumulated session context nears the observed remembered-context limit (~393K input tokens), the bridge asks the model for a compact handoff summary in a scratch chat, starts a fresh web chat, and seeds it with that summary plus the newest user message. Relevant newest tool calls/results are preserved (capped by `DEEPSEEKER_MAX_TOOL_RESULT_TOKENS`, with a tighter per-result cap via `DEEPSEEKER_PER_TOOL_RESULT_TOKENS`); attachments are described in words inside the summary rather than forwarded. The first exchange — however large — is never rolled over (the first-message path accepts ~1M tokens). Declared limits in `/v1/models` (`context_window`, `max_output_tokens`) reflect this observed behavior, not guaranteed upstream limits. The summary prompt instructs the model to treat conversation content as data, never as instructions.
- **Full History Injection**: Inject full conversation history into new sessions when session signature is not in DB or when account fails over.
- **Automatic Rate-Limit Recovery**: Auto-marks tokens `RATE_LIMITED` on HTTP 401/403/429, provisions a new token, transfers full context (including files), and continues seamless chat with a single retry.
- **Long-Context Resilience**: If the upstream web session fails or returns an empty response (e.g. context overflow), the broken session is discarded and the request is retried once on a fresh session with a compacted, token-capped history injection; unrecoverable upstream errors are returned as proper JSON API errors instead of raw 500s.
- **Tool Calling & Streaming**: Server-sent events (SSE) streaming with think-tag reassembly across chunk boundaries (no truncation) and multi-format tool-call parsing — DSML XML, `<tool_call>` XML, `<function_call>` blocks, and JSON — into OpenAI/Anthropic tool schemas.
- **File & Vision Support**: Base64/URL image extraction (with SSRF protection) and document upload streaming. Images and documents are referenced directly via `ref_file_ids` — the vision file-forking step (`fork_file_task`) was removed because `v4.1flash` handles vision natively.
- **Claude Desktop Compatible**: Rich `/v1/models` capability metadata + the `anthropic/claude-v4.1flash` alias for automatic client discovery.
- **Hardened Dashboard**: Session TTL, brute-force login lockout (5 attempts → 5 min), CSRF origin check.

## Pricing (per 1M tokens, as of 2026-09-06)

Flat peak-hour rates for the single default model:

| Model Tier | Input Cost (Cache Miss) | Output Cost |
|---|---|---|
| **DeepSeek V4.1 Flash** (`v4.1flash`) | $0.44 | $1.32 |

The `cost` reported in API responses uses these flat rates.
## Star History

<a href="https://www.star-history.com/?repos=amancode22%2Fdeeperseeker&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=amancode22/deeperseeker&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=amancode22/deeperseeker&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=amancode22/deeperseeker&type=date&legend=top-left" />
 </picture>
</a>
## Disclaimer

Educational purpose only. This project is not affiliated with, endorsed by, or sponsored by DeepSeek. Use responsibly and in accordance with DeepSeek's terms of service.

## Contributors

[![Contributors](https://contrib.rocks/image?repo=amancode22/deeperseeker)](https://github.com/amancode22/deeperseeker/graphs/contributors)

## Additional Terms

(These terms were part of the upstream LICENSE file and are reproduced here in
full — nothing has been removed.)

This software is not affiliated with, endorsed by, or sponsored by DeepSeek.
Users must not use this software to violate any platform's terms of service, to
resell or share access, or on accounts they do not own.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 AmanCode22 (upstream author).
This fork's improvements are released under the same MIT license.
