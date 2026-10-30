import asyncio
import json
import logging
import mimetypes
import os
import random
import re
import secrets
import sqlite3
import time
import unicodedata
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from urllib.parse import urlparse

import deepseek_tokenizer
import uvicorn
from dotenv import load_dotenv
from uvicorn.logging import AccessFormatter
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import i18n

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── [v74] 重试保险丝（真幂等）─────────────────────────────────────────
# 官方 API 无状态 → 客户端重试绝对安全；我们有状态会话，完成后重试会让
# 上游重复执行。此缓存记录"刚完成"的请求结果：窗口内同签名请求直接回放，
# 不重复执行。仅记录成功完成的回合（失败/中断不记，重试照常放行）。
# 权限取舍：窗口默认 120 秒——网络重试都在秒级；人类点"重新生成"通常会
# 先删掉上一条回复（消息列表变化 → 签名不同 → 不受影响）。
_DUP_CACHE = {}
_DUP_CACHE_MAX = 256


def _dup_dbg(msg):
    try:
        with open(os.path.join(BASE_DIR, "dup_debug.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")
    except Exception:
        pass


def _dup_record(sig, body):
    _DUP_CACHE[sig] = {"ts": time.time(), "body": body}
    while len(_DUP_CACHE) > _DUP_CACHE_MAX:
        _DUP_CACHE.pop(next(iter(_DUP_CACHE)))


def _dup_lookup(sig):
    ent = _DUP_CACHE.get(sig)
    if not ent:
        return None
    if time.time() - ent["ts"] > int(os.getenv("DEEPSEEKER_DUP_WINDOW", "120")):
        _DUP_CACHE.pop(sig, None)
        return None
    return ent["body"]


def _json_mode_coerce(resp_text):
    """[v73] JSON 模式出口校验：从原始输出提取合法 JSON 值。

    返回 (text, ok)。三层尝试：原文直解 → 剥 markdown 围栏 → 首尾括号截取。
    全部失败时原样返回（不阻塞交付）。"""
    body = re.sub(r"<think>.*?</think>", "", resp_text, flags=re.DOTALL).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    try:
        json.loads(body)
        return body, True
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i = body.find(open_c)
        j = body.rfind(close_c)
        if i != -1 and j > i:
            cand = body[i:j + 1]
            try:
                json.loads(cand)
                return cand, True
            except Exception:
                continue
    return resp_text, False


def _dump_prompt_debug(session_id: str, text: str):
    """DEEPSEEKER_STREAM_DEBUG=1 时，把每轮实际发给模型的 prompt 落盘。

    定位"结果错位/串线"：模型对"哪个结果对应哪条调用"的判断完全取决于
    这份打包文本（[TOOL RESULTS] 的 Command 配对）。有它就能对照模型
    自报的"收到旧结果"判断是真错位还是模型误读。独立目录独立 20 份环，
    不与输出 dump（debug_streams）互相挤占。"""
    if os.getenv("DEEPSEEKER_STREAM_DEBUG") != "1" or not text:
        return
    try:
        d = os.path.join(BASE_DIR, "debug_prompts")
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"{int(time.time())}_prompt_{str(session_id)[:8]}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
        files = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
        for old in files[:-20]:
            try:
                os.remove(os.path.join(d, old))
            except OSError:
                pass
    except Exception:
        pass


def _dump_stream_debug(session_id: str, text: str, kind: str = "stream"):
    """DEEPSEEKER_STREAM_DEBUG=1 时，把官网返回的原始聚合文本落盘，
    用于定位流式损坏发生在哪一环（官网输出 / 服务端处理 / 客户端显示）。
    只保留最近 20 份，防止磁盘累积。
    同时写入 db 表 debug_dumps —— 面板文件 API 的子目录读取通道不稳定，
    根目录 db 下载是可靠的取证通道。"""
    if os.getenv("DEEPSEEKER_STREAM_DEBUG") != "1" or not text:
        return
    try:
        d = os.path.join(BASE_DIR, "debug_streams")
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"{int(time.time())}_{kind}_{str(session_id)[:8]}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
        files = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
        for old in files[:-20]:
            try:
                os.remove(os.path.join(d, old))
            except OSError:
                pass
    except Exception:
        pass
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, "deeperseeker.db"), timeout=10)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS debug_dumps (ts INTEGER, name TEXT, content TEXT)"
            )
            conn.execute(
                "INSERT INTO debug_dumps (ts, name, content) VALUES (?, ?, ?)",
                (int(time.time()), f"{int(time.time())}_{kind}_{str(session_id)[:8]}", text),
            )
            conn.execute(
                "DELETE FROM debug_dumps WHERE ts NOT IN (SELECT ts FROM debug_dumps ORDER BY ts DESC LIMIT 20)"
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
# B12 (Stage 1 audit): the global os.chdir(BASE_DIR) is gone — every relative
# path resolves from BASE_DIR explicitly. Import-time process-wide state is
# hostile to embedding, multi-worker setups and packaging.

load_dotenv(os.path.join(BASE_DIR, ".env"))

security = HTTPBasic()


import official_api
from functions import (
    CookieGenerationError,
    UpstreamError,
    cookie_file_path,
    data_dir,
    add_token,
    count_tokens,
    create_new_chat,
    delete_token,
    delete_sessions_for_chat,
    find_session,
    get_auth_token,
    get_chat_baseline,
    get_rolling_state,
    set_rolling_state,
    get_token,
    get_tokens,
    init_db,
    mark_limited,
    mark_active,
    is_context_limit_error,
    record_usage,
    usage_by_day_key,
    get_pool_view,
    effort_level,
    request_effort as _request_effort,
    request_tools as _request_tools,
    next_parent,
    parse_tools,
    pick_token,
    warmup_pool,
    acquire_token_slot,
    record_file,
    get_file_token,
    save_session,
    set_chat_baseline,
    send_message,
    close_session,
    StreamToolParser,
    upload_file,
    get_file_content,
    _RetryBoundary,
)


def _resolve_api_key():
    """B9 (Stage 1 audit): never ship an open relay.

    The API key used to fall back to the publicly documented 'dseeker'
    silently, and the dashboard to admin/admin — an exposed host plus these
    defaults manufactured an open relay (the failure class that killed
    ds2api). Now: an unset key is GENERATED (dsk- + 24 urlsafe chars),
    printed once on boot and persisted next to the database so restarts keep
    the same key. An explicit DEEPSEEKER_API_KEY is honored unchanged.

    Returns (api_key, was_generated)."""
    key = os.getenv("DEEPSEEKER_API_KEY", "").strip()
    if key:
        return key, False
    key_file = os.path.join(data_dir(), "api_key.txt")
    try:
        with open(key_file) as f:
            saved = f.read().strip()
        if saved:
            return saved, True
    except OSError:
        pass
    generated = "dsk-" + secrets.token_urlsafe(24)
    try:
        os.makedirs(os.path.dirname(key_file) or ".", exist_ok=True)
        with open(key_file, "w") as f:
            f.write(generated + "\n")
        os.chmod(key_file, 0o600)
    except OSError:
        pass  # best-effort persistence; the key is printed below regardless
    return generated, True


API_KEY, _api_key_generated = _resolve_api_key()
ADMIN_USER = os.getenv("DEEPSEEKER_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("DEEPSEEKER_ADMIN_PASSWORD", "admin")

from middleware import RecovererMiddleware, RealIPMiddleware, RequestIDMiddleware
from plugin_helper import (
    build_prompt,
    build_summary_request_prompt,
    context_window_tokens,
    estimate_conversation_tokens,
    extract_and_upload_files,
    generate_signature,
    generate_signature_sync,
    max_output_tokens,
    needs_rollover,
    strip_summary_tags,
    build_rolling_digest_prompt,
    compute_rollover_split,
    chunk_messages,
    render_history_tail,
    messages_plain_text as _messages_plain_text,
    ROLLOVER_TAIL_TOKENS,
    DIGEST_CHUNK_TOKENS,
    MAX_HISTORY_TOKENS,
    MAX_SUMMARY_TOKENS,
    estimate_conversation_tokens_cached,
    resume_candidate_messages,
    RESUME_NOTICE,
)


def _packed_first_message_tokens(messages) -> int:
    """Size of what build_prompt actually packs into THIS chat's first
    message: the full client history, capped at MAX_HISTORY_TOKENS. Used as
    the chat baseline so the rollover growth-trigger measures in-session
    accumulation from the real packed start. (The client-side total history
    grows monotonically across rollovers and must never be the anchor.)"""
    return min(estimate_conversation_tokens_cached(messages), MAX_HISTORY_TOKENS)


logger = logging.getLogger("uvicorn.error")

# Surface deeperseeker.* logs (cookie generation, prunes, rate-limit events,
# upstream failures) alongside uvicorn's own output, unless already configured.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _log_security_banner():
    """B9 (Stage 1 audit): surface insecure defaults loudly on boot instead of
    quietly shipping an open relay."""
    host = (os.getenv("HOST") or "").strip().lower()
    loopback = host in ("", "127.0.0.1", "localhost", "::1")
    if _api_key_generated:
        logger.warning(
            "SECURITY: DEEPSEEKER_API_KEY was not set — a strong API key was generated for this "
            "instance and saved to api_key.txt next to the database. It is shown ONCE here:\n"
            "  API key: %s",
            API_KEY,
        )
    elif API_KEY.strip().lower() == "dseeker":
        logger.warning(
            "SECURITY: DEEPSEEKER_API_KEY is the publicly documented default 'dseeker'. "
            "Set a strong key before exposing this service beyond loopback."
        )
    if ADMIN_USER == "admin" and ADMIN_PASSWORD == "admin":
        if loopback:
            logger.warning(
                "SECURITY: the dashboard uses the default admin/admin credentials — "
                "set DEEPSEEKER_ADMIN_USER / DEEPSEEKER_ADMIN_PASSWORD."
            )
        else:
            logger.error(
                "SECURITY: the dashboard uses admin/admin while binding a NON-LOOPBACK host (%s). "
                "Anyone who can reach this service owns its token pool — set DEEPSEEKER_ADMIN_USER "
                "and DEEPSEEKER_ADMIN_PASSWORD before exposing it.",
                host or "0.0.0.0",
            )


_log_security_banner()

# The token used for this request, logged as "key: <alias>".
# Must stay a dict: BaseHTTPMiddleware runs the endpoint in a child task, and
# only in place mutation of the same object reaches the access log context.
_key_holder = ContextVar("deeperseeker_key", default=None)
_ALIAS_MAX_LEN = 64
# Strip characters that forge log lines, drive the cursor, or render invisibly.
_ALIAS_STRIP_CATS = frozenset({"Cc", "Cf", "Zl", "Zp"})
_ALIAS_INVISIBLE = frozenset(
    [0x034F, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x180B, 0x180C, 0x180D,
     0x2800, 0x3164, 0xFFA0]
    + list(range(0xFE00, 0xFE10))
    + list(range(0xE0100, 0xE01F0))
)
# Noncharacters are illegal in interchange; some log processors reject them.
_ALIAS_NONCHARACTERS = frozenset(
    list(range(0xFDD0, 0xFDF0))
    + [plane << 16 | low for plane in range(0x11) for low in (0xFFFE, 0xFFFF)]
)


def _sanitize_alias(name):
    if not name:
        return None
    cleaned = "".join(
        ch for ch in str(name)
        if ord(ch) not in _ALIAS_INVISIBLE
        and ord(ch) not in _ALIAS_NONCHARACTERS
        and unicodedata.category(ch) not in _ALIAS_STRIP_CATS
    ).strip()
    return cleaned[:_ALIAS_MAX_LEN] or None


def _set_key_name(name):
    holder = _key_holder.get()
    if holder is not None:
        holder["name"] = _sanitize_alias(name)


class KeyAccessFormatter(AccessFormatter):
    def formatMessage(self, record):
        line = super().formatMessage(record)
        # Sanitize again at emit time so no raw value reaches the log line.
        name = _sanitize_alias((_key_holder.get() or {}).get("name"))
        return f"{line} key: {name}" if name else line


def _install_key_access_formatter():
    for handler in logging.getLogger("uvicorn.access").handlers:
        formatter = handler.formatter
        if not isinstance(formatter, AccessFormatter) or isinstance(formatter, KeyAccessFormatter):
            continue
        handler.setFormatter(
            KeyAccessFormatter(
                fmt=formatter._fmt,
                datefmt=formatter.datefmt,
                use_colors=getattr(formatter, "use_colors", None),
            )
        )


def count_tok(text):
    return len(deepseek_tokenizer.ds_token.encode(text))


async def _db(fn, *args, **kwargs):
    """Run a blocking SQLite store helper off the event loop (B5, Stage 1 audit).

    Every sessions/tokens read + write used to run inline: under concurrent
    streams each sqlite3.connect() round-trip and WAL commit stalled the whole
    loop exactly when the proxy was busiest — head-of-line blocking on every
    request (B5). to_thread keeps the store helpers sync (they are shared by
    CLI paths and tests) while the loop never blocks on disk I/O. The
    aiosqlite migration belongs to the store split (Stage 6)."""
    return await asyncio.to_thread(fn, *args, **kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _db(init_db)
    _install_key_access_formatter()
    # [v57 移植 v35] 预热：wasm 编译 + 上游 TLS + PoW 池。30s 封顶，失败不影响启动。
    try:
        await asyncio.wait_for(warmup_pool(), timeout=30)
    except Exception as exc:
        print(f"[warmup] skipped: {exc}", flush=True)
    # ds_backup: token-pool disaster recovery — auto-restore missing tokens at
    # startup, periodic encrypted off-site backup while running. Silently
    # no-ops unless DEEPSEEKER_BACKUP_SECRET is configured.
    backup_task = None
    try:
        import ds_backup
        restored = await asyncio.to_thread(ds_backup.restore_if_needed)
        print(f"[ds_backup] {restored}", flush=True)
        backup_task = asyncio.create_task(ds_backup.background_loop())
    except Exception as exc:
        print(f"[ds_backup] disabled: {exc}", flush=True)
    yield
    if backup_task is not None:
        backup_task.cancel()
    # Stage 1 minor list: release the shared aiohttp ClientSession so shutdown
    # does not leak its connector sockets.
    await close_session()


app = FastAPI(title="DeeperSeeker", lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    _key_holder.set({})
    cl = request.headers.get("content-length", "")
    if cl.isdigit() and int(cl) > 32 * 1024 * 1024:
        return JSONResponse({"error": "Request body too large"}, status_code=413)
    return await call_next(request)


# Stage 0.1 + 0.2 — core rails: fail-closed real-client-IP resolution behind
# proxies (TRUSTED_PROXIES), per-request correlation ids with access logging,
# and a last-resort exception barrier that turns handler crashes into logged
# JSON 500s. Starlette runs the LAST-registered middleware FIRST (outermost),
# so registration order RealIP -> RequestID -> Recoverer yields the execution
# order Recoverer -> RequestID -> RealIP -> limit_body_size -> routes: the
# recoverer sees every error below it, and every response (including its own
# 500s) carries the request id.
app.add_middleware(RealIPMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RecovererMiddleware)


SESSIONS = {}
SESSION_TTL = 7 * 24 * 3600
# B12 (Stage 1 audit): NOTE — SESSIONS (and _login_fails below) are per-process
# admin state: they reset on restart and are not shared across workers. This
# service is single-worker by design (uvicorn workers=1); a shared store
# belongs to the Stage 6 store split. Expired entries are pruned
# opportunistically on login instead of living until restart.
# One asyncio.Lock per conversation signature, used to serialize first-time
# session creation. Signatures are unique per message prefix, so without a cap
# this dict grows FOREVER — after many chats it becomes a serious memory leak.
# _take_lock() enforces the cap as a true LRU (the old evict-half loop broke
# out when its whole chunk was locked and then inserted the new key anyway,
# so the cap never held under load — PR #26 review, High). Under pressure it
# registers over cap rather than aliasing chats onto a shared lock — see
# _take_lock for why that trade is required for correctness.
_sig_locks = OrderedDict()
SIG_LOCKS_MAX = int(os.getenv("DEEPSEEKER_MAX_SIG_LOCKS", "4096"))
_login_fails = {"count": 0, "locked_until": 0}

# Stage 0.3 — Per-chat locks (session-collision fix).
# One asyncio.Lock per UPSTREAM chat session id. _sig_locks above only
# serializes first-time session CREATION for one signature; it does nothing
# for two concurrent requests that already share (or race to use) the same
# upstream DeepSeek chat: both would send with the same parent_message_id,
# fork the upstream conversation, and the last save_session() would corrupt
# the stored parent counter. The chat lock serializes the send -> save
# critical section per upstream session, so same-chat requests queue instead
# of colliding (different chats remain fully parallel). Keyed by upstream
# session id rather than signature, because each completed turn derives a new
# signature while the upstream chat stays the same. The registry is capped by
# _take_lock() (true LRU; see its docstring for the PR #26 review fix that
# replaced the old evict-half loop, which could not actually cap).
_chat_locks = OrderedDict()
CHAT_LOCKS_MAX = int(os.getenv("DEEPSEEKER_MAX_CHAT_LOCKS", "4096"))


def _take_lock(label, registry, key, max_entries):
    """Return the lock for `key` from an LRU-capped registry, creating it on
    first use.

    Review fix (PR #26, High): the previous eviction loop shared by
    _sig_locks/_chat_locks scanned the oldest MAX//2+1 entries, broke out when
    ALL of them were locked, and then setdefault'ed the new key anyway — so
    under sustained load with many live chats the dict grew without bound; the
    "memory-leak guard" was the leak. Semantics now:
      - a hit moves the key to the most-recently-used end;
      - over cap, the least-recently-used UNLOCKED entry is evicted (a locked
        entry is never evicted — that would fork a chat's critical section;
        worst case stays one benign re-creation race, as before);
      - if EVERY entry is held, the new key is STILL registered, growing the
        registry over cap. Addendum: an earlier draft returned a shared
        fallback lock WITHOUT registering the key — that traded the memory
        bound for a correctness one. If pressure dropped while that request
        was still in flight, the next request for the same chat found room,
        created a fresh per-key lock, and two live holders sat in one chat's
        critical section — reopening the exact parent_message_id race Stage
        0.3 exists to close, precisely under the load the branch was designed
        for. Registering over cap keeps same chat -> same lock object
        unconditionally. The cost stays bounded: over-cap entries appear only
        when every existing lock is held (in-flight pressure), so depth
        tracks request concurrency during pressure windows — never total chat
        history — and drained entries are inert until recycled by LRU churn.
        Each over-cap registration logs a warning with the live depth: the
        ops signal for sustained pressure (raise the cap if it fires
        continuously).
    """
    lock = registry.get(key)
    if lock is not None:
        registry.move_to_end(key)
        return lock
    if len(registry) >= max_entries:
        for old_key, held in registry.items():
            if not held.locked():
                del registry[old_key]
                break
        # nothing unlocked? fall through and register over cap — refusing to
        # register (or aliasing to a shared lock) would break same-chat
        # identity, which is the invariant this registry exists to guarantee
    lock = asyncio.Lock()
    registry[key] = lock
    if len(registry) > max_entries:
        logger.warning(
            "%s lock registry over cap: %d entries (cap %d) — every existing lock is "
            "held; depth tracks in-flight requests, not chat history",
            label, len(registry), max_entries,
        )
    return lock


def _chat_lock(session_id):
    return _take_lock("chat", _chat_locks, str(session_id), CHAT_LOCKS_MAX)


class _OwnedChatLock:
    """Ownership token for one acquisition of a per-chat lock.

    Review fix (PR #26, Medium): release sites used `if lock.locked():
    lock.release()`, but locked() reports whether ANYONE holds the lock —
    after an early release and another request's acquisition, a late release
    dropped the OTHER request's lock and put two requests inside the critical
    section the lock exists to prevent. A token tracks only its own
    acquisition: release() is idempotent and can never release a stranger's
    hold, which also makes ownership transfer to a stream generator and
    release-before-retry safe by construction. `owned` says whether THIS
    token still holds the lock (the question locked() could not answer)."""

    __slots__ = ("lock", "_owned")

    def __init__(self, lock):
        self.lock = lock
        self._owned = False

    async def acquire(self):
        await self.lock.acquire()
        self._owned = True

    def release(self):
        if not self._owned:
            return
        self._owned = False
        try:
            self.lock.release()
        except RuntimeError:
            pass  # defensive: releasing an already-released lock must never kill a request

    @property
    def owned(self):
        return self._owned


async def _own_chat_lock(session_id):
    """Acquire this chat's lock and return its ownership token."""
    token = _OwnedChatLock(_chat_lock(session_id))
    await token.acquire()
    return token


def _release_chat_lock_stream(gen, owner, slot=None):
    """Wrap a streaming generator so the per-chat lock stays held until the
    stream completes (or the client aborts), then is released exactly once.

    The lock is acquired in handle_chat before the upstream POST; for streaming
    responses the final save_session() happens inside the stream generator, so
    ownership of the lock must transfer from handle_chat to the generator —
    releasing any earlier would reopen the parent_message_id race the lock
    exists to prevent. `owner` is a _OwnedChatLock token: its release() drops
    ONLY this holder's acquisition, never a stranger's (PR #26 review, Medium).
    `slot` (B3) is the request's in-flight token reservation: it transfers to
    the generator alongside the lock so pick_token()'s least-in-flight view
    stays correct for the whole stream duration."""
    async def _wrapped():
        try:
            async for chunk in gen:
                yield chunk
        finally:
            owner.release()
            if slot is not None:
                slot.release()
    return _wrapped()


def get_current_admin(request: Request):
    sid = request.cookies.get("session_id")
    if not sid or sid not in SESSIONS or time.time() - SESSIONS[sid] > SESSION_TTL:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    SESSIONS[sid] = time.time()
    origin = request.headers.get("origin", "")
    if origin:
        parsed = urlparse(origin).netloc
        if parsed and parsed != request.headers.get("host", ""):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return "admin"


def get_api_key(request: Request):
    auth = request.headers.get("authorization", "")
    api_key_header = request.headers.get("x-api-key", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    elif auth:
        return auth
    return api_key_header


def check_key(request: Request):
    key = get_api_key(request)
    return secrets.compare_digest(key.encode("utf-8"), API_KEY.encode("utf-8"))


def _upstream_http_code(exc):
    """HTTP status carried by an upstream failure (B6).

    Typed UpstreamError exposes .status directly; anything else (connection-
    level failures, cookie generation) maps to None and callers fall back to
    502. The old str(e) regex parsing of the 'HTTP (\\d{3}):' prefix is gone —
    any upstream wording change could silently disable token rotation and
    rate-limit marking."""
    status = getattr(exc, "status", None)
    if isinstance(status, int) and 400 <= status <= 599:
        return status
    return None


def _api_error_response(e, is_anthropic=False):
    code = _upstream_http_code(e) or 502
    if isinstance(e, CookieGenerationError):
        code = 503  # WAF cookies cannot be produced right now — upstream unreachable, not a client error
    if code < 400 or code > 599:
        code = 502
    if is_anthropic:
        payload = {"type": "error", "error": {"type": "api_error", "message": str(e)[:500]}}
    else:
        err_type = "rate_limit_error" if code == 429 else "api_error"
        payload = {"error": {"message": str(e)[:500], "type": err_type, "code": code}}
    return JSONResponse(payload, status_code=code)


def _referenced_file_ids(messages):
    """File ids referenced by the conversation (uploaded earlier via /v1/files
    or Anthropic file sources). Pure scan, no I/O — used to prefer the
    file-owner token and to detect foreign-owned references (B4)."""
    ids = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "file" and isinstance(part.get("file"), dict) and part["file"].get("file_id"):
                ids.append(part["file"]["file_id"])
            elif (
                part.get("type") in ("document", "image")
                and isinstance(part.get("source"), dict)
                and part["source"].get("type") == "file"
                and part["source"].get("file_id")
            ):
                ids.append(part["source"]["file_id"])
    return ids


async def _copy_file_to_token(file_id, fetch_token, target_token):
    """Fetch a file's bytes with fetch_token and upload them to the account of
    target_token (B4 re-home). Returns the new file_id, or None on failure."""
    mime = None
    chunks = []
    size = 0
    try:
        gen = get_file_content(fetch_token, file_id)
        mime = await gen.__anext__()  # first yield is the mime type
        async for chunk in gen:
            size += len(chunk)
            if size > 25 * 1024 * 1024:
                logger.warning("File ownership: re-home of %s aborted (over 25 MB)", file_id)
                return None
            chunks.append(chunk)
    except StopAsyncIteration:
        return None
    except Exception:
        logger.exception("File ownership: fetch of %s failed during re-home", file_id)
        return None
    ext = (mimetypes.guess_extension(mime) if mime else None) or ".bin"
    filename = f"rehomed_{file_id}{ext}"
    async for status, data in upload_file(b"".join(chunks), filename, mime or "application/octet-stream", target_token):
        if status == "success":
            return data["file_id"]
    return None


async def _rehome_foreign_files(file_ids, token_id, tok):
    """Return file_ids usable by token_id's account (B4).

    Uploads are pinned to their token and upstream files are account-scoped,
    so a reference owned by another token would 404 at chat time. Foreign-owned
    ids are copied onto this chat's token (fetch with the owner, upload with
    the chat's token). Unknown (legacy, unregistered) ids pass through
    unchanged — nothing better than the old behavior is possible for them."""
    out = []
    for fid in file_ids:
        owner = await _db(get_file_token, fid)
        if owner is None or owner == token_id:
            out.append(fid)
            continue
        owner_tok = await _db(get_token, owner)
        fetch_token = owner_tok["token"] if owner_tok else tok["token"]
        new_id = await _copy_file_to_token(fid, fetch_token, tok["token"])
        if new_id:
            await _db(record_file, new_id, token_id)
            logger.info(
                "File ownership: re-uploaded file %s (token #%s) onto token #%s as %s",
                fid, owner, token_id, new_id,
            )
            out.append(new_id)
        else:
            # Best effort: keep the original reference rather than dropping it.
            out.append(fid)
    return out


def _replay_stream(gen, first):
    async def _wrapped():
        if first is not None:
            yield first
        async for chunk in gen:
            yield chunk
    return _wrapped()


async def _preflight_stream(gen):
    """Consume the first chunk eagerly so upstream errors surface before streaming starts."""
    try:
        first = await gen.__anext__()
    except StopAsyncIteration:
        first = None
    return _replay_stream(gen, first)


# B10 (Stage 1 audit): bounded retry budget for generic upstream failures
# (empty SSE, transient 5xx, poisoned sessions): up to MAX_UPSTREAM_ATTEMPTS
# total attempts, each preferring a DIFFERENT token than the one that just
# failed, with jittered backoff between attempts. Exhausting the budget
# returns the upstream error (502-class) with the redacted trace.
MAX_UPSTREAM_ATTEMPTS = max(1, int(os.getenv("DEEPSEEKER_MAX_UPSTREAM_ATTEMPTS", "3")))


async def handle_chat(
    messages,
    model,
    thinking=False,
    search=False,
    stream=False,
    tools=None,
    is_anthropic=False,
    req_model=None,
    scope="",
    _attempt=0,
    _auth_rotated=False,
    _exclude_token=None,
    json_mode=False,
    _force_rollover=False,
):
    # [v77 PR#70/72 移植] 把请求级信息放进 ContextVar：parse_tools 反推
    # 工具名用工具表；build_prompt 挂 effort 指令用档位。
    _request_tools.set(tools or [])
    auth_token = await _db(get_auth_token)
    if not auth_token:
        return JSONResponse({"error": "No auth token. Add via dashboard."}, status_code=401)

    sig = await generate_signature(messages, model, scope)
    sess = await _db(find_session, sig)
    resume_recovered = False
    if not sess and os.getenv("DEEPSEEKER_RESUME_MATCH", "1") == "1":
        # Interrupted-turn recovery: when the client walks away mid-generation
        # (user stop / client timeout), the resent history is missing the last
        # assistant turn, so the exact signature misses and a fresh chat would
        # re-pack the whole history. Strip trailing unanswered turns and retry
        # the prefix signature: a hit re-attaches to the ORIGINAL chat with
        # its full upstream memory intact.
        for cand in resume_candidate_messages(messages):
            sess = await _db(find_session, await generate_signature(cand, model, scope))
            if sess:
                resume_recovered = True
                logger.info(
                    "Resume recovery: re-attached to chat %s after interrupted turn",
                    str(sess.get("session_id", ""))[:8],
                )
                break
    rollover_summary = None
    ref_ids = []  # B4: file ids referenced by the conversation (set by the create path)

    if sess:

        token_id = sess["token_id"]
        session_id = sess["session_id"]
        parent_message_id = sess["parent_message_id"]
        tok = await _db(get_token, token_id)
        if not tok or tok["status"] == "RATE_LIMITED":
            new_token_id = await _db(pick_token)
            if new_token_id and (not tok or new_token_id != token_id):
                new_tok = await _db(get_token, new_token_id)
                if new_tok:
                    _set_key_name(new_tok.get("alias"))
                    # Stage 0.3: rotation re-creates the upstream chat; hold the
                    # new chat's lock across its send -> save section so a
                    # concurrent same-signature request cannot race the swap.
                    rot_owner = None
                    rot_slot = acquire_token_slot(new_token_id)  # B3: reservation follows the send
                    try:
                        await _db(delete_sessions_for_chat, token_id, session_id)
                        new_session_id = await create_new_chat(new_tok["token"])
                        rot_owner = await _own_chat_lock(new_session_id)
                        if needs_rollover(messages, tools=tools):
                            scratch_chat = await create_new_chat(new_tok["token"])
                            summary_gen = send_message(
                                scratch_chat, new_tok["token"], build_summary_request_prompt(messages), 0, False, False, []
                            )
                            rollover_summary = strip_summary_tags(await collect_response(summary_gen))[: MAX_SUMMARY_TOKENS * 4]
                        prompt = await build_prompt(messages, tools or [], model, is_first_message=True, rollover_summary=rollover_summary)

                        file_ids = await extract_and_upload_files(messages, new_tok["token"])
                        for fid in file_ids:
                            await _db(record_file, fid, new_token_id)
                        gen = send_message(new_session_id, new_tok["token"], prompt, 0, thinking, search, file_ids)
                        gen = await _preflight_stream(gen)
                    except Exception as e:
                        if rot_owner is not None:
                            rot_owner.release()
                        rot_slot.release()
                        logger.exception("Token-rotation recovery failed (chat %s): %s", session_id, e)
                        if _attempt + 1 >= MAX_UPSTREAM_ATTEMPTS:
                            return _api_error_response(e, is_anthropic)
                        return await handle_chat(messages, model, thinking, search, stream, tools, is_anthropic, req_model, scope, _attempt=_attempt + 1)
                    if stream:
                        gen = _release_chat_lock_stream(gen, rot_owner, rot_slot)
                        if is_anthropic:
                            return StreamingResponse(stream_anthropic_response(gen, model, messages, new_token_id, new_session_id, sig, tools, req_model, 0, scope), media_type="text/event-stream")
                        return StreamingResponse(stream_response(gen, model, messages, new_token_id, new_session_id, sig, tools, 0, scope), media_type="text/event-stream")
                    else:
                        try:
                            resp_text = await collect_response(gen)
                        except Exception as e:
                            logger.exception("Upstream failed during token-rotation request: %s", e)
                            if rot_owner is not None:
                                rot_owner.release()
                            rot_slot.release()
                            if _attempt + 1 >= MAX_UPSTREAM_ATTEMPTS:
                                return _api_error_response(e, is_anthropic)
                            return await handle_chat(messages, model, thinking, search, stream, tools, is_anthropic, req_model, scope, _attempt=_attempt + 1)
                        await _db(mark_active, new_token_id)
                        rot_slot.release()

                        parsed_tools, clean_text = parse_tools(resp_text)
                        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
                        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()
                        next_messages = messages.copy()
                        ast_msg = {"role": "assistant"}
                        if parsed_tools:
                            ast_msg["tool_calls"] = parsed_tools
                        else:
                            ast_msg["content"] = clean_text
                        next_messages.append(ast_msg)
                        next_sig = await generate_signature(next_messages, model, scope)

                        await _db(save_session, sig, new_token_id, new_session_id, next_parent(0))
                        await _db(set_chat_baseline, new_session_id, _packed_first_message_tokens(messages),
                                  estimate_conversation_tokens_cached(messages))
                        await _db(save_session, next_sig, new_token_id, new_session_id, next_parent(0))
                        if rot_owner is not None:
                            rot_owner.release()
                        return format_response(resp_text, model, messages, tools)
            return JSONResponse({"error": {"message": "No active tokens available (all rate limited). Try again later.", "type": "rate_limit_error"}}, status_code=429)
    else:

        create_lock = _take_lock("sig", _sig_locks, sig, SIG_LOCKS_MAX)
        async with create_lock:
            sess = await _db(find_session, sig)
            if not sess:
                # B10: the retry budget passes the id of the token that just
                # failed so pick_token() rotates off it while any other token
                # is available.
                token_id = await _db(
                    pick_token,
                    exclude={_exclude_token} if _exclude_token is not None else None,
                )
                if not token_id:
                    return JSONResponse({"error": "No tokens available"}, status_code=503)
                # B4: upstream files are account-scoped. If the conversation's
                # first turn references uploaded files with a single known
                # owner, run the chat on that token — a scheduler pick from a
                # different account would get "file not found" upstream.
                ref_ids = _referenced_file_ids(messages)
                if ref_ids:
                    owners = {await _db(get_file_token, fid) for fid in ref_ids}
                    owners.discard(None)
                    if len(owners) == 1:
                        owner_id = owners.pop()
                        if owner_id != token_id:
                            owner_tok = await _db(get_token, owner_id)
                            if owner_tok and owner_tok["status"] == "ACTIVE":
                                logger.info(
                                    "File ownership: chat references file(s) pinned to token #%s; using it",
                                    owner_id,
                                )
                                token_id = owner_id
                tok = await _db(get_token, token_id)
                if not tok:
                    return JSONResponse({"error": "Token not found"}, status_code=503)

                # Accumulated-context rollover (issue #22): when the conversation is
                # nearing the observed remembered-context limit, first obtain a
                # model-generated handoff summary via a scratch chat (the request
                # itself is near the context limit, so it must not be sent into any
                # chat that has to absorb it), then start the real chat seeded with
                # that summary via build_prompt(rollover_summary=...). A single large
                # first exchange is untouched (the first-message path accepts ~1M
                # tokens); this only fires for accumulated session context.
                if needs_rollover(messages, tools=tools):
                    scratch_chat = await create_new_chat(tok["token"])
                    summary_gen = send_message(
                        scratch_chat, tok["token"], build_summary_request_prompt(messages), 0, False, False, []
                    )
                    summary = strip_summary_tags(await collect_response(summary_gen))
                    rollover_summary = summary[: MAX_SUMMARY_TOKENS * 4]  # ~4 chars/token cap
                    logger.info(
                        "Context rollover: handoff summary of ~%d tokens prepared in scratch chat %s",
                        count_tok(rollover_summary), scratch_chat,
                    )

                session_id = await create_new_chat(tok["token"])
                await _db(save_session, sig, token_id, session_id, 0)
                await _db(set_chat_baseline, session_id, _packed_first_message_tokens(messages),
                          estimate_conversation_tokens_cached(messages))
                parent_message_id = 0
            else:
                token_id = sess["token_id"]
                session_id = sess["session_id"]
                parent_message_id = sess["parent_message_id"]

    tok = await _db(get_token, token_id)
    if not tok:
        return JSONResponse({"error": "Token expired"}, status_code=503)
    _set_key_name(tok.get("alias"))

    # B1 (Stage 1 audit): the accumulated-context rollover used to run BEFORE
    # the per-chat lock was acquired — only first-time creation was guarded by
    # the sig-lock — so two concurrent requests with the same signature could
    # both decide "rollover", both delete the session rows and both create a
    # fresh upstream chat (last save_session() wins; the loser chat leaks and
    # parent ids diverge). The whole rollover decision now happens under the
    # CURRENT chat's lock, and the stored session state is re-read once the
    # lock is held: a concurrent same-signature request may have already
    # rolled the chat over (or advanced its parent) while this frame waited.
    lock_owner = await _own_chat_lock(session_id)
    lock_transferred = False
    slot = None  # B3: in-flight token reservation for the send below
    try:
        fresh = await _db(find_session, sig)
        if fresh:
            if fresh["session_id"] != session_id:
                # The chat moved under us (a concurrent request already rolled
                # it over). Follow it and hold the NEW chat's lock instead.
                lock_owner.release()
                token_id = fresh["token_id"]
                session_id = fresh["session_id"]
                parent_message_id = fresh["parent_message_id"]
                tok = await _db(get_token, token_id)
                if not tok:
                    return JSONResponse({"error": "Token expired"}, status_code=503)
                _set_key_name(tok.get("alias"))
                lock_owner = await _own_chat_lock(session_id)
                # B1 residual (Stage 1 review, finding 3): the parent adopted
                # above was read while we still held the OLD chat's lock; the
                # awaits since then (get_token, acquiring the NEW chat's lock)
                # gave a same-signature request a window to complete a turn on
                # this chat — sending with that stale parent would fork it,
                # the exact bug class B1 closes. Re-read under the fresh lock,
                # exactly like the rollover branch below.
                fresh = await _db(find_session, sig)
                if fresh and fresh["session_id"] == session_id:
                    parent_message_id = fresh["parent_message_id"]
            else:
                # Same chat: adopt the stored parent so a request that was
                # queued behind a completed turn never re-sends a stale
                # parent_message_id (which would fork the upstream exchange).
                parent_message_id = fresh["parent_message_id"]

        _rollover_baseline, _creation_total = await _db(get_chat_baseline, session_id)
        _tail_text = None  # [v77.5/B] 换本时的最近原文尾巴（未换本则 None）
        # [v72] 定期重建：除了上下文超限，会话轮数超过阈值也触发同一套
        # rollover 流程（删映射→生成摘要→新上游会话→按客户端真实历史全量
        # 重打包，重建后首条路径自动带任务锚点+对账单）。动机：长会话中
        # 上游历史积累畸形输出与旧叙事偏置（生产实证 2026-10-25 会话
        # ae65e2aa：数小时旧叙事淹没 30 秒前的金手指指令），且滑窗会无声
        # 丢失中段信息；定期重建等于定期把模型的世界观对齐到客户端账本。
        _rebuild_due = parent_message_id >= int(os.getenv("DEEPSEEKER_REBUILD_PARENT", "240"))
        if parent_message_id != 0 and (needs_rollover(messages, _rollover_baseline, tools=tools, creation_total=_creation_total) or _rebuild_due or _force_rollover):
            logger.info(
                "Context rollover: accumulated context over limit%s; purging session mappings for chat %s",
                " (periodic rebuild: parent=%d)" % parent_message_id if _rebuild_due else "",
                session_id,
            )
            await _db(delete_sessions_for_chat, token_id, session_id)
            # [v77.5/B] 接力摘要三件套：
            #   1) 滚动摘要——只消化"上次消化点之后、原文尾巴之前"的增量，
            #      旧内容由上一代摘要保住，代代相传，无天花板；
            #   2) 最近原文——尾巴段原样进新会话首条，压缩损失不碰最新上下文；
            #   3) 分段接力——切片超过 DIGEST_CHUNK_TOKENS 时按消息切块顺序
            #      消化（每块 = 旧摘要 + 块文本 → 更新版摘要），永不硬吞。
            _old_roll, _roll_marker = await _db(get_rolling_state, session_id)
            _slice_msgs, _tail_msgs, _sizes, _total_now = compute_rollover_split(
                messages, marker=_roll_marker, tail_budget=int(ROLLOVER_TAIL_TOKENS * 0.9))
            rollover_summary = _old_roll
            _chunks = chunk_messages(_slice_msgs, DIGEST_CHUNK_TOKENS)
            for _ci, _chunk in enumerate(_chunks):
                _ctext = _messages_plain_text(_chunk)
                _dprompt = build_rolling_digest_prompt(rollover_summary, _ctext, has_old=bool(rollover_summary))
                _sc = await create_new_chat(tok["token"])
                _sgen = send_message(
                    _sc, tok["token"], _dprompt, 0, False, False, []
                )
                rollover_summary = strip_summary_tags(await collect_response(_sgen))[: MAX_SUMMARY_TOKENS * 4]
                logger.info(
                    "Rollover digest chunk %d/%d: +%d tokens -> rolling summary %d tokens",
                    _ci + 1, len(_chunks), count_tok(_ctext), count_tok(rollover_summary),
                )
            _new_marker = sum(_sizes[: len(messages) - len(_tail_msgs)])
            _tail_text = render_history_tail(_tail_msgs, ROLLOVER_TAIL_TOKENS) if len(_tail_msgs) > 1 else ""
            session_id = await create_new_chat(tok["token"])
            await _db(save_session, sig, token_id, session_id, 0)
            # [v77.3] rollover 后记账：baseline=摘要版首条的实际 token（很小），
            # creation_total=客户端全量（摘要代表全部旧历史）——增长从此处
            # 重新起算，旧历史不再被算成"增长"（风暴根源）。
            # [v77.7] baseline 必须含尾巴重量：换本后的新首条 = 摘要 + 尾巴
            # （ROLLOVER_TAIL_TOKENS）+ 最新消息。漏算尾巴会导致下次换本在
            # "尾巴 + 阈值"时才触发——尾巴越大，中途静默遗忘越多。
            await _db(set_chat_baseline, session_id,
                      len(rollover_summary) // 3 + ROLLOVER_TAIL_TOKENS + estimate_conversation_tokens_cached(messages[-1:]),
                      estimate_conversation_tokens_cached(messages))
            # [v77.5/B] 滚动摘要记账：本代已消化到"尾巴起点"
            await _db(set_rolling_state, session_id, rollover_summary, _new_marker)
            parent_message_id = 0
            # The rollover itself was serialized under the OLD chat's lock;
            # the send -> save section below must hold the FRESH chat's lock.
            # Queued same-signature requests re-derive the new mapping from
            # the DB via the re-read above, so nobody double-rolls-over.
            lock_owner.release()
            lock_owner = await _own_chat_lock(session_id)
            # A request that read the fresh mapping in the window between our
            # save and our acquisition may have already appended to this
            # chat; adopt the stored parent so we never fork it.
            fresh = await _db(find_session, sig)
            if fresh and fresh["session_id"] == session_id:
                parent_message_id = fresh["parent_message_id"]

        # B3: every send reserves one in-flight slot against its token —
        # pick_token() balances by these counts, so the pairing must hold for
        # both the freshly picked (create path) and the session-owned token.
        # Streams take the slot with them via _release_chat_lock_stream; every
        # other exit releases it in the finally below.
        slot = acquire_token_slot(token_id)

        is_first = parent_message_id == 0

        # [v74] 重试保险丝：窗口内同签名（完全相同的消息列表）已完成 → 直接
        # 回放缓存结果，不重复执行。官方 API 因无状态而天然幂等；此为有状态
        # 通道的对位机制。仅拦非 Anthropic 路径；窗口 120 秒可调
        # （DEEPSEEKER_DUP_WINDOW）。人类"重新生成"通常先删上一条回复，
        # 消息列表变化 → 签名不同 → 不受影响。
        if not is_anthropic:
            cached_body = _dup_lookup(sig)
            _dup_dbg(f"lookup sig={str(sig)[:16]} hit={cached_body is not None} cache_n={len(_DUP_CACHE)}")
            if cached_body is not None:
                logger.warning(
                    "[dup-guard] sig=%s 窗口内已完成，回放缓存结果（不重复执行）",
                    str(sig)[:12],
                )
                if stream:
                    return StreamingResponse(
                        _pseudo_stream(cached_body, req_model or model),
                        media_type="text/event-stream",
                    )
                return cached_body

        # Stage 0.3: the lock is held across the whole send -> save critical
        # section; for streams, ownership transfers to the response generator
        # via _release_chat_lock_stream (the final save_session happens there).
        file_ids = await extract_and_upload_files(messages, tok["token"], last_user_only=not is_first)
        # B4: references pinned to another account would 404 upstream — copy
        # them onto this chat's token. Fresh uploads from this request (and
        # re-homed copies) are recorded; references that already have an owner
        # keep it (first owner wins).
        if file_ids:
            ref_set = set(ref_ids)
            file_ids = await _rehome_foreign_files(file_ids, token_id, tok)
            for fid in file_ids:
                if fid not in ref_set:
                    await _db(record_file, fid, token_id)
        prompt = await build_prompt(messages, tools or [], model, is_first,
                                    rollover_summary=rollover_summary,
                                    rollover_tail_text=_tail_text)
        if json_mode:
            # [v73] JSON 模式：提示层强约束（第一层防线；第二层出口校验在交付前）
            prompt += (
                "\n\n[OUTPUT FORMAT · 强制]\n"
                "Respond with ONE valid JSON value (object, array, string, number or boolean) and NOTHING else.\n"
                "No markdown fences, no explanations, no text before or after the JSON.\n"
                "If you still need tool results, call tools first; once data is sufficient, output ONLY the final JSON."
            )
        if resume_recovered and not is_first:
            # 接回中断会话：压制历史旧指令，防止模型反问"指令怎么不一样了"
            prompt = RESUME_NOTICE + prompt
        if tools and not is_first:
            # 工具指令漂移修复 v2：完整工具说明只在会话首条注入，会被官网
            # 滑窗（~65 万）逐渐挤出；且历史里积累的畸形 DSML 输出会被模型
            # 模仿（恶性循环）。对策：每轮在最新消息末尾重申迷你铁律——它
            # 永远在滑窗之内，长对话中遵从度不衰减。成本 ~100 token/轮。
            # [v75-c] 警告牌去粘性：旧实现扫最近 6 条消息，客户端历史里滞留的
            # 旧拒绝记录会让调用成功后仍持续贴牌（生产实证 2026-10-25 会话
            # e1751fe3：11 次贴牌 3 次误贴，全是成功后误贴，加剧模型格式摇摆）。
            # 新实现只认"最近一条工具结果"：窗口内最后一条 role=tool 消息是
            # 拒绝才贴牌；陈年错误（滑出最近 3 条）不翻出来骂。历史全量统计：
            # 旧规则 8 正确 + 3 误贴 + 0 漏贴；新规则 8 正确 + 0 误贴 + 0 漏贴。
            _last_tool_rejected = False
            for _m in messages[-3:]:
                if isinstance(_m, dict) and _m.get("role") == "tool":
                    _last_tool_rejected = bool(re.search(
                        r"Tool call rejected|missing required parameter|参数为空|arguments (?:were |)empty|invalid tool call",
                        str(_m.get("content") or ""), re.I))
            if _last_tool_rejected:
                # 客户端回传了工具拒绝/参数错误反馈：用户不必再手动提醒，
                # 服务端检测到后自动加码。
                prompt += (
                    "\n\n[⚠️ 上一次工具调用已被系统拒绝]\n"
                    "历史中检测到工具调用被拒绝的记录（参数为空或缺失）。"
                    "这一次必须严格按下述规范执行，任何再次被拒绝的调用都会终止任务。"
                    "特别注意第 7 条收尾纪律：闭合后立即停止，禁止输出任何残片垃圾——"
                    "历史中已多次出现这类垃圾（\"参数>\"、\"???参数>\"），它们会导致调用被整条拒绝："
                )
            prompt += (
                "\n\n[工具调用规范 · 每轮重申]\n"
                "1. 需要工具时：输出一个工具调用块，块内为合法 JSON，含 name 与 arguments 两个键。\n"
                "2. arguments 的每个必填字段都要有具体值；空对象或字段缺失的调用会被系统直接拒绝。\n"
                "3. 参数值写完立即闭合，闭合后不得再追加任何字符。\n"
                "4. 不需要工具时直接给正文，不输出任何标签。\n"
                "5. 历史中出现过的畸形调用样例一律忽略，只以本规范为准。\n"
                "6. 各工具的参数定义见对话开头的工具说明；记不清必填项就先向用户确认。"
            )

        _dump_prompt_debug(session_id, prompt)
        gen = send_message(session_id, tok["token"], prompt, parent_message_id, thinking, search, file_ids)
        gen = await _preflight_stream(gen)
        if stream and tools and not is_anthropic and os.getenv("DEEPSEEKER_STREAM_SPLITTER", "1") != "1":
            # 降级路径（开关关闭时）：工具轮走可靠的非流式管线 + 伪流式输出。
            # 默认关闭——v14 的 _StreamTagSplitter 已把流式标记解析改为
            # 跨 chunk 安全（穷举切分一致性测试通过），无需牺牲首字节。
            # 真出问题随时 DEEPSEEKER_STREAM_SPLITTER=0 秒回退。
            result = await handle_chat(
                messages, model, thinking, False, tools, is_anthropic,
                req_model, scope, _attempt=_attempt,
                _auth_rotated=_auth_rotated, _exclude_token=_exclude_token,
            )
            try:
                body = json.loads(bytes(result.body).decode("utf-8"))
            except Exception:
                return result
            if "choices" not in body:
                return result  # 上游错误原样透传
            return StreamingResponse(
                _pseudo_stream(body, model),
                media_type="text/event-stream",
            )
        if stream:
            gen = _release_chat_lock_stream(gen, lock_owner, slot)
            lock_transferred = True
            if is_anthropic:
                return StreamingResponse(stream_anthropic_response(gen, model, messages, token_id, session_id, sig, tools, req_model, parent_message_id, scope), media_type="text/event-stream")
            return StreamingResponse(stream_response(gen, model, messages, token_id, session_id, sig, tools, parent_message_id, scope), media_type="text/event-stream")
        else:
            resp_text = await collect_response(gen)
            _dump_stream_debug(session_id, resp_text)
            await _db(mark_active, token_id)

            if json_mode:
                # [v73] 第二层防线：出口校验 + 修复（提取合法 JSON，失败原样放行）
                resp_text, _json_ok = _json_mode_coerce(resp_text)
            parsed_tools, clean_text = parse_tools(resp_text)
            clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
            clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = await generate_signature(next_messages, model, scope)

            await _db(save_session, sig, token_id, session_id, next_parent(parent_message_id))
            await _db(save_session, next_sig, token_id, session_id, next_parent(parent_message_id))
            _resp = format_response(resp_text, model, messages, tools)
            # [v77 P0-6 审计补漏] OpenAI 非流式完成轮的用量落库——此前只在
            # 流式（stream_response）与 Anthropic 非流式路径落库，OpenAI
            # 非流式完成轮漏记（线上实证 2026-10-27：工具轮无 usage 行）。
            # 此处即成功路径（失败走上方 except），无需 failed 判断
            with suppress(Exception):
                _u = (_resp or {}).get("usage") or {}
                await _db(record_usage, scope or "unknown", "openai",
                          req_model or model,
                          _u.get("prompt_tokens") or _count_messages_tokens(messages),
                          _u.get("completion_tokens") or count_tok(_completion_usage_text(clean_text, parsed_tools)))
            # [v74] 记录完成结果：窗口内同签名重试直接回放，不重复执行
            if isinstance(_resp, dict) and "choices" in _resp:
                _dup_record(sig, _resp)
                _dup_dbg(f"record sig={str(sig)[:16]}")
            return _resp
    except asyncio.CancelledError:
        # B2 (Stage 1 audit): the client went away mid-request — the exchange
        # never completed upstream, so the stored parent_message_id would fork
        # the conversation on the next turn. Purge the rows and let the
        # cancellation propagate; the finally below releases the chat lock.
        await _db(delete_sessions_for_chat, token_id, session_id)
        raise
    except Exception as e:
        code = _upstream_http_code(e)
        if code in (401, 403, 429):
            await _db(mark_limited, token_id, code)
            await _db(delete_sessions_for_chat, token_id, session_id)
            if not _auth_rotated:
                new_token_id = await _db(pick_token)
                if new_token_id and new_token_id != token_id:
                    logger.warning(
                        "Upstream HTTP %s on token #%s (session %s); rotating to token #%s",
                        code,
                        token_id,
                        session_id,
                        new_token_id,
                    )
                    lock_owner.release()
                    return await handle_chat(
                        messages,
                        model,
                        thinking,
                        search,
                        stream,
                        tools,
                        is_anthropic,
                        req_model,
                        scope,
                        _attempt=_attempt,
                        _auth_rotated=True,
                        _exclude_token=token_id,
                    )
            logger.warning(
                "Chat request rejected by upstream (session %s, parent %s): %s",
                session_id,
                parent_message_id,
                e,
            )
            return _api_error_response(e, is_anthropic)

        logger.exception("Chat request failed (session %s, parent %s): %s", session_id, parent_message_id, e)
        await _db(delete_sessions_for_chat, token_id, session_id)
        # [v77 PR#67 移植] 上下文属于上游 CHAT 而不属于 TOKEN——普通重试
        # （换号、同会话）必然同样死亡。分类为上下文型失败时，重试强制走
        # 摘要 rollover；每轮最多一次（force 标志不粘滞，review fix）。
        _force_rollover_next = bool(is_context_limit_error(e)) and not _force_rollover
        if _force_rollover_next:
            logger.warning(
                "Context-class upstream failure on chat %s (%s); retry will force a summary rollover",
                session_id,
                e,
            )
        if _attempt + 1 >= MAX_UPSTREAM_ATTEMPTS:
            # B10: budget exhausted — surface the (already redacted/truncated)
            # upstream error instead of retrying forever.
            return _api_error_response(e, is_anthropic)
        # PR #26 review fix (Blocker 2 — retry self-deadlock): the recursive
        # call can resolve to the SAME chat (the retry re-derives the session
        # from whatever the persistence layer still returns). Recursing while
        # this frame still owns the lock made the retry wait on a lock its own
        # caller held — the request hung until the client gave up, on exactly
        # the errors the retry exists for. Surrender the lock first: release()
        # is scoped to this holder and idempotent, so the finally below becomes
        # a no-op, the retry re-acquires cleanly, and queued same-chat requests
        # are no longer starved for the entire retry either.
        lock_owner.release()
        # B10: jittered backoff, then retry on a DIFFERENT token — the old
        # single retry re-entered pick_token()'s random draw and could land on
        # the same poisoned token/session again (the #33 symptom persisting).
        await asyncio.sleep(random.uniform(0.25, 0.75) * (1.5 ** _attempt))
        return await handle_chat(
            messages,
            model,
            thinking,
            search,
            stream,
            tools,
            is_anthropic,
            req_model,
            scope,
            _attempt=_attempt + 1,
            _auth_rotated=_auth_rotated,
            _exclude_token=token_id,
            _force_rollover=_force_rollover_next,
        )
    finally:
        if not lock_transferred:
            lock_owner.release()
            if slot is not None:
                slot.release()


# --- usage 计费 token 的增量缓存 (v57 移植 v37 + 修正) ---------------------
# count_tok(_messages_text(messages)) 每轮都把整段历史重新 tokenize 一遍，
# 算出来的数字只塞进 usage 字段。上下文越大越慢：25% CPU 配额下 80 万
# token 要 2-5 秒，而 Anthropic 路径把这个计算放在首字节之前，直接压在
# TTFB 上。正常对话每轮重发的都是同一份前缀，因此按消息前缀缓存，命中
# 最长前缀后只 tokenize 新增的尾部几条。
#
# [v57 修正1 复合漂移] v37 原版把"派生估计值"也存进缓存当基值，误差逐轮
# 复合——实测 50 轮对话漂移累积到 15 token（每轮 +1：拼接处的 "\n" 在
# 全量文本里与相邻字符合并，分开计则各算各的）。本版规则：入缓存的只能
# 是"直接重算"的值，派生估计（基值 + 裸计尾条）绝不入缓存——它只回答
# 本轮，不当基值。误差因此退化为单点拼接效应，<=1 token 且不复合
# （50 轮随机对话实测最大 1）。
#
# [v57 修正2 双口径] v37 的 miss 路径返回并缓存 文本+"\n" 的计数，而
# 上游 test_stream_usage 断言 usage.prompt_tokens 等于裸文本直算值——
# 尾部 "\n" 多算一个 token 就会差 1。因此缓存条目改存两个直接重算值：
#   direct = count_tok(文本)          精确命中时的返回值，与 v33 直算口径
#                                    一致（也是上游断言要求的口径）；
#   base   = count_tok(文本 + "\n")   仅作尾条拼接的基值：尾部分隔符算进
#                                    基值，下一轮的拼接点才与全量文本一致。
# 精确命中返回 direct；带尾条命中返回 base + 裸计尾条。派生值一律不入
# 缓存，直接重算从"仅首轮"变成"约每 9 轮一次"（8 深度窗口内没有直接值
# 时可命中；仍在首字节前，比每轮全量重算快 9 倍）。
_USAGE_TOK_CACHE = None
_USAGE_TOK_CACHE_MAX = 2048


def _usage_prefix_key(messages):
    import hashlib
    h = hashlib.sha256()
    for m in messages:
        h.update(b"\x00")
        h.update(str(m.get("role", "")).encode("utf-8", "ignore"))
        c = m.get("content", "")
        blob = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False, default=str)
        h.update(blob.encode("utf-8", "ignore"))
    return h.hexdigest()


def _count_messages_tokens(messages):
    global _USAGE_TOK_CACHE
    if _USAGE_TOK_CACHE is None:
        from collections import OrderedDict
        _USAGE_TOK_CACHE = OrderedDict()
    cache = _USAGE_TOK_CACHE
    if not messages:
        return 0
    n = len(messages)
    for back in range(0, min(8, n) + 1):
        prefix = messages[: n - back] if back else messages
        key = _usage_prefix_key(prefix)
        cached = cache.get(key)
        if cached is None:
            continue
        if back:
            # 基值含尾部分隔符 "\n"，尾条裸计；派生值绝不入缓存
            return cached[1] + count_tok(_messages_text(messages[n - back:]))
        return cached[0]
    # 窗口内没有直接值：两个口径都直接重算，一起入缓存。
    # 只存直接值——派生值不当基值，误差因此不会逐轮复合。
    direct = count_tok(_messages_text(messages))
    base = count_tok(_messages_text(messages) + "\n")
    full_key = _usage_prefix_key(messages)
    cache[full_key] = (direct, base)
    cache.move_to_end(full_key)
    while len(cache) > _USAGE_TOK_CACHE_MAX:
        cache.popitem(last=False)
    return direct


async def collect_response(gen):
    text = ""
    async for chunk in gen:
        text += chunk
    return text


def _messages_text(messages):
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            parts.append(" ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"))
        else:
            parts.append(str(c))
    return "\n".join(parts)


async def _hold_think_tags(gen):
    carry = ""
    async for chunk in gen:
        chunk = carry + chunk
        carry = ""
        hold = 0
        for tag in ("<think>", "</think>"):
            for i in range(1, len(tag)):
                if chunk.endswith(tag[:i]):
                    hold = max(hold, i)
        if hold:
            carry = chunk[-hold:]
            chunk = chunk[:-hold]
        if chunk:
            yield chunk
    if carry:
        yield carry


def _chat_chunk(choices, cid, created, model):
    """Build an OpenAI chat.completion.chunk with all fields strict clients require.

    Some clients (Vercel AI SDK used by Trilium, Zed) validate every streamed
    chunk against OpenAI's schema and reject frames missing `index` (and often
    `id`/`object`/`created`). Emit them all so the stream is spec-conformant.

    cid/created are REQUIRED on purpose: one completion must be exactly one
    id/created, generated once per stream and threaded through every chunk.
    A silent `cid or ("chatcmpl-" + uuid4())` fallback minted a fresh id per
    chunk (the Stage 2 bug) — a call site that forgets to pass cid must now
    fail loudly instead of corrupting the stream.
    """
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }


def _choice(delta, index=0, finish_reason=None):
    return {"index": index, "delta": delta, "finish_reason": finish_reason}


class _StreamTagSplitter:
    """跨 chunk 安全的 <think> 分离器（真流式零损坏）。

    旧的逐 chunk 状态机对每个分片单独判断 `"</think>" in chunk`：标记
    一旦跨分片（`</th` + `ink>`）就漏判，状态错乱 → 思考内容泄漏进正文、
    相邻字节被吞（replace/split 丢字节）、并把 DSML 之类的其它标记
    切碎直达客户端。

    本实现只在缓冲区里"完整匹配到标记"才处理，并把尾部"可能是标记
    前缀"的一小段扣留等待后续数据；所有输出字节都取自原文，从不做
    replace/split。只有 <think>/</think> 的前缀会被扣留——其它以 '<'
    开头的标记（<｜DSML｜> 工具家族等）一律原样透传，绝不切碎。"""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.buf = ""
        self.in_think = False
        self.think_done = False

    def _hold(self, tag):
        """把 buf 切成 (可安全输出, 需扣留)。"""
        if not tag:
            return self.buf, ""
        idx = self.buf.rfind("<")
        if idx == -1:
            return self.buf, ""
        suffix = self.buf[idx:]
        if len(suffix) < len(tag) and tag.startswith(suffix):
            return self.buf[:idx], suffix
        return self.buf, ""

    def feed(self, chunk):
        """返回 [(kind, text)]，kind ∈ {"reasoning", "content"}。"""
        self.buf += chunk
        out = []
        while self.buf:
            if self.in_think:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    safe, self.buf = self._hold(self.CLOSE)
                    if safe:
                        out.append(("reasoning", safe))
                    return out
                if i:
                    out.append(("reasoning", self.buf[:i]))
                self.buf = self.buf[i + len(self.CLOSE):]
                self.in_think = False
                self.think_done = True
                continue
            i = -1 if self.think_done else self.buf.find(self.OPEN)
            if i == -1:
                safe, self.buf = self._hold(None if self.think_done else self.OPEN)
                if safe:
                    out.append(("content", safe))
                return out
            if i:
                # [v59] 开标签前的纯空白是 _send_message_once "\n<think>"  yield 的
                # 前导分隔符，不属于正文：放行会让每个流式轮次的 content 都被
                # 漏进一个孤换行（工具轮正文恰好只有它，2026-10-24 用户报告）。
                # Anthropic 路径早就 lstrip("\n") 剥掉它，此处对齐同一语义；
                # 有实际文字的前导内容（模型真的先写正文再思考）仍原样放行。
                pre = self.buf[:i]
                if pre.strip():
                    out.append(("content", pre))
            self.buf = self.buf[i + len(self.OPEN):]
            self.in_think = True
            continue
        return out

    def flush(self):
        out = []
        if self.buf:
            out.append(("reasoning" if self.in_think else "content", self.buf))
            self.buf = ""
        return out


async def _pseudo_stream(result: dict, model: str):
    """把非流式的完整响应切成 SSE chunk 伪流式吐出（工具轮可靠管线）。

    一次性给完 tool_calls（OpenAI 流式允许参数整体到达）；正文分 8 片，
    片间微延迟让客户端有流式观感。"""
    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    choice = result["choices"][0]
    msg = choice.get("message", {})
    fin = choice.get("finish_reason") or "stop"
    content = msg.get("content") or ""
    if msg.get("reasoning_content"):
        yield f"data: {json.dumps(_chat_chunk([_choice({'reasoning_content': msg['reasoning_content']})], cid=cid, created=created, model=model))}\n\n"
    step = max(1, len(content) // 8)
    for i in range(0, len(content), step):
        yield f"data: {json.dumps(_chat_chunk([_choice({'content': content[i:i + step]})], cid=cid, created=created, model=model))}\n\n"
    if msg.get("tool_calls"):
        yield f"data: {json.dumps(_chat_chunk([_choice({'tool_calls': msg['tool_calls']})], cid=cid, created=created, model=model))}\n\n"
    yield f"data: {json.dumps(_chat_chunk([_choice({}, finish_reason=fin)], cid=cid, created=created, model=model))}\n\n"
    yield "data: [DONE]\n\n"


async def stream_response(gen, model, messages, token_id, session_id, sig, tools, parent_message_id=0, scope=""):
    parser = StreamToolParser()
    # One completion = one id/created: every SSE chunk of a stream must share
    # the same id + created so clients and gateways (New API, sub2api) can
    # reassemble and bill it as a single completion. Generated once here and
    # threaded through every chunk below — mirrors msg_id in
    # stream_anthropic_response (which already does this correctly).
    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    full_text = ""
    is_thinking = False
    aborted = False
    failed = False
    try:
        splitter = _StreamTagSplitter()
        async for chunk in gen:
            # [v61] 续写轮边界哨兵：空串 str 子类，必须先于下面的 falsy 检查
            # 拦截。重置 splitter——其 think_done 闭锁（见 _StreamTagSplitter
            # / functions._RetryBoundary 文档）会让续写轮开头的 "\n<think>"
            # 不再被识别，整段思考漏进 content。重置后续写轮的思考正确分流
            # 到 reasoning。丢弃的是 attempt 尾部 ≤6 字符的 held 标签前缀，
            # 续写轮以 "\n" 开头不依赖跨 attempt 的标签完整性。
            if isinstance(chunk, _RetryBoundary):
                splitter = _StreamTagSplitter()
                continue
            if not chunk:
                continue
            full_text += chunk
            for kind, text in splitter.feed(chunk):
                if not text:
                    continue
                if kind == "reasoning":
                    yield f"data: {json.dumps(_chat_chunk([_choice({'reasoning_content': text})], cid=cid, created=created, model=model))}\n\n"
                else:
                    for r in parser.feed(text):
                        if "text" in r:
                            yield f"data: {json.dumps(_chat_chunk([_choice({'content': r['text']})], cid=cid, created=created, model=model))}\n\n"
        for kind, text in splitter.flush():
            if not text:
                continue
            if kind == "reasoning":
                yield f"data: {json.dumps(_chat_chunk([_choice({'reasoning_content': text})], cid=cid, created=created, model=model))}\n\n"
            else:
                for r in parser.feed(text):
                    if "text" in r:
                        yield f"data: {json.dumps(_chat_chunk([_choice({'content': r['text']})], cid=cid, created=created, model=model))}\n\n"
        _dump_stream_debug(session_id, full_text)
        await _db(mark_active, token_id)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        # B2 (Stage 1 audit): the exchange never completed upstream. The stored
        # session rows still point at the pre-abort parent_message_id, so the
        # next turn would reuse them and fork/duplicate the conversation.
        # Purge them — the next request opens a fresh chat, which is already
        # the supported first-message path. Deletion must not be skipped even
        # while the task is being torn down, hence it lives here and not in
        # the finally block.
        try:
            await _db(delete_sessions_for_chat, token_id, session_id)
        except Exception:
            logger.exception("stream_response: failed to purge session rows after client abort")
        raise
    except Exception as e:
        failed = True
        code = _upstream_http_code(e)
        if code in (401, 403, 429):
            await _db(mark_limited, token_id, code)
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.warning("stream_response upstream HTTP %s: %s", code, e)
        else:
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.exception("stream_response failed")
        try:
            yield f"data: {json.dumps({'error': {'message': str(e)[:300]}})}\n\n"
        except Exception:
            pass
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

        if not failed:
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            await _db(save_session, sig, token_id, session_id, next_parent(parent_message_id))
            await _db(save_session, next_sig, token_id, session_id, next_parent(parent_message_id))

        if not aborted and not failed:
            # [v74] 记录完成结果：窗口内同签名重试直接回放，不重复执行
            try:
                _dup_record(sig, {
                    "id": cid, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": ast_msg,
                                 "finish_reason": "tool_calls" if parsed_tools else "stop"}],
                })
            except Exception:
                pass
            try:
                if not parsed_tools:
                    for r in parser.flush():
                        if "text" in r:
                            yield f"data: {json.dumps(_chat_chunk([_choice({'content': r['text']})], cid=cid, created=created, model=model))}\n\n"

                if parsed_tools:
                    for i, tc in enumerate(parsed_tools):
                        delta_tc = {"index": i, "id": tc["id"], "type": "function",
                                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                        yield f"data: {json.dumps(_chat_chunk([_choice({'tool_calls': [delta_tc]})], cid=cid, created=created, model=model))}\n\n"
                    yield f"data: {json.dumps(_chat_chunk([_choice({}, finish_reason='tool_calls')], cid=cid, created=created, model=model))}\n\n"
                else:
                    yield f"data: {json.dumps(_chat_chunk([_choice({}, finish_reason='stop')], cid=cid, created=created, model=model))}\n\n"
                # OpenAI-compatible gateways (New API, sub2api, ...) read billing
                # usage from the trailing usage chunk. Always emit it here so
                # clients that omit stream_options.include_usage still get counted:
                # the empty choices array is what gateways expect, and official
                # SDKs simply ignore it. Skipped on aborted/failed streams so a
                # partial response never pollutes billing.
                in_tokens = _count_messages_tokens(messages)
                # B7: usage on the cleaned completion, not raw full_text —
                # think-tag reasoning and tool markup are not billed output.
                usage_out = _completion_usage_text(clean_text, parsed_tools)
                out_tokens = count_tok(usage_out) if usage_out else 0
                usage_chunk = _chat_chunk([], cid=cid, created=created, model=model)
                usage_chunk["usage"] = {"prompt_tokens": in_tokens, "completion_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
                yield f"data: {json.dumps(usage_chunk)}\n\n"
                # [v77 P0-6 移植] 完成轮用量落库（客户端可见成功才记，
                # 失败/中止流不记——重试只计一次是结构性规则）
                with suppress(Exception):
                    await _db(record_usage, scope or "unknown", "openai", model, in_tokens, out_tokens)
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                pass


async def stream_anthropic_response(gen, model, messages, token_id, session_id, sig, tools, req_model=None, parent_message_id=0, scope=""):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    in_tokens = _count_messages_tokens(messages)
    model_name = req_model if req_model else model
    start_evt = f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': in_tokens, 'output_tokens': 1}}})}\n\n"
    yield start_evt

    parser = StreamToolParser()
    full_text = ""
    text_block_started = False
    block_index = 0
    aborted = False
    failed = False

    try:
        is_thinking = False
        async for chunk in _hold_think_tags(gen):
            if not chunk:
                continue
            full_text += chunk

            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "").lstrip("\n")
                start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking'}})}\n\n"
                yield start_block

            end_thinking = False
            if "</think>" in chunk:
                is_thinking = False
                end_thinking = True
                parts = chunk.split("</think>")
                think_part = parts[0]
                chunk = parts[1].lstrip("\n") if len(parts) > 1 else ""
                if think_part:
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': think_part}})}\n\n"
                    yield delta_evt

            if is_thinking and chunk:
                delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': chunk}})}\n\n"
                yield delta_evt
                continue

            if end_thinking:
                stop_evt = f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                yield stop_evt
                block_index += 1
                if not chunk:
                    continue

            for r in parser.feed(chunk):
                if "text" in r:
                    if not text_block_started:
                        start_block = f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield start_block
                        text_block_started = True
                    delta_evt = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': r['text']}})}\n\n"
                    yield delta_evt
        await _db(mark_active, token_id)
    except (asyncio.CancelledError, GeneratorExit):
        aborted = True
        failed = True
        # B2 (Stage 1 audit): same purge as stream_response — a client abort
        # mid-stream leaves session rows pointing at a parent the upstream
        # chat never answered, and the next turn would fork the exchange.
        try:
            await _db(delete_sessions_for_chat, token_id, session_id)
        except Exception:
            logger.exception("stream_anthropic_response: failed to purge session rows after client abort")
        raise
    except Exception as e:
        failed = True
        code = _upstream_http_code(e)
        if code in (401, 403, 429):
            await _db(mark_limited, token_id, code)
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.warning("stream_anthropic_response upstream HTTP %s: %s", code, e)
        else:
            await _db(delete_sessions_for_chat, token_id, session_id)
            logger.exception("stream_anthropic_response failed")
        try:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)[:300]}})}\n\n"
        except Exception:
            pass
    finally:
        parsed_tools, clean_text = parse_tools(full_text)
        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()
        # B7: usage on the cleaned completion, not raw full_text. Computed
        # AFTER the strips above (mirrors stream_response): this path used to
        # count tokens before them, so /v1/messages streams billed the whole
        # <think> reasoning share as output_tokens (Stage 1 review, finding 1).
        usage_out = _completion_usage_text(clean_text, parsed_tools)
        out_tokens = count_tok(usage_out) if usage_out else 0

        if not failed:
            # [v77 P0-6 移植] 非流式完成轮用量落库
            with suppress(Exception):
                await _db(record_usage, scope or "unknown", "anthropic", req_model or model,
                          _count_messages_tokens(messages), out_tokens)
            next_messages = messages.copy()
            ast_msg = {"role": "assistant"}
            if parsed_tools:
                ast_msg["tool_calls"] = parsed_tools
            else:
                ast_msg["content"] = clean_text
            next_messages.append(ast_msg)
            next_sig = generate_signature_sync(next_messages, model, scope)
            await _db(save_session, sig, token_id, session_id, next_parent(parent_message_id))
            await _db(save_session, next_sig, token_id, session_id, next_parent(parent_message_id))

        # Declared BEFORE _tb so the helper's closure reads top-down — it used
        # to be defined after _tb and worked only by late binding (Stage 1
        # minor list: hostile to readers).
        block_index_local = [block_index]

        def _tb(text):
            return (f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                    f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n")

        tail_events = ""
        if is_thinking:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
            block_index_local[0] += 1

        flushed_text = ""
        if not parsed_tools:
            for r in parser.flush():
                if "text" in r:
                    flushed_text += r["text"]

        if not text_block_started and not parsed_tools and (clean_text or flushed_text):
            tail_events += _tb(clean_text or flushed_text)
        elif text_block_started and not parsed_tools:
            tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"

        if parsed_tools:
            for tc in parsed_tools:
                tool_input = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                json_str = json.dumps(tool_input)
                tail_events += f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index_local[0], 'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': tc['function']['name'], 'input': {}}})}\n\n"
                tail_events += f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index_local[0], 'delta': {'type': 'input_json_delta', 'partial_json': json_str}})}\n\n"
                tail_events += f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index_local[0]})}\n\n"
                block_index_local[0] += 1
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        else:
            tail_events += f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        tail_events += f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        if not aborted and not failed:
            try:
                for evt in tail_events.split("\n\n"):
                    if evt.strip():
                        yield evt + "\n\n"
            except asyncio.CancelledError:
                pass


def _completion_usage_text(clean_text, parsed_tools):
    """Text whose token count is billed as completion tokens (B7, Stage 1 audit).

    The cleaned reply — <think> reasoning and tool-call markup stripped — plus
    the serialized tool-call arguments the client actually receives. Counting
    the raw upstream text over-billed by the thinking+markup share, inflating
    completion_tokens and every gateway cost derived from them."""
    parts = [clean_text] if clean_text else []
    for tc in parsed_tools or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        if fn.get("name"):
            parts.append(str(fn["name"]))
        if fn.get("arguments"):
            parts.append(str(fn["arguments"]))
    return "\n".join(parts)


def format_response(text, model, messages, tools=None):
    from functions import DEEPSEEK_TARIFFS
    parsed_tools, clean_text = parse_tools(text)

    reasoning = None
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*", text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
    clean_text = re.sub(r"</?(?:tool_calls?|invoke|function_call|parameter)[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

    in_tokens = _count_messages_tokens(messages)
    # B7: bill the CLEANED completion (see _completion_usage_text), not the
    # raw upstream text whose thinking + markup share the client never asked
    # to pay for.
    usage_text = _completion_usage_text(clean_text, parsed_tools)
    out_tokens = count_tok(usage_text) if usage_text else 0
    tariff = DEEPSEEK_TARIFFS["deepseek-v4.1-flash"]
    cost = (in_tokens / 1_000_000 * tariff["cache_miss_input"]) + (out_tokens / 1_000_000 * tariff["output_generation"])

    msg_dict = {
        "role": "assistant",
        "content": clean_text if not parsed_tools else None,
        "tool_calls": parsed_tools if parsed_tools else None,
    }
    if reasoning:
        msg_dict["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg_dict,
            "finish_reason": "tool_calls" if parsed_tools else "stop",
        }],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "cost": round(cost, 6),
        },
    }


def format_anthropic_response(result, model):
    choice = result["choices"][0]
    msg = choice["message"]
    ant_content = []

    if msg.get("reasoning_content"):
        ant_content.append({"type": "thinking", "thinking": msg["reasoning_content"]})

    if msg.get("content"):
        ant_content.append({"type": "text", "text": msg["content"]})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"]
            tool_input = json.loads(args) if isinstance(args, str) else args
            ant_content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": tool_input,
            })
    usage = result.get("usage", {})
    msg_id = result["id"]
    if not msg_id.startswith("msg_"):
        msg_id = f"msg_{msg_id.replace('chatcmpl-', '')}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": ant_content,
        "model": model,
        "stop_reason": "tool_use" if msg.get("tool_calls") else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }



@app.post("/v1/files")
@app.post("/v1/files/upload")
async def files_upload(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    # [v77 PR#72 审计修复] effort 档位在此设置（此前链路断裂：
    # request_effort 无人 set，max 档指令从未挂上）
    _request_effort.set(effort_level(body) if isinstance(body, dict) else None)
    # [v77 P0-7] 显式模式开关：DEEPSEEKER_MODE=official 时走官方 API，
    # 无静默回退。
    if official_api.official_mode_enabled():
        _body = await request.json()
        return await _official_chat_response(
            _body, _body.get("model") or "deepseek-chat",
            bool(_body.get("stream")), _body.get("tools"), request)
    tok_id = await _db(pick_token)
    if not tok_id:
        return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = await _db(get_token, tok_id)
    if not tok:
        return JSONResponse({"error": "Token not found"}, status_code=503)
    _set_key_name(tok.get("alias"))
    slot = acquire_token_slot(tok_id)  # B3: upload counts toward the token's in-flight load
    try:
        form = await request.form()
        file_obj = form.get("file")
        if not file_obj:
            return JSONResponse({"error": "No file provided"}, status_code=400)
        file_bytes = await file_obj.read(25 * 1024 * 1024 + 1)
        if len(file_bytes) > 25 * 1024 * 1024:
            return JSONResponse({"error": "File too large"}, status_code=413)
        filename = getattr(file_obj, "filename", "file.bin")
        content_type = getattr(file_obj, "content_type", "application/octet-stream")
        file_info = None
        async for status, data in upload_file(file_bytes, filename, content_type, tok["token"]):
            if status == "success":
                file_info = data
                break
        if not file_info:
            return JSONResponse({"error": "Upload failed"}, status_code=500)
    finally:
        slot.release()
    # B4: pin the upload to the token that performed it so later chats can
    # prefer (or re-home onto) the owning account.
    await _db(record_file, file_info["file_id"], tok_id)

    if request.url.path.startswith("/v1/files/upload"):
        return {
            "id": file_info["file_id"],
            "type": "file",
            "filename": filename,
            "size": file_info["size"],
            "created_at": file_info["anthropic_timestamp"],
        }
    return {
        "id": file_info["file_id"],
        "object": "file",
        "bytes": file_info["size"],
        "created_at": file_info["openai_timestamp"],
        "filename": filename,
        "purpose": "answers",
    }


@app.get("/v1/files/{file_id}/content")
@app.get("/v1/files/{file_id}")
async def files_content(file_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    # B4 (Stage 1 review, finding 2): uploads are pinned to their token and
    # upstream files are account-scoped, so fetching with a scheduler-picked
    # token 404s whenever that token is not the owner — the same broken flow
    # B4 already fixed for chat. Prefer the registered owner and fall back to
    # the scheduler only for legacy/unregistered ids (or an owner whose token
    # row has since been removed).
    tok_id = await _db(get_file_token, file_id)
    if tok_id is not None:
        tok = await _db(get_token, tok_id)
        if tok is None:
            tok_id = None  # owner's token row is gone — degrade to the scheduler
    if tok_id is None:
        tok_id = await _db(pick_token)
        if not tok_id:
            return JSONResponse({"error": "No tokens available"}, status_code=503)
    tok = await _db(get_token, tok_id)
    if not tok:
        return JSONResponse({"error": "Token not found"}, status_code=503)
    _set_key_name(tok.get("alias"))
    slot = acquire_token_slot(tok_id)
    try:
        gen = get_file_content(tok["token"], file_id)
        try:
            mime = await gen.__anext__()
        except StopAsyncIteration:
            return JSONResponse({"error": "File not found"}, status_code=404)
        except Exception:
            return JSONResponse({"error": "File fetch failed"}, status_code=502)
    finally:
        # Released once the fetch handshake is done; the download itself
        # streams from the already-established upstream response.
        slot.release()
    async def stream_chunks():
        async for chunk in gen:
            yield chunk
    return StreamingResponse(stream_chunks(), media_type=mime or "application/octet-stream")


def is_thinking_enabled(body, request=None):
    effort = body.get("effort")
    if effort is not None:
        e_str = str(effort).strip().lower()
        if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
            return True
        if e_str in ["low", "minimal", "none", "off", "disable", "disabled", "false"]:
            return False

    out_cfg = body.get("output_config")
    if isinstance(out_cfg, dict):
        out_effort = out_cfg.get("effort") or out_cfg.get("reasoning_effort")
        if out_effort is not None:
            e_str = str(out_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                return True
            if e_str in ["low", "minimal", "none", "off", "disable", "disabled", "false"]:
                return False

    thinking_val = body.get("thinking")
    if isinstance(thinking_val, dict):
        t_type = str(thinking_val.get("type", "")).strip().lower()
        if t_type in ["enabled", "adaptive", "true"]:
            return True
        if t_type == "disabled":
            return False
        budget = thinking_val.get("budget_tokens", 0)
        if isinstance(budget, (int, float)) and budget > 0:
            return True
        t_effort = thinking_val.get("effort") or thinking_val.get("reasoning_effort") or thinking_val.get("level")
        if t_effort is not None:
            e_str = str(t_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                return True
            if e_str in ["low", "minimal", "none", "off", "disable", "disabled", "false"]:
                return False
    elif isinstance(thinking_val, str):
        t_str = thinking_val.strip().lower()
        if t_str in ["medium", "high", "max", "ultra", "extreme", "true", "enabled", "adaptive", "on"]:
            return True
        if t_str in ["low", "minimal", "none", "off", "disable", "disabled", "false"]:
            return False
    elif isinstance(thinking_val, bool):
        return thinking_val

    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort is not None:
        effort_str = str(reasoning_effort).strip().lower()
        if effort_str in ["medium", "high", "max", "ultra", "extreme", "auto", "default", "true", "on", "enabled", "adaptive", "yes"]:
            return True
        if effort_str in ["low", "minimal", "none", "off", "disable", "disabled", "false", "no"]:
            return False

    if request:
        req_effort = request.headers.get("anthropic-thinking") or request.headers.get("x-anthropic-thinking") or request.headers.get("effort") or request.headers.get("x-effort")
        if req_effort:
            e_str = str(req_effort).strip().lower()
            if e_str in ["medium", "high", "max", "ultra", "extreme", "enabled", "adaptive", "on"]:
                return True
    # Default ON: DeepSeek's web model always reasons internally; clients that
    # want it off must say so explicitly (reasoning_effort/thinking = off).
    return True


# DeepSeek now serves a single model (v4.1flash) as the website default.
# Every request — whatever model name the client sends, including legacy
# aliases (instant, expert, vision, anthropic/claude-*) — is served by it.
SINGLE_MODEL = "v4.1flash"


def resolve_model(model_raw):
    return SINGLE_MODEL


async def _json_body(request: Request):
    """Parse the body as JSON; empty/malformed payloads are a client error.

    Stage 1 minor list: the completion endpoints used to call request.json()
    directly, so an empty body or invalid JSON surfaced as a 500 — a client
    mistake reported as a server fault (and paged as one)."""
    try:
        raw = await request.body()
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read request body")
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="Empty request body; expected a JSON object")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return body


# ============ [v77 P0-7 移植] 官方 API 直通（DEEPSEEKER_MODE=official 显式开启） ============
_OFFICIAL_PASSTHROUGH_FIELDS = (
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "response_format",
    "stop",
)


async def _official_chat_response(body, model, stream, tools, request):
    """[v77 P0-7] 用官方 API 服务 /v1/chat/completions。

    同一鉴权（调用方已查）、同一种用量记录（surface='official'、掩码 key、
    上游报真实 usage）、工具按原生 OpenAI 格式转发。官方 key 永不进日志
    或错误体（official_api 内部脱敏）。"""
    payload = {k: body[k] for k in _OFFICIAL_PASSTHROUGH_FIELDS if k in body}
    payload["model"] = official_api.map_model(model)
    if tools:
        payload["tools"] = tools
    if not stream:
        try:
            result = await official_api.official_chat(payload, stream=False)
        except UpstreamError as e:
            return _api_error_response(e, is_anthropic=False)
        usage = result.get("usage") or {}
        with suppress(Exception):
            await _db(record_usage, get_api_key(request), "official",
                      result.get("model") or payload["model"],
                      usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        return JSONResponse(result)
    scope = get_api_key(request)

    async def _gen():
        total_in = total_out = 0
        try:
            async for line in official_api.official_chat(payload, stream=True):
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    try:
                        chunk = json.loads(line[len("data: "):])
                        usage = chunk.get("usage")
                        if usage:
                            total_in = usage.get("prompt_tokens", total_in)
                            total_out = usage.get("completion_tokens", total_out)
                    except json.JSONDecodeError:
                        pass
                yield line
        except UpstreamError as e:
            with suppress(Exception):
                yield f"data: {json.dumps({'error': {'message': str(e)[:300]}})}\n\n"
            return
        with suppress(Exception):
            await _db(record_usage, scope, "official", payload["model"], total_in, total_out)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # NOTE (Stage 1 minor list): `stop` and `max_tokens` are accepted but
    # IGNORED — the upstream web-session API exposes no stop/length controls
    # and v4.1flash ends its turn on its own. Local stop-trim is a possible
    # follow-up; it is intentionally not silently claimed as supported.
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await _json_body(request)
    messages = body.get("messages", [])
    model = resolve_model(body.get("model"))
    thinking = is_thinking_enabled(body, request)
    search = body.get("search", False)
    stream = body.get("stream", False)
    tools = body.get("tools", None)
    # [v73] JSON 输出模式：仅当客户端显式声明时激活，正常请求零影响
    resp_fmt = body.get("response_format") or {}
    json_mode = isinstance(resp_fmt, dict) and resp_fmt.get("type") in ("json_object", "json_schema")
    return await handle_chat(messages, model, thinking, search, stream, tools, scope=get_api_key(request), json_mode=json_mode)


def _responses_input_to_messages(inputs):
    """Map the Responses API `input` array onto chat messages.

    Stage 1 minor list: `input_image` parts are now mapped onto the
    chat-completions image shape (the vision path already exists upstream);
    previously only input_text/input_file were recognized and images were
    passed through untouched."""
    messages = []
    for item in inputs:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue

        role = item.get("role", "user")
        content = item.get("content", [])
        msg_content = []
        if isinstance(content, str):
            msg_content = content
        else:
            for c in content:
                if c.get("type") == "input_text":
                    msg_content.append({"type": "text", "text": c.get("text")})
                elif c.get("type") == "input_file":
                    msg_content.append({"type": "file", "file_id": c.get("file_id")})
                elif c.get("type") == "input_image":
                    url = c.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if url:
                        msg_content.append({"type": "image_url", "image_url": {"url": url}})
                    elif c.get("file_id"):
                        msg_content.append({"type": "file", "file_id": c.get("file_id")})
                    else:
                        msg_content.append(c)
                else:
                    msg_content.append(c)
        messages.append({"role": role, "content": msg_content})
    return messages


@app.post("/v1/responses")
async def openai_responses(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await _json_body(request)
    model = resolve_model(body.get("model"))
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, dict):
        inputs = [inputs]

    messages = _responses_input_to_messages(inputs)

    thinking = is_thinking_enabled(body, request)
    search = body.get("search", False)
    stream = body.get("stream", False)
    tools = body.get("tools", None)

    result = await handle_chat(messages, model, thinking, search, stream, tools, scope=get_api_key(request))

    if stream:
        return result

    if isinstance(result, dict) and "choices" in result:
        message = result["choices"][0]["message"]
        out_content = []
        if message.get("content"):
            out_content.append({"type": "text", "text": message["content"]})
        if message.get("tool_calls"):
            out_content.extend([{"type": "tool_call", "id": tc["id"], "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in message["tool_calls"]])

        msg_output = {
            "type": "message",
            "role": "assistant",
            "content": out_content
        }
        if message.get("reasoning_content"):
            msg_output["reasoning_content"] = message["reasoning_content"]

        return {
            "id": result["id"],
            "object": "response",
            "model": result["model"],
            "output": [msg_output],
            "usage": result.get("usage", {})
        }
    return result


def convert_anthropic_messages(messages):
    """Translate Anthropic message dicts into OpenAI-style dicts.

    tool_use blocks become assistant tool_calls and tool_result blocks become
    role="tool" messages so that build_prompt()/extract_tool_results() see real
    tool results and the signature cache can match across turns.
    """
    openai_msgs = []
    for m in messages:
        content = m.get("content", "")
        tool_calls = []
        tool_results = []
        if isinstance(content, list):
            parts = []
            image_parts = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif c.get("type") == "image":
                    image_parts.append(c)
                elif c.get("type") == "tool_use":
                    tool_calls.append({
                        "id": c.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                        "type": "function",
                        "function": {
                            "name": c.get("name", ""),
                            "arguments": json.dumps(c.get("input", {})),
                        },
                    })
                elif c.get("type") == "tool_result":
                    res_content = c.get("content", "")
                    if isinstance(res_content, list):
                        for item in res_content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                image_parts.append(item)
                        res_content = " ".join(item.get("text", "") for item in res_content if isinstance(item, dict) and item.get("type") == "text")
                    elif not isinstance(res_content, str):
                        res_content = str(res_content)
                    tool_results.append({"tool_call_id": c.get("tool_use_id", ""), "content": res_content})
            if image_parts:
                content = [{"type": "text", "text": s} for s in parts if s] + image_parts
            else:
                content = "\n".join(p for p in parts if p)
        if m.get("role") == "system":
            if content:
                openai_msgs.append({"role": "system", "content": content})
            continue
        if m.get("role") == "assistant":
            if isinstance(content, str) and (not content.strip() or content.strip() == "(no content)"):
                content = None
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if msg["content"] is None and not tool_calls:
                continue
            openai_msgs.append(msg)
            continue
        for tr in tool_results:
            openai_msgs.append({"role": "tool", "tool_call_id": tr["tool_call_id"], "content": tr["content"]})
        has_content = bool(content) if not isinstance(content, list) else len(content) > 0
        if has_content or not tool_results:
            openai_msgs.append({"role": m.get("role", "user"), "content": content})
    return openai_msgs


@app.post("/v1/messages")
@app.post("/messages")
async def anthropic_messages(request: Request):
    # NOTE (Stage 1 minor list): Anthropic `stop_sequences` is accepted but
    # IGNORED — the upstream web-session API exposes no stop controls.
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    body = await _json_body(request)
    system = body.get("system", "")

    messages = body.get("messages", [])
    model = resolve_model(body.get("model"))

    thinking = is_thinking_enabled(body, request)
    stream = body.get("stream", False)
    tools = body.get("tools", [])

    openai_msgs = []
    if system:
        if isinstance(system, list):
            system_str = " ".join(c.get("text", "") for c in system if isinstance(c, dict) and c.get("type") == "text")
        else:
            system_str = str(system)
        if system_str:
            openai_msgs.append({"role": "system", "content": system_str})

    openai_msgs.extend(convert_anthropic_messages(messages))

    openai_tools = []
    for t in tools:
        if t.get("type") == "function":
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", "NO DESCRIPTION"),
                    "parameters": t.get("input_schema", {}),
                },
            })
        elif "name" in t:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })

    output_config = body.get("output_config")
    if isinstance(output_config, dict) and output_config.get("format", {}).get("type") == "json_schema":
        json_schema = output_config["format"].get("schema")
        if json_schema:
            openai_msgs.insert(0, {"role": "system", "content": f"You MUST return valid JSON adhering strictly to this JSON Schema:\n{json.dumps(json_schema)}"})

    req_model = body.get("model")
    if stream:
        return await handle_chat(openai_msgs, model, thinking, False, True, openai_tools or None, is_anthropic=True, req_model=req_model, scope=get_api_key(request))

    result = await handle_chat(openai_msgs, model, thinking, False, False, openai_tools or None, is_anthropic=True, req_model=req_model, scope=get_api_key(request))
    if not isinstance(result, dict) or "choices" not in result:
        return result
    return format_anthropic_response(result, req_model)


@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)

    # Declared limits reflect OBSERVED DeepSeek web behavior (issue #22), not
    # guaranteed upstream API limits: a single first message passes up to ~1M
    # tokens, remembered in-session context reaches ~393K input tokens before
    # the bridge summarizes and rolls the conversation into a fresh chat, and
    # observed per-response output is ~4,000-8,192 tokens.
    base_models = [
        {
            "id": SINGLE_MODEL,
            "object": "model",
            "type": "model",
            "name": SINGLE_MODEL,
            "display_name": "DeepSeek V4.1 Flash",
            "created": 1785456000,
            "created_at": "2026-07-31T00:00:00Z",
            "owned_by": "deeperseeker",
            "context_window": context_window_tokens(),
            "max_output_tokens": max_output_tokens(),
            "capabilities": {
                "batch": {"supported": True},
                "code_execution": {"supported": True},
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": True},
                        "adaptive": {"supported": True}
                    }
                },
                "effort": {
                    "supported": True,
                    "low": {"supported": True},
                    "medium": {"supported": True}
                },
                "context_management": {
                    "clear_thinking_20251015": {"supported": True},
                    "compact_20260112": {"supported": True},
                    "supported": True
                }
            }
        }
    ]

    # Single model, plus the anthropic/claude-* alias for Claude Desktop
    # auto-discovery. Both IDs serve the same upstream v4.1flash model.
    claude_aliases = []
    for m in base_models:
        alias = dict(m)
        alias["id"] = f"anthropic/claude-{m['id']}"
        alias["name"] = f"anthropic/claude-{m['name']}"
        alias["display_name"] = f"Claude {m['display_name']}"
        claude_aliases.append(alias)

    all_models = base_models + claude_aliases

    return {
        "object": "list",
        "data": all_models,
        "has_more": False,
        "first_id": all_models[0]["id"],
        "last_id": all_models[-1]["id"]
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return i18n.page(request, "login.html", {"error": None})


def _prune_admin_sessions():
    """B12: expired dashboard sessions used to stay in memory until restart.
    Called opportunistically on login so the dict tracks live sessions only."""
    now = time.time()
    expired = [sid for sid, ts in SESSIONS.items() if now - ts > SESSION_TTL]
    for sid in expired:
        SESSIONS.pop(sid, None)


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    _prune_admin_sessions()
    if time.time() < _login_fails["locked_until"]:
        return i18n.page(request, "login.html", {"error": i18n.msg("too_many_attempts")})
    if secrets.compare_digest(username.encode("utf-8"), ADMIN_USER.encode("utf-8")) and secrets.compare_digest(password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")):
        _login_fails["count"] = 0
        sid = str(uuid.uuid4())
        SESSIONS[sid] = time.time()
        resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")
        resp.set_cookie("session_id", sid, httponly=True, samesite="lax")
        return resp
    _login_fails["count"] += 1
    if _login_fails["count"] >= 5:
        _login_fails["locked_until"] = time.time() + 300
        _login_fails["count"] = 0
    return i18n.page(request, "login.html", {"error": i18n.msg("invalid_credentials")})



@app.get("/set-lang/{lang}")
async def set_lang(lang: str, request: Request):
    """Pre-login language switcher: remember the choice in a cookie and return."""
    # Scheme-relative and cross-origin targets are refused (open-redirect guard).
    nxt = request.query_params.get("back") or request.headers.get("referer") or "/"
    if not nxt.startswith("/") or nxt.startswith("//") or "\\" in nxt:
        nxt = "/"
    if lang in i18n.SUPPORTED_LANGS:
        resp = RedirectResponse(nxt, status_code=303)
        resp.set_cookie(i18n.COOKIE_NAME, lang, httponly=True, samesite="lax",
                        max_age=365 * 24 * 3600)
        return resp
    resp = RedirectResponse(nxt, status_code=303)
    return resp

@app.get("/logout")
async def logout(request: Request):
    sid = request.cookies.get("session_id")
    SESSIONS.pop(sid, None)
    resp = HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    resp.delete_cookie("session_id")
    return resp


@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    tokens = await _db(get_tokens)
    return i18n.page(request, "dashboard.html", {"tokens": tokens})


@app.post("/v1/pool/restore")
async def v1_pool_restore(request: Request):
    """[v77 P0-3] 恢复隔离账号（管理员手动操作，隔离的唯一出口）。"""
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    try:
        body = await request.json()
        token_id = int(body.get("token_id"))
    except Exception:
        return JSONResponse({"error": "token_id required"}, status_code=400)
    await _db(restore_token, token_id)
    return JSONResponse({"ok": True, "token_id": token_id})


@app.get("/pool")
async def pool_dashboard(request: Request):
    """[v77 P0-3/P0-6 移植] 池状态 + 用量统计可视化页（管理员）。"""
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    return HTMLResponse("""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>账号池与用量</title>
<style>
body{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}
h1{font-size:20px} h2{font-size:16px;margin:24px 0 8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #334155;padding:6px 10px;text-align:left}
th{background:#1e293b;color:#94a3b8}
.ok{color:#4ade80}.warn{color:#fbbf24}.err{color:#f87171}
button{background:#334155;color:#e2e8f0;border:0;border-radius:4px;padding:3px 10px;cursor:pointer}
button:hover{background:#475569}
.meta{color:#64748b;font-size:12px;margin-top:16px}
</style></head><body>
<h1>账号池状态 <span class="meta" id="ts"></span></h1>
<table id="pool"><thead><tr><th>ID</th><th>别名</th><th>状态</th><th>说明</th><th>进行中</th><th>上次使用</th><th>冷却</th><th>操作</th></tr></thead><tbody></tbody></table>
<h2>用量统计（按天 × 按 key）</h2>
<table id="usage"><thead><tr><th>日期</th><th>Key</th><th>通道</th><th>请求数</th><th>输入 tokens</th><th>输出 tokens</th></tr></thead><tbody></tbody></table>
<div class="meta">每 10 秒自动刷新 · v77</div>
<script>
async function refresh(){
  const [pool, usage] = await Promise.all([
    fetch('/v1/pool').then(r=>r.json()).catch(()=>null),
    fetch('/v1/usage').then(r=>r.json()).catch(()=>null),
  ]);
  if (pool && pool.pool){
    const tb = document.querySelector('#pool tbody'); tb.innerHTML='';
    for (const t of pool.pool){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${t.id}</td><td>${t.label}</td>`
        + `<td class="${t.state_class}">${t.status}</td><td>${t.state_note}</td>`
        + `<td>${t.in_flight}</td><td>${t.last_used}</td><td>${t.cooldown}</td>`
        + `<td>${t.can_restore ? `<button onclick="restore(${t.id})">恢复</button>` : ''}</td>`;
      tb.appendChild(tr);
    }
  }
  if (usage && usage.usage){
    const tb = document.querySelector('#usage tbody'); tb.innerHTML='';
    for (const u of usage.usage){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${u.day}</td><td>${u.key}</td><td>${u.surfaces}</td>`
        + `<td>${u.requests}</td><td>${u.input_tokens}</td><td>${u.output_tokens}</td>`;
      tb.appendChild(tr);
    }
  }
  document.getElementById('ts').textContent = new Date().toLocaleTimeString();
}
async function restore(id){
  await fetch('/v1/pool/restore', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({token_id:id})});
  refresh();
}
refresh(); setInterval(refresh, 10000);
</script></body></html>""")


@app.get("/v1/usage")
async def v1_usage(request: Request):
    """[v77 P0-6] 用量统计 JSON（管理员）。"""
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    since = float(request.query_params.get("since", "0") or 0)
    return JSONResponse({"usage": await _db(usage_by_day_key, since)})


@app.get("/v1/pool")
async def v1_pool(request: Request):
    """[v77 P0-3] 账号池运维视图 JSON（管理员，无凭据字段）。"""
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    return JSONResponse({"pool": await _db(get_pool_view)})


@app.post("/tokens/add")
async def tokens_add(request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    form = await request.form()
    auth_token = form.get("auth_token", "").strip().strip("'\"")
    alias = form.get("alias", "").strip() or None
    if auth_token:
        await _db(add_token, auth_token, alias)
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.post("/tokens/{token_id}/delete")
async def tokens_delete(token_id: int, request: Request):
    try:
        get_current_admin(request)
    except HTTPException:
        return HTMLResponse("<meta http-equiv='refresh' content='0;url=/login'>")
    await _db(delete_token, token_id)
    return HTMLResponse("<meta http-equiv='refresh' content='0;url=/dashboard'>")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return await dashboard(request)


async def _deep_health_probe():
    """[v77 P0-9 移植] 一次端到端上游往返（含 PoW）：创建空聊天会话。

    成本：每次调用一个空上游会话——这是本桥能跑的最诚实的探针，因为
    会话创建正是 PoW/客户端版本/账号死亡时最先断的那一跳。探针从不发
    消息，也绝不报告 token 材料。状态感知选号：走 pick_token()，跳过
    隔离与冷却中的行（旧式抓第一行的做法会误报降级、还会把上游流量
    送进已知死亡的凭据）。"""
    token_id = await _db(pick_token)
    auth_token = None
    if token_id is not None:
        with suppress(Exception):
            for t in await _db(get_tokens):
                if t["id"] == token_id:
                    auth_token = t.get("token")
                    break
    if not auth_token:
        return {"upstream_reachable": False, "pow_ok": False, "latency_ms": None,
                "reason": "no ACTIVE token in pool"}
    started = time.perf_counter()
    try:
        await create_new_chat(auth_token)
    except Exception as e:
        logger.warning("Deep health probe failed: %s", e)
        return {"upstream_reachable": False, "pow_ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "reason": "upstream error"}
    return {"upstream_reachable": True, "pow_ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "session_created": True}


@app.get("/health")
async def health(request: Request):
    # [v77 P0-9 移植] ?deep=1 需要鉴权，且 401 先于任何上游流量——
    # 浅层检查保持公开，探针永远不能被当成免费中转测试器。
    deep = (request.query_params.get("deep") or "").strip().lower() in ("1", "true", "yes")
    if deep and not check_key(request):
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    active = sum(1 for t in await _db(get_tokens) if t["status"] == "ACTIVE")
    cookies_valid = False
    try:
        with open(cookie_file_path()) as f:
            c = json.load(f)
        exp = c.get("expiry")
        cookies_valid = bool(exp and exp > time.time())
    except Exception:
        cookies_valid = False
    # WAF cookies are deprecated in favor of Android headers (which do not require cookies).
    # Token presence determines service readiness; cookie status is kept for backup visibility.
    ok = active > 0
    data = {"status": "ok" if ok else "degraded"}
    if check_key(request):
        data["active_tokens"] = active
        data["cookies_valid"] = cookies_valid
        data["version"] = "v77"
        # [v77 P0-9 移植] 深度探针：真实端到端上游往返
        if deep:
            probe = await _deep_health_probe()
            data["deep"] = probe
            if probe.get("upstream_reachable") is not True:
                data["status"] = "degraded"
                ok = False
    return JSONResponse(data, status_code=200 if ok else 503)


if __name__ == "__main__":
    # Pterodactyl-style panels (Wispbyte, HeavenCloud, ...) inject SERVER_PORT
    # instead of PORT, so fall back to it before the built-in default.
    _port = os.getenv("PORT") or os.getenv("SERVER_PORT") or "4000"
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(_port))
