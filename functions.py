import asyncio
import contextvars
import base64
import json
import logging
import mimetypes
import os
import random
import re
import threading
import sqlite3
import string
import time
from datetime import datetime, timezone

import aiohttp
import deepseek_tokenizer
import wasmtime
try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

logger = logging.getLogger("deeperseeker.functions")

# B12 (Stage 1 audit): all relative runtime paths are anchored to the package
# directory — the old global os.chdir(BASE_DIR) at import mutated process-wide
# state (hostile to embedding, testing and PyPI packaging).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

wasm_path = os.path.join(BASE_DIR, "wasm", "deepseek_pow_solver.wasm")
_session = None
_db = os.getenv("DB_PATH") or os.path.join(BASE_DIR, "deeperseeker.db")


def data_dir():
    """Directory holding the SQLite database (and runtime files stored next to
    it, e.g. the generated API key). Honors DB_PATH; anchored to the package
    directory by default so nothing depends on the process CWD."""
    return os.path.dirname(os.path.abspath(_db))


def cookie_file_path():
    """Resolve where the DeepSeek cookie file lives.

    Order: DEEPSEEKER_COOKIE_PATH env > the target of a legacy Docker symlink
    > next to the real DB file (which honors DB_PATH) > the package directory.
    Writing goes to the RESOLVED path so os.replace() can never destroy a
    symlink that bridges the file into the persistent data volume.
    """
    p = os.getenv("DEEPSEEKER_COOKIE_PATH")
    if p:
        return p
    p = os.path.join(BASE_DIR, "aws_cookies_deepseek.json")
    try:
        if os.path.islink(p):
            target = os.path.realpath(p)
            if target:
                return target
    except Exception:
        pass
    d = os.path.dirname(os.path.abspath(_db))
    if d and os.path.abspath(d) != os.path.abspath(os.getcwd()):
        return os.path.join(d, "aws_cookies_deepseek.json")
    return p

try:
    _TZ_OFFSET = str(int(datetime.now().astimezone().utcoffset().total_seconds()))
except Exception:
    _TZ_OFFSET = "19800"


_DB_TLS = threading.local()


class _PooledConn:
    """SQLite 连接句柄（线程内复用）。

    get_db() 原先每次调用都 sqlite3.connect() 再跑一次 PRAGMA journal_mode，
    慢 CPU 上单次约 15ms，而一个请求会调 6~10 次。现在连接按线程复用，
    close() 只归还句柄不真正关闭，对调用方语义不变。
    """
    __slots__ = ("_conn",)

    def __init__(self, conn):
        self._conn = conn

    def close(self):
        try:
            if self._conn.in_transaction:
                self._conn.rollback()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_conn"), name)


def get_db():
    conn = getattr(_DB_TLS, "conn", None)
    # [v57 新修] 连接按 (_db 路径, 线程) 双键缓存。v46 原版只按线程缓存，
    # _db 一旦被换过（测试夹具 functions._db = tmp/xxx.db、未来的多库/
    # 迁移场景），池子会把旧库的句柄当新库的返回——init_db 也在旧库上
    # 建表，新路径永远空，后续读写全部落错库，且没有任何报错。
    # 生产环境 _db 启动后不变，多一次字符串比较没有可测开销。
    if conn is None or getattr(_DB_TLS, "db_key", None) != _db:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        conn = sqlite3.connect(_db, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        _DB_TLS.conn = conn
        _DB_TLS.db_key = _db
    return _PooledConn(conn)


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT,
            token TEXT,
            status TEXT DEFAULT 'ACTIVE'
        );
        CREATE TABLE IF NOT EXISTS sessions (
            signature TEXT PRIMARY KEY,
            token_id INTEGER,
            deepseek_session_id TEXT,
            parent_message_id INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS session_map (
            old_session TEXT PRIMARY KEY,
            new_session TEXT,
            token_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            token_id INTEGER,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS chat_memory (
            session_id TEXT PRIMARY KEY,
            baseline INTEGER DEFAULT 0,
            created_at REAL
        );
    """)
    # B3 (Stage 1 audit) migration: pool cooldown + usage tracking. Older
    # databases miss the columns; the ALTERs are idempotent behind the
    # pragma check.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(tokens)").fetchall()}
    if "rate_limited_until" not in existing_cols:
        conn.execute("ALTER TABLE tokens ADD COLUMN rate_limited_until REAL")
    if "last_used" not in existing_cols:
        conn.execute("ALTER TABLE tokens ADD COLUMN last_used REAL")
    # [v77 P0-3 移植] 连续 403 计数：中间无成功轮的 403 次数。
    # mark_active() 与 restore_token() 归零。
    # [v77.3] rollover 风暴修复：记录"建会话时的客户端历史总量"，
    # 增长参照点从 baseline（首条实际 token）改为它——rollover 摘要会话的
    # baseline 很小但代表全部旧历史，用 baseline 当参照会把没进会话的
    # 旧历史全算成"增长"，导致每轮都触发 rollover（生产实证 2026-10-27
    # 13:32-14:08：17 次连环 rollover，每 1.5-2 分钟一次）。
    existing_cm_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(chat_memory)").fetchall()
    }
    if "creation_total" not in existing_cm_cols:
        conn.execute(
            "ALTER TABLE chat_memory ADD COLUMN creation_total REAL NOT NULL DEFAULT 0"
        )
    # [v77.5/B] 接力摘要：rolling_summary=累计续接笔记；rolling_marker=
    # 已消化到的客户端历史 token 位置（与 digest 切片同一估算口径）。
    if "rolling_summary" not in existing_cm_cols:
        conn.execute(
            "ALTER TABLE chat_memory ADD COLUMN rolling_summary TEXT NOT NULL DEFAULT ''"
        )
    if "rolling_marker" not in existing_cm_cols:
        conn.execute(
            "ALTER TABLE chat_memory ADD COLUMN rolling_marker REAL NOT NULL DEFAULT 0"
        )
    if "consecutive_403" not in existing_cols:
        conn.execute(
            "ALTER TABLE tokens ADD COLUMN consecutive_403 INTEGER NOT NULL DEFAULT 0"
        )
    # [v77 P0-6 移植] usage 统计：每个"客户端可见成功"的完成轮一行，
    # 读取时按 天×掩码key 聚合。保留原始行而非预聚合计数器——
    # 失败的尝试没有行可记，"重试只计一次"成为结构性规则。
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            day TEXT NOT NULL,
            key_alias TEXT NOT NULL,
            surface TEXT NOT NULL,
            model TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_day_key ON usage(day, key_alias)"
    )
    existing_session_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "last_used" not in existing_session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN last_used REAL")
    conn.commit()
    conn.close()
    try:
        prune_sessions()
    except Exception:
        logger.exception("Startup session pruning failed (non-fatal)")


_session_lock = asyncio.Lock()


async def get_session():
    global _session
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                logger.info("Opening shared aiohttp ClientSession")
                # 长保活连接池：默认 keepalive 15s 太短——用户打字/思考的间隔
                # 普遍超过它，导致每轮都重新 TLS 握手（1-2s）。10 分钟保活 +
                # DNS 缓存，把连续对话稳定在已热连接上（内存代价 ~几 MB socket）。
                connector = aiohttp.TCPConnector(
                    limit=64,
                    keepalive_timeout=600,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )
                _session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=None, connect=15, sock_read=600),
                )
    return _session


async def close_session():
    """Close the shared aiohttp ClientSession (Stage 1 minor list).

    Called from app lifespan shutdown: without it, every uvicorn reload/exit
    leaked the session's connector sockets and logged 'Unclosed client
    session' warnings."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


# ==============================================================================
# Stage 0.4 — Dual-endpoint failover
#
# Every upstream POST goes through post_with_failover(): if the primary host
# fails at the connection level (ClientError / timeout) or answers HTTP 5xx,
# the identical request is replayed against the fallback endpoint. 4xx
# responses are returned as-is — a bad token or permission error will not
# improve on a different endpoint, so we fail fast instead of retrying.
#
# Endpoints (env, both optional to override):
#   DEEPSEEKER_UPSTREAM_BASE     primary   (default https://chat.deepseek.com)
#   DEEPSEEKER_UPSTREAM_FALLBACK fallback  (default: none; single-endpoint mode)
# ==============================================================================

def _build_upstream_bases():
    primary = (os.getenv("DEEPSEEKER_UPSTREAM_BASE") or "https://chat.deepseek.com").strip().rstrip("/")
    fallback = (os.getenv("DEEPSEEKER_UPSTREAM_FALLBACK") or "").strip().rstrip("/")
    return [primary] + ([fallback] if fallback else [])


UPSTREAM_BASES = _build_upstream_bases()


async def post_with_failover(path, *, headers, session=None, header_timeout=None, **kwargs):
    """POST to an upstream endpoint with automatic failover across
    UPSTREAM_BASES. This is a plain coroutine that RETURNS an owned response —
    it is not an async context manager:

        resp = await post_with_failover(...)
        async with resp:
            ...

    The caller owns and closes the returned response (async with). Raises the
    last connection error if every endpoint is unreachable, or returns the
    final 4xx/5xx response when the last endpoint answers with one (callers
    keep their existing error paths). `session` is injectable for tests;
    defaults to the shared ClientSession.

    [v63] `header_timeout`（秒，默认 None=不启用）给「等响应头」阶段加独立
    预算：长流式请求的 body 读取有调用方自己的 stall 防护，但 POST 发出后
    上游若"接了请求却不回响应头"，旧行为只能干等 session 级 sock_read
    （线上 600s——2026-10-24 实测造成 12 分钟无响应事故）。传入该参数时
    每个端点的 POST 用 asyncio.wait_for 包裹，到点取消并经既有 except 分支
    走 failover；不传则行为与旧版逐字节一致。"""
    if session is None:
        session = await get_session()
    last_exc = None
    for attempt, base in enumerate(UPSTREAM_BASES):
        is_last = attempt == len(UPSTREAM_BASES) - 1
        try:
            if header_timeout is not None:
                # [v63] 响应头阶段独立预算（S8）：wait_for 到点即取消 POST，
                # asyncio.TimeoutError 落入既有 except 分支触发 failover。
                # 实测：取消后连接被正确关闭移除，连接池无累积；正常路径
                # POST 毫秒返回，wait_for 零开销（0.000s）。
                resp = await asyncio.wait_for(
                    session.post(base + path, headers=headers, **kwargs),
                    timeout=header_timeout)
            else:
                resp = await session.post(base + path, headers=headers, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Upstream POST %s failed on %s: %s", path, base, e)
            last_exc = e
            if is_last:
                raise
            continue
        if resp.status >= 500 and not is_last:
            try:
                body = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                body = ""
            # PR #26 review fix (Medium): hand the failed connection back to
            # the pool. Without this, repeated failovers leak responses and
            # eventually exhaust the connector.
            resp.release()
            logger.warning(
                "Upstream POST %s -> HTTP %d on %s; failing over to fallback endpoint",
                path, resp.status, base,
            )
            last_exc = UpstreamError(resp.status, body[:200])
            continue
        if attempt > 0:
            logger.info("Upstream POST %s succeeded on fallback endpoint %s", path, base)
        return resp
    raise last_exc if last_exc is not None else RuntimeError("post_with_failover: no endpoints configured")


# B11 (Stage 1 audit): the Android client identity used to be hardcoded to one
# version for ALL tokens — when DeepSeek's app moves, a single stale string
# degrades every account simultaneously, and identical fingerprints correlate
# bans. The version is env-configurable now; per-token identity profiles
# (rotating UA/version/locale) remain a P1.
DEEPSEEKER_CLIENT_VERSION = os.getenv("DEEPSEEKER_CLIENT_VERSION", "2.4.5")


def get_headers(auth_token, pow=None):
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
        "user-agent": "Dalvik/2.1.0 (Linux; U; Android 14; Pixel 7)",
        "x-client-platform": "android",
        "x-client-version": DEEPSEEKER_CLIENT_VERSION,
        "x-client-locale": "en_US",
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-timezone-offset": _TZ_OFFSET,
    }
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    if pow:
        headers["x-ds-pow-response"] = pow
    return headers


# ==============================================================================
# BACKUP WAF COOKIE GENERATION (DEPRECATED IN FAVOR OF ANDROID CLIENT HEADERS)
#
# DeepSeek's backend does not enforce AWS WAF on requests using Android client
# headers. The Playwright/Chromium cookie generation below is retained as a backup
# so that if DeepSeek ever tightens WAF rules in the future, it can easily be
# re-enabled simply by uncommenting cookies=cookie in API calls.
# ==============================================================================

_cookie_lock = asyncio.Lock()

COOKIE_REGEN_ATTEMPTS = int(os.getenv("DEEPSEEKER_COOKIE_ATTEMPTS", "2"))
COOKIE_REGEN_TIMEOUT = float(os.getenv("DEEPSEEKER_COOKIE_TIMEOUT", "120"))
COOKIE_FAIL_COOLDOWN = float(os.getenv("DEEPSEEKER_COOKIE_COOLDOWN", "20"))

_cookie_fail = {"until": 0.0, "error": ""}


class CookieGenerationError(Exception):
    """Raised when the DeepSeek WAF cookie file cannot be produced."""


class UpstreamError(Exception):
    """Upstream DeepSeek failure carrying the HTTP status and body (B6).

    Replaces the old Exception(f"HTTP {status}: ...") string protocol whose
    'HTTP (\\d{3}):' regex parsing in app.py broke token rotation and error
    mapping whenever a wording changed. Callers read .status directly; str()
    keeps the familiar 'HTTP <status>: <body>' shape for logs."""

    def __init__(self, status, body=""):
        self.status = int(status)
        self.body = str(body)
        super().__init__(f"HTTP {self.status}: {self.body}")


def _read_cookie_file():
    """Return valid (unexpired) cookies, or None."""
    try:
        with open(cookie_file_path()) as f:
            c = json.load(f)
        if c.get("expiry") is not None and c["expiry"] > time.time():
            return c["cookie"]
    except Exception:
        pass
    return None


def _read_stale_cookie_file():
    """Return cookies even if expired (last-resort fallback)."""
    try:
        with open(cookie_file_path()) as f:
            c = json.load(f)
        return c.get("cookie") or None
    except Exception:
        return None


async def get_cookies():
    cookies = _read_cookie_file()
    if cookies:
        return cookies

    def _cooldown_error():
        return CookieGenerationError(
            _cookie_fail["error"] + " (cooling down; will retry automatically — try again shortly)"
        )

    if time.time() < _cookie_fail["until"]:
        stale = _read_stale_cookie_file()
        if stale:
            return stale
        raise _cooldown_error()
    async with _cookie_lock:
        cookies = _read_cookie_file()
        if cookies:
            return cookies
        if time.time() < _cookie_fail["until"]:
            stale = _read_stale_cookie_file()
            if stale:
                return stale
            raise _cooldown_error()
        last_err = None
        for attempt in range(1, COOKIE_REGEN_ATTEMPTS + 1):
            try:
                logger.info("Generating DeepSeek cookies (attempt %d/%d)...", attempt, COOKIE_REGEN_ATTEMPTS)
                await asyncio.wait_for(_generate_cookies(), timeout=COOKIE_REGEN_TIMEOUT)
                cookies = _read_cookie_file()
                if cookies:
                    _cookie_fail["until"] = 0.0
                    _cookie_fail["error"] = ""
                    return cookies
                last_err = CookieGenerationError("cookie file missing/invalid after generation")
            except Exception as e:
                last_err = e
                logger.warning("DeepSeek cookie generation attempt %d/%d failed: %s", attempt, COOKIE_REGEN_ATTEMPTS, e)
            if attempt < COOKIE_REGEN_ATTEMPTS:
                await asyncio.sleep(min(5 * attempt, 10))
        _cookie_fail["until"] = time.time() + COOKIE_FAIL_COOLDOWN
        _cookie_fail["error"] = f"Could not generate DeepSeek cookies: {last_err}"
        logger.error("%s", _cookie_fail["error"])
        stale = _read_stale_cookie_file()
        if stale:
            logger.warning("Serving STALE DeepSeek cookies after generation failure (upstream may reject them)")
            return stale
        raise CookieGenerationError(_cookie_fail["error"])


async def _generate_cookies():
    if async_playwright is None:
        raise CookieGenerationError("playwright is not installed")
    launch_kwargs = {"headless": False}
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        launch_kwargs["args"] = ["--no-sandbox"]
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_kwargs)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto("https://chat.deepseek.com/", wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_selector("body", timeout=30000)
            try:
                await page.wait_for_url("**/sign_in*", timeout=30000)
            except Exception:
                pass
            cookies = await context.cookies()
        finally:
            await browser.close()
    final_cookies = {}
    expiry = None
    for i in cookies:
        if i.get("name") == "aws-waf-token":
            expiry = i.get("expires")
        final_cookies[i["name"]] = i["value"]
    final_cookies["ds_cookie_preference"] = "%257B%2522level%2522%253A%2522all%2522%257D"
    if not expiry or expiry < 0:
        expiry = time.time() + 1800
    target = cookie_file_path()
    target_dir = os.path.dirname(os.path.abspath(target))
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = os.path.join(target_dir, os.path.basename(target) + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(json.dumps({"cookie": final_cookies, "expiry": expiry}))
    os.replace(tmp_path, target)
    logger.info("DeepSeek cookies saved to %s (expires %s)", target, datetime.fromtimestamp(expiry) if expiry else "n/a")


def get_auth_token():
    conn = get_db()
    row = conn.execute("SELECT token FROM tokens LIMIT 1").fetchone()
    conn.close()
    if row:
        return row[0]
    return None


def add_token(token, alias=None):
    conn = get_db()
    if not conn.execute("SELECT 1 FROM tokens WHERE id = 1").fetchone():
        next_id = 1
    else:
        row = conn.execute("""
            SELECT min(t1.id + 1)
            FROM tokens t1
            LEFT JOIN tokens t2 ON t1.id + 1 = t2.id
            WHERE t2.id IS NULL
        """).fetchone()
        next_id = row[0] if row and row[0] else 1
    conn.execute("INSERT INTO tokens (id, alias, token, status) VALUES (?, ?, ?, 'ACTIVE')", (next_id, alias, token))
    conn.commit()
    conn.close()


def get_tokens():
    conn = get_db()
    rows = conn.execute("SELECT id, alias, token, status FROM tokens").fetchall()
    conn.close()
    return [{"id": r[0], "alias": r[1], "token": r[2], "status": r[3]} for r in rows]


def get_token(token_id):
    conn = get_db()
    row = conn.execute("SELECT id, alias, token, status FROM tokens WHERE id = ?", (token_id,)).fetchone()
    conn.close()
    if row:
        return {"id": row[0], "alias": row[1], "token": row[2], "status": row[3]}
    return None


def delete_token(token_id):
    conn = get_db()
    conn.execute("DELETE FROM tokens WHERE id = ?", (token_id,))
    conn.commit()
    conn.close()


# ==============================================================================
# B3 (Stage 1 audit) — token pool scheduler v2
#
# The old pool selected ACTIVE tokens with ORDER BY RANDOM() and, when every
# token was RATE_LIMITED, fell back to the FIRST token by id. Consequences
# (root cause of issue #32): a limited token never recovered while any other
# token stayed ACTIVE, a single-token pool had no backoff at all, and
# concurrent requests stampeded onto one account — uneven wear and a
# ban-ladder accelerator. The scheduler now provides:
#   * a rate_limited_until cooldown with automatic RATE_LIMITED -> ACTIVE
#     recovery once the cooldown expires (flipped back on the next pick),
#   * least-in-flight selection (spreads load instead of RANDOM()),
#   * an idle-oldest tie-break (the never/least-recently-used token leads),
#   * a soft per-token concurrency cap: capped tokens are only used when
#     nothing else is free,
#   * a soonest-to-recover fallback when every token is still cooling down.
# ==============================================================================

# [v77 P0-3 移植] 连续 403 达到该次数（中间无成功轮）即视为封号阶梯，
# 账号退出轮换，直到管理员手动恢复。0 = 关闭升级（维持只冷却的旧行为）；
# 401 无条件隔离。
QUARANTINE_AFTER_403 = max(0, int(os.getenv("DEEPSEEKER_QUARANTINE_AFTER_403", "3")))

TOKEN_COOLDOWN_SECONDS = max(1, int(os.getenv("DEEPSEEKER_RATE_LIMIT_COOLDOWN", "60")))
TOKEN_CONCURRENCY_CAP = max(1, int(os.getenv("DEEPSEEKER_TOKEN_CONCURRENCY", "8")))

# Per-token in-flight refcounts (per process; single event-loop worker). Every
# send reserves one slot via acquire_token_slot() and releases it when the
# request ends — success, upstream failure or client abort — mirroring the
# ownership-transfer pattern of the per-chat locks.
_in_flight = {}


def _bump_in_flight(token_id, delta):
    n = _in_flight.get(token_id, 0) + delta
    if n > 0:
        _in_flight[token_id] = n
    else:
        _in_flight.pop(token_id, None)


class _TokenSlot:
    """One in-flight reservation against a token. release() is once-only, so
    transferring the slot to a streaming generator (exactly like the per-chat
    lock owner) is safe by construction."""

    __slots__ = ("token_id", "_held")

    def __init__(self, token_id):
        self.token_id = token_id
        self._held = True
        _bump_in_flight(token_id, +1)

    def release(self):
        if not self._held:
            return
        self._held = False
        _bump_in_flight(self.token_id, -1)


def acquire_token_slot(token_id):
    """Reserve one in-flight slot for token_id; pair it with slot.release()."""
    return _TokenSlot(token_id)


def token_in_flight(token_id):
    """Current in-flight request count for a token (selection signal)."""
    return _in_flight.get(token_id, 0)


def pick_token(exclude=None):
    """Select the best available token (scheduler v2 — block comment above).

    `exclude` (optional, B10) lists token ids the caller just saw fail: they
    are avoided while ANY other token is available, so an empty-SSE retry
    cannot land on the same poisoned token.
    """
    now = time.time()
    conn = get_db()
    rows = conn.execute(
        "SELECT id, status, rate_limited_until, last_used FROM tokens"
    ).fetchall()
    conn.close()
    if not rows:
        return None

    excluded = set(exclude or ())
    available = []
    recovered = []
    for tid, status, until, last_used in rows:
        if status == "QUARANTINED":
            continue  # [v77 P0-3] 终态：仅 restore_token() 可恢复
        if status == "RATE_LIMITED" and until is not None and until > now:
            continue  # still cooling down
        if status == "RATE_LIMITED":
            recovered.append(tid)  # cooldown expired -> auto-recover below
        available.append((tid, last_used or 0.0))

    pickable = [entry for entry in available if entry[0] not in excluded] or available
    if pickable:
        if recovered:
            conn = get_db()
            conn.executemany(
                "UPDATE tokens SET status = 'ACTIVE', rate_limited_until = NULL WHERE id = ?",
                [(tid,) for tid in recovered],
            )
            conn.commit()
            conn.close()
            logger.info("Token pool: cooldown expired, auto-recovered token(s) %s", recovered)

        def _sort_key(entry):
            tid, last_used = entry
            in_flight = _in_flight.get(tid, 0)
            # Soft cap: over-cap tokens sort behind every under-cap one.
            over_cap = 1 if in_flight >= TOKEN_CONCURRENCY_CAP else 0
            return (over_cap, in_flight, last_used, tid)

        return min(pickable, key=_sort_key)[0]

    # Every token is rate-limited and still cooling: pick the one that
    # recovers first (bounded wait) instead of stampeding onto the lowest id
    # with no backoff at all.
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM tokens WHERE status != 'QUARANTINED' "
        "ORDER BY COALESCE(rate_limited_until, 0) ASC, id ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else None


def mark_quarantined(token_id, reason):
    """[v77 P0-3 移植] 账号退出轮换，直到管理员恢复。

    与 RATE_LIMITED 不同：此状态没有计时器——rate_limited_until 清空，
    pick_token() 的自动恢复永远不会把隔离行误认成冷却完毕的行。
    """
    logger.warning("Token #%d QUARANTINED (manual return required): %s", token_id, reason)
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET status = ?, rate_limited_until = NULL WHERE id = ?",
        ("QUARANTINED", token_id),
    )
    conn.commit()
    conn.close()


def restore_token(token_id):
    """[v77 P0-3 移植] 隔离的唯一出口：管理员显式操作。

    last_used 刻意不动——恢复的账号带着最旧的用量戳，
    idle-oldest 决胜会让它温和回到轮换，而不是被灌一波突发流量。
    """
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET status = ?, rate_limited_until = NULL, "
        "consecutive_403 = 0 WHERE id = ?",
        ("ACTIVE", token_id),
    )
    conn.commit()
    conn.close()
    logger.info("Token #%d restored to ACTIVE by operator", token_id)


def _bump_403_streak(token_id):
    """[v77 P0-3 移植] 递增并返回该账号的连续 403 计数。"""
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET consecutive_403 = consecutive_403 + 1 WHERE id = ?",
        (token_id,),
    )
    row = conn.execute(
        "SELECT consecutive_403 FROM tokens WHERE id = ?", (token_id,)
    ).fetchone()
    conn.commit()
    conn.close()
    return row[0] if row else 0


def mark_limited(token_id, code=None):
    """[v77 P0-3 移植] 向账号记一次上游失败——池的唯一门户。

    不带 code：与既有行为完全一致（RATE_LIMITED 冷却自动恢复）。
    带上游状态码：
      * 401 -> 立即 QUARANTINED。被拒的凭据不是暂态，冷却 60 秒只会
        把同一个死账号重新武装。
      * 403 -> 照旧冷却；连续 403 达到 QUARANTINE_AFTER_403（中间无
        成功轮）则 QUARANTINED。
      * 其他（429）-> 冷却，照旧。
    """
    if code == 401:
        mark_quarantined(token_id, "upstream 401 (auth rejected)")
        return
    if code == 403:
        streak = _bump_403_streak(token_id)
        threshold = QUARANTINE_AFTER_403
        if threshold and streak >= threshold:
            mark_quarantined(
                token_id,
                f"{streak} consecutive upstream 403s (threshold {threshold})",
            )
            return
    until = time.time() + TOKEN_COOLDOWN_SECONDS
    logger.warning(
        "Token #%d marked RATE_LIMITED for %ds (recovers automatically)",
        token_id,
        TOKEN_COOLDOWN_SECONDS,
    )
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET status = ?, rate_limited_until = ? WHERE id = ?",
        ("RATE_LIMITED", until, token_id),
    )
    conn.commit()
    conn.close()


def mark_active(token_id):
    """[v77 P0-3 移植] 记一次成功轮：回到轮换、连续计数清零。

    QUARANTINED 行刻意不动：请求会重叠，隔离前的发送可能在拉黑之后
    才落地成功；放行会污染管理员控制的状态，只有 restore_token()
    能结束隔离。
    """
    conn = get_db()
    conn.execute(
        "UPDATE tokens SET status = ?, rate_limited_until = NULL, last_used = ?, "
        "consecutive_403 = 0 WHERE id = ? AND status != 'QUARANTINED'",
        ("ACTIVE", time.time(), token_id),
    )
    conn.commit()
    conn.close()


def _format_span(seconds):
    """[v77 P0-3 移植] 池视图的紧凑时长：42s / 7m / 3h / 2d。"""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def get_pool_view():
    """[v77 P0-3 移植] 每账号的运维视图。

    状态、负载、计时器以可直接渲染的字符串给出，且不含任何凭据字段：
    token 列刻意缺席——此输出直接进 HTML，浏览器会缓存、截图会带走。
    """
    now = time.time()
    conn = get_db()
    rows = conn.execute(
        "SELECT id, alias, status, rate_limited_until, last_used, consecutive_403 "
        "FROM tokens ORDER BY id"
    ).fetchall()
    conn.close()

    view = []
    for tid, alias, status, until, last_used, streak in rows:
        streak = streak or 0
        if status == "QUARANTINED":
            state_class, cooldown = "err", "—"
            note = "quarantined — manual return required"
        elif status == "RATE_LIMITED" and until and until > now:
            state_class, cooldown = "warn", f"{_format_span(until - now)} left"
            note = "cooling down, recovers automatically"
        elif status == "RATE_LIMITED":
            state_class, cooldown = "warn", "expired"
            note = "rejoins rotation on the next pick"
        else:
            state_class, cooldown = "ok", "—"
            note = "in rotation"
        if streak and status != "QUARANTINED":
            note = f"{note}, 403 streak {streak}"
        age = "never" if not last_used else f"{_format_span(now - last_used)} ago"
        view.append(
            {
                "id": tid,
                "label": alias if alias else f"untagged #{tid}",
                "status": status,
                "state_class": state_class,
                "state_note": note,
                "in_flight": _in_flight.get(tid, 0),
                "last_used": age,
                "cooldown": cooldown,
                "can_restore": status == "QUARANTINED",
            }
        )
    return view


# ============ [v77 P0-6 移植] usage 统计 ============


def mask_api_key(key):
    """掩码 key 别名（展示用）：只留末 4 字符。

    dashboard 与所有聚合都渲染它——页面截图不得携带可用 key 材料。
    超短 key（<= 8 字符）整体掩码。"""
    key = str(key or "").strip()
    if not key:
        return "unknown"
    if len(key) <= 8:
        return "…"
    return "…" + key[-4:]


def record_usage(key_alias, surface, model, in_tokens, out_tokens, ts=None):
    """[v77 P0-6 移植] 记一个完成轮。key_alias 应为已掩码值——
    调用点传 mask_api_key(raw_key)；本函数绝不把裸 key 落库。"""
    alias = (
        mask_api_key(key_alias)
        if key_alias and not str(key_alias).startswith("…")
        else str(key_alias or "unknown")
    )
    when = time.time() if ts is None else float(ts)
    day = time.strftime("%Y-%m-%d", time.gmtime(when))
    conn = get_db()
    conn.execute(
        "INSERT INTO usage (ts, day, key_alias, surface, model, input_tokens, output_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            when,
            day,
            alias,
            str(surface or "unknown"),
            str(model or ""),
            max(0, int(in_tokens or 0)),
            max(0, int(out_tokens or 0)),
        ),
    )
    conn.commit()
    conn.close()


def usage_by_day_key(since_ts=0.0):
    """[v77 P0-6 移植] 按 天×掩码key 聚合用量。

    返回可直接渲染的行。(day, key_alias) 索引覆盖 GROUP BY——
    一个月的数据也只是几毫秒 SQLite，无需预聚合表。"""
    conn = get_db()
    rows = conn.execute(
        "SELECT day, key_alias, "
        "GROUP_CONCAT(DISTINCT surface), COUNT(*), "
        "SUM(input_tokens), SUM(output_tokens) "
        "FROM usage WHERE ts >= ? "
        "GROUP BY day, key_alias ORDER BY day DESC, key_alias ASC",
        (float(since_ts or 0.0),),
    ).fetchall()
    conn.close()

    out = []
    for day, alias, surfaces, requests, in_tok, out_tok in rows:
        in_tok = in_tok or 0
        out_tok = out_tok or 0
        out.append(
            {
                "day": day,
                "key": alias,
                "surfaces": surfaces or "unknown",
                "requests": requests,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        )
    return out


# ============ [v77 PR#67 移植] 上下文超限错误分类器 ============

_CONTEXT_LIMIT_HINT_RE = re.compile(
    r"maximum.?context"
    r"|context.?(?:length|limit|window)"
    r"|token.?limit"
    r"|(?:prompt|payload|conversation|messages?).{0,24}too.?long"
    r"|exceed(?:ed|s)?[^.\n]{0,30}context",
    re.IGNORECASE,
)


def is_context_limit_error(exc):
    """[v77 PR#67 移植] 公开分类器：这次失败是不是上游会话上下文耗尽？

    agent 客户端死在这里：上下文类失败曾被原样换号重试（上下文属于
    CHAT 不属于 TOKEN，每次尝试都同样失败），最后才作为上下文错误
    暴露给客户端。调用方用它触发摘要 rollover。匹配：
      * UpstreamError 响应体带上下文措辞（HTTP 400 'maximum context
        length' / 'context limit' / 'too long' 等）；
    刻意不匹配：泛化 5xx/超时——'deadline exceeded'、'502 upstream
    timeout' 是重试型，不是上下文型。松散词只在锚定上下文材料时
    才算数：裸 'exceed' 或前面没有 prompt/message 的 'too long'
    不再分类为上下文（上游 'header limit exceeded' 曾误触发一次
    浪费的摘要 rollover）。"""
    if isinstance(exc, UpstreamError):
        body = getattr(exc, "body", None) or ""
        return bool(_CONTEXT_LIMIT_HINT_RE.search(str(body)))
    text = str(exc or "")
    return bool(
        re.search(
            r"maximum.?context|context.?(?:length|limit|window)|token.?limit",
            text,
            re.IGNORECASE,
        )
    )


def find_session(sig):
    conn = get_db()
    row = conn.execute("SELECT token_id, deepseek_session_id, parent_message_id FROM sessions WHERE signature = ?", (sig,)).fetchone()
    if row:
        # B13: touch on read so pruning evicts by real recency of USE, not by
        # insertion order — long-running chats get older rows every turn, and
        # they are precisely the sessions pruning must protect.
        conn.execute("UPDATE sessions SET last_used = ? WHERE signature = ?", (time.time(), sig))
        conn.commit()
    conn.close()
    if row:
        return {"token_id": row[0], "session_id": row[1], "parent_message_id": row[2]}
    return None


# Cap on stored session signatures. Every request stores 2 rows and nothing
# ever removed them, so after many chats the SQLite file (and its WAL) grew
# unbounded — on volume-limited deployments a full disk freezes ALL requests,
# including brand-new chats. PRUNE_EVERY saves trigger a prune that keeps the
# MAX_SESSIONS MOST-RECENTLY-USED rows (B13: by last_used, not insertion
# order — the old rowid policy evicted long-running chats, the most valuable
# sessions, exactly because they hold the oldest rows).
MAX_SESSIONS = int(os.getenv("DEEPSEEKER_MAX_SESSIONS", "20000"))
PRUNE_EVERY = int(os.getenv("DEEPSEEKER_PRUNE_EVERY", "500"))
_save_counter = {"n": 0}


def prune_sessions():
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        deleted = 0
        if total > MAX_SESSIONS:
            # B13: evict LEAST-RECENTLY-USED. NULL last_used (never re-hit)
            # sorts as 0, i.e. evicted first; rowid breaks ties deterministically.
            deleted = conn.execute(
                "DELETE FROM sessions WHERE rowid IN "
                "(SELECT rowid FROM sessions ORDER BY COALESCE(last_used, 0) ASC, rowid ASC LIMIT ?)",
                (total - MAX_SESSIONS,),
            ).rowcount
        deleted_map = conn.execute(
            "DELETE FROM session_map WHERE created_at < datetime('now', '-7 days')"
        ).rowcount
        conn.commit()
        if deleted or deleted_map:
            logger.info("Pruned %d session signature(s) and %d stale session_map row(s)", deleted, deleted_map)
    finally:
        conn.close()


def save_session(sig, token_id, session_id, parent_message_id=0):
    conn = get_db()
    conn.execute(
        """INSERT OR REPLACE INTO sessions (signature, token_id, deepseek_session_id, parent_message_id)
           VALUES (?, ?, ?, ?)""",
        (sig, token_id, session_id, parent_message_id),
    )
    conn.commit()
    conn.close()
    _save_counter["n"] += 1
    if _save_counter["n"] >= PRUNE_EVERY:
        _save_counter["n"] = 0
        try:
            prune_sessions()
        except Exception:
            logger.exception("Session pruning failed (non-fatal)")


def get_chat_baseline(session_id):
    """(baseline, creation_total) — 建会话时的两个记账值。

    baseline：本会话首条实际进入上游的 token 数（首条快速通道的负重）。
    creation_total：建会话那一刻客户端全量历史的 token 数——增长参照点。
    Rollover 比较 GROWTH（= 当前全量 - creation_total）而不是重测整个
    只增不减的客户端历史（否则每一轮都触发一次全新 rollover）。
    旧行（无 creation_total 列）返回 creation_total=0，调用方回退用
    baseline 当参照（与 v72-v76 行为一致）。"""
    if not session_id:
        return 0, 0
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT baseline, creation_total FROM chat_memory WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row or row[0] is None:
            return 0, 0
        ct = int(row[1]) if (len(row) > 1 and row[1] is not None) else 0
        return int(row[0]), ct
    finally:
        conn.close()


def set_chat_baseline(session_id, baseline, creation_total=0):
    """Record the conversation size at which this upstream chat was opened.

    baseline：首条实际进入上游的 token 数；creation_total：那一刻客户端
    全量历史的 token 数（增长参照点）。"""
    if not session_id:
        return
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO chat_memory (session_id, baseline, created_at, creation_total) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET baseline = excluded.baseline, "
            "creation_total = excluded.creation_total",
            (session_id, int(baseline or 0), time.time(), int(creation_total or 0)),
        )
        conn.commit()
    finally:
        conn.close()


def get_rolling_state(session_id):
    """(rolling_summary, rolling_marker) — 本会话创建时交接下来的接力摘要。

    rolling_summary：截至上一代换本的累计续接笔记（'' = 无，首次换本）。
    rolling_marker：客户端历史已被摘要消化的 token 位置（摘要切片的左边界）。
    旧行/缺行返回 ("", 0) —— 退化为"全量消化（原文尾巴除外）"，与 v77.4 等价。
    """
    if not session_id:
        return "", 0
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT rolling_summary, rolling_marker FROM chat_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return "", 0
        return (row[0] or ""), float(row[1] or 0)
    finally:
        conn.close()


def set_rolling_state(session_id, summary, marker):
    """记录接力摘要与消化位置（保留行内 baseline/creation_total 记账）。"""
    if not session_id:
        return
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO chat_memory (session_id, baseline, created_at, creation_total,"
            " rolling_summary, rolling_marker) VALUES (?, 0, ?, 0, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET rolling_summary = excluded.rolling_summary,"
            " rolling_marker = excluded.rolling_marker",
            (session_id, time.time(), summary or "", float(marker or 0)),
        )
        conn.commit()
    finally:
        conn.close()


def delete_session(sig):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE signature = ?", (sig,))
    conn.commit()
    conn.close()


def next_parent(parent_message_id):
    """Compute the parent_message_id to use for the next turn on a chat session.

    Invariant: each successful /chat/completion request appends exactly one user
    message and one assistant message to the DeepSeek chat session, so if the
    request was sent with parent_message_id P, the last message id afterwards is
    P + 1 and the next request must use P + 2.

    DeepSeek's web API exposes no endpoint to list a session's messages, so this
    increment cannot be verified against the server; it is centralized here so
    every save_session() call site (token-rotation branch, non-stream and both
    streaming paths) stays consistent if the invariant ever changes.
    """
    return parent_message_id + 2


def delete_sessions_for_chat(token_id, session_id):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE token_id = ? AND deepseek_session_id = ?", (token_id, session_id))
    conn.commit()
    conn.close()


# ==============================================================================
# B4 (Stage 1 audit) — file ownership registry
#
# /v1/files uploads picked a RANDOM token and upstream files are
# account-scoped, so a later /v1/chat/completions referencing that file_id
# could pick a different token and get "file not found" — the OpenAI-style
# upload->reference flow (Claude Code / Cline file flows) was broken by
# design. Uploads are now pinned to their token; chat handlers prefer the
# file-owner token on the first turn and re-home foreign references onto the
# chat's own token on later turns.
# ==============================================================================


def record_file(file_id, token_id):
    """Pin an upstream file_id to the token (account) that owns it.
    First owner wins: INSERT OR IGNORE keeps the original mapping stable."""
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO files (file_id, token_id, created_at) VALUES (?, ?, ?)",
        (file_id, token_id, time.time()),
    )
    conn.commit()
    conn.close()


def get_file_token(file_id):
    """Return the token id that owns an uploaded file_id, or None."""
    conn = get_db()
    row = conn.execute("SELECT token_id FROM files WHERE file_id = ?", (file_id,)).fetchone()
    conn.close()
    return row[0] if row else None


# DeepSeek now serves a single model (v4.1flash) as the website default. The
# web API uses the website default when model_type is null, so no explicit
# value is sent. If DeepSeek ever exposes an explicit model_type enum value
# for v4.1flash, this constant is the single place to set it.
DEFAULT_MODEL_TYPE = None

# Flat peak-hour rates (per 1M tokens) for the single default model,
# per https://api-docs.deepseek.com/quick_start/pricing
DEEPSEEK_TARIFFS = {
    "deepseek-v4.1-flash": {
        "cache_miss_input": 0.44,
        "output_generation": 1.32,
    },
}


def count_tokens(text, model="deepseek-v4.1-flash"):
    return len(deepseek_tokenizer.ds_token.encode(text))


_BAD_ARG_KEY_RE = re.compile(r"""[\s"'<>=]|^(?:string|type|bool|int|float|number|true|false|null)$""", re.IGNORECASE)


class _TruncatedArgs(Exception):
    """[v57 移植 v50] 参数在字符串内部被切断：内容不可信，整条调用必须作废。

    旧行为是补个引号和右括号就放行，于是 3KB 脚本被截断后写出半个文件——
    下游完全不知道数据是坏的。这里改为显式失败，宁可丢一次调用。
    """


def _repair_truncated_json(s, strict=False):
    if not isinstance(s, str):
        return s
    t = s.strip()
    if not t or t[0] not in "{[":
        return s
    try:
        json.loads(t)
        return t
    except Exception:
        pass

    def _scan(u):
        in_str = False
        esc = False
        stack = []
        for ch in u:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
        return in_str, stack

    in_str, stack = _scan(t)
    if in_str:
        # [v57 移植 v50] 字符串在中间被切断：边界不可知，补引号会让半个脚本
        # 通过校验，写出一个「语法合法、内容残缺」的文件——这是最难查的
        # 一类损坏。strict 模式下拒绝修复，返回 None，让调用方明确报错。
        if strict:
            return None
        t += '"'
    in_str, stack = _scan(t)
    t += "".join("}" if c == "{" else "]" for c in reversed(stack))
    try:
        json.loads(t)
        return t
    except Exception:
        return s


def _sanitize_args(obj):
    """[v57 移植 v50] 严格模式。

    字符串入参先走 _repair_truncated_json(strict=True)：中途切断的 JSON
    直接抛 _TruncatedArgs，由 normalize_tool_call 丢弃整条调用——绝不
    降级成 {} 放行（那会变成一条「合法但参数为空」的调用，客户端拒绝时
    报错信息完全指不到真正原因）。键清洗沿用 v41：去掉空白键和模型
    臆造的非参数键（tool_title/type 之类）。
    """
    if isinstance(obj, str):
        obj = _repair_truncated_json(obj, strict=True)
        if obj is None:
            raise _TruncatedArgs("参数在字符串内部被切断")
        try:
            obj = json.loads(obj)
        except Exception:
            return {}
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        k2 = k.strip()
        if not k2 or _BAD_ARG_KEY_RE.search(k2):
            continue
        out[k2] = v
    return out


def _coerce_args_json(args):
    try:
        return json.dumps(_sanitize_args(args), ensure_ascii=False)
    except _TruncatedArgs as e:
        logger.warning("[tool-args] 参数不可信，丢弃整条调用: %s", e)
        raise
    except Exception:
        return json.dumps({})


_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _valid_tool_call(t):
    """[v57 新修] Drop calls the client would refuse anyway.

    v36 原版按扁平 {"name","arguments"} 写，但 normalize_tool_call 返回
    OpenAI 嵌套格式 {"id","type","function":{...}} 且 arguments 是 JSON
    字符串——两个格式对不上，闸门恒判 False，流式工具调用 100% 静默丢弃，
    这个 bug 横跨 15+ 个版本没人发现。本函数是全部归档版本中唯一的
    双格式修复：嵌套/扁平都认，字符串 arguments 先 json.loads。

    [v57 新修 A6] 空参数放行：v36 把 len(args)==0 一律判死，误杀
    list_tabs 这类无参工具的合法调用。代理层没有工具 schema 知识，
    无权判"必填"——空 arguments 放行，由客户端拒绝并给出准确错误，
    好过在代理层静默丢调用让工具根本不执行。

    [v57 新修] 名字正则去掉 {1,63} 上限：v36 加这个上限想防"传输截断
    出的短名"，但真实 MCP 工具名可以远超 64 字符（上游自己的测试用
    5000+ 字符的名字断言必须解析通过），上限只误杀不防骗——截断出的
    长名客户端会以"未知工具"明确拒绝，好过在代理层静默丢弃。截断
    防护由 len(name) < 2 承担（实测残名都是单字符）。
    """
    if not isinstance(t, dict):
        return False
    fn = t.get("function")
    if isinstance(fn, dict):
        name = (fn.get("name") or "").strip()
        args = fn.get("arguments")
    else:
        name = (t.get("name") or "").strip()
        args = t.get("arguments")
    if len(name) < 2 or _TOOL_NAME_RE.match(name) is None:
        return False
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return False
    if args is None:
        return True
    if not isinstance(args, dict):
        return False
    return True


def normalize_tool_call(tool_data_or_name, args_if_name=None):
    if isinstance(tool_data_or_name, str):
        name = tool_data_or_name
        args = args_if_name if args_if_name is not None else {}
    elif isinstance(tool_data_or_name, dict):
        tool_data = tool_data_or_name
        if "function" in tool_data and isinstance(tool_data["function"], dict):
            fn = tool_data["function"]
            name = fn.get("name") or tool_data.get("name")
            args = fn.get("arguments") or fn.get("parameters") or fn.get("input") or fn.get("args") or fn.get("params") or {}
        else:
            name = tool_data.get("name") or tool_data.get("tool") or tool_data.get("tool_name") or tool_data.get("function") or tool_data.get("action")
            args = tool_data.get("arguments") or tool_data.get("parameters") or tool_data.get("input") or tool_data.get("args") or tool_data.get("params") or tool_data.get("tool_input") or tool_data.get("action_input") or {}
    else:
        return None

    if not name or not isinstance(name, str):
        return None
    # [v66] 下划线开头的"工具名"是打坏标记的残片（生产实证 "_call"：模型
    # 或历史垃圾让解析器抠出残名，客户端执行必失败显示 "$ _call 无效命令"）。
    # 在源头拒收——无论哪个投递路径都不会再产出此类调用。
    if name.lstrip().startswith("_"):
        return None
    # [v57 移植 v50] 统一走 _coerce_args_json：字符串 arguments 中途切断时
    # 抛 _TruncatedArgs，整条调用丢弃，不再把截断的 JSON 双引号包成字符串
    # 放行（v41 及之前的 A4 bug）。
    try:
        args_str = _coerce_args_json(args)
    except _TruncatedArgs:
        return None
    call_id = "call_" + "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": args_str,
        },
    }


def clean_json_str(s):
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _code_fence_spans(text):
    """Return the (start, end) spans of markdown fenced code blocks.

    Tool-call markup inside a fence is documentation/example text, not an
    actual tool call, so matches falling inside these spans are ignored.
    Unclosed fences extend to end-of-text.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"```.*?(?:```|$)", text, re.DOTALL)]


# [FIX 6] Bare decorated DSML opener without its leading '<': the model
# occasionally emits "｜｜DSML｜｜ calls>..." (or the half-width/spaced
# variant "| | DSML | | calls>") missing the opening angle bracket. Every
# entry regex in this module requires '<', so these fragments used to leak
# into the chat as prose and the tool never executed (2026-10-20 field
# evidence: "| | DSML | | calls>" between two successful tool rounds).
# Repair: a line that is essentially a bare decorated DSML opener gets its
# '<' prepended and its bars de-spaced/normalized so the entry regex can
# match. Line-anchored on purpose — prose merely *mentioning* DSML is left
# untouched.
_BARE_OPEN_LINE_RE = re.compile(
    r"(?m)^([ \t]*[｜\|][ \t｜\|]*DSML[ \t｜\|]*(?:calls|invoke|tool_calls?|function_calls?|function_call|parameter)\b[^\n]*)$",
    re.IGNORECASE,
)

_BARE_OPEN_HEAD_RE = re.compile(
    r"^([ \t]*[｜\|][ \t｜\|]*DSML[ \t｜\|]*)([A-Za-z_][A-Za-z0-9_]*)([^\n]*)$",
    re.IGNORECASE,
)

# [v57 新修] 第 1 组捕获可选的 ｜｜DSML｜｜ 装饰前缀：带装饰的裸开口不是
# A2 现场问题（那个连装饰都没有），交给 _BARE_OPEN_LINE_RE 补 '<'。v38 原版
# 没有这个组，会把装饰直接剥掉——开标签变成 <｜｜DSML｜｜<tool_call>、闭标签
# 变成 </｜｜DSML｜｜<tool_call>，工具虽然还能解析，但块前残留的
# <｜｜DSML｜｜ 会泄进正文（上游测试 json_body chunk=5 抓到该回归）。
_BARE_PLAIN_OPEN_RE = re.compile(
    r"(?m)(?<!<)([｜\| \t]{0,8}DSML[｜\| \t]{0,8})?(tool_calls?|function_calls?|function_call|invoke|calls) *> *(?=[{])",
    re.IGNORECASE,
)


def _repair_bare_dsml_lines(text):
    def _fix_bare_plain(m):
        if m.group(1):
            # 带装饰：原样返回，交给下面的 _BARE_OPEN_LINE_RE 补 '<'
            return m.group(0)
        return chr(60) + m.group(2) + ">"

    # [v57 新修] 廉价前置过滤：_BARE_PLAIN_OPEN_RE 能命中的必要条件是文本
    # 里出现 tool_call / function_call / invoke / calls 之一（IGNORECASE）。
    # 纯正文块一次子串扫描即可放行，不必付全缓冲正则扫描——50KB buffer 上
    # 3.9ms -> 0.03ms，长回答流式场景由 O(n^2) 累积扫描降为 O(n)。
    # 四个字面量一个都不在时正则必然不匹配，与原 DSML 守卫早退等价，无损。
    _low = text.lower()
    if ("tool_call" in _low or "function_call" in _low
            or "invoke" in _low or "calls" in _low):
        text = _BARE_PLAIN_OPEN_RE.sub(_fix_bare_plain, text)
    if "DSML" not in text:
        return text

    def _fix(m):
        body = m.group(1)
        mm = _BARE_OPEN_HEAD_RE.match(body)
        if not mm:
            return body
        head, tag, rest = mm.groups()
        compact = re.sub(r"[ \t]", "", head).replace("|", "｜")
        return "<" + compact + tag + rest

    return _BARE_OPEN_LINE_RE.sub(_fix, text)


# Residue that ends up INSIDE a parameter value when the model emits a
# stray decorated closer before the structural close (e.g.
# ">检查...内容</| | DSML | |></｜DSML｜>parameter>"). Structural cleanup
# never touches it because it is not at a tag boundary — strip it from
# values directly. Bare decorated markers only; real prose is untouched.
_VALUE_RESIDUE_RE = re.compile(
    r"(?:</)?[｜\| \t]{0,8}DSML[｜\| \t]{0,8}\s*>[^<\n]{0,12}>"  # marker + trailing fragment (e.g. ">参数>")
    r"|(?:</)?[｜\| \t]{0,8}DSML[｜\| \t]{0,8}\s*>"              # marker with '>' but no fragment
    r"|(?:</)?[｜\| \t]{0,8}DSML[｜\| \t]{0,8}[^>\n]{0,40}>"     # [v66] 残缺收尾：标记后粘着 tag 尾巴（生产实证 2026-10-25
                                                                # 会话 d440af4c：模型写 "标题｜｜DSML｜｜ parameter>"（少 '</'），
                                                                # 旧正则所有分支都以 </? 开头——那是"必须有 <"，不是"可选 </"——
                                                                # 无 < 的残片只能命中裸标记分支，尾巴 " parameter>" 漏进
                                                                # tool_title，客户端工具卡片显示 "检查批量脚本状态 parameter>"）
    r"|[｜\|]{2}\s*DSML\s*[｜\|]{2}",                            # [v33] bare marker, no '>' at all
    re.IGNORECASE,
)


def _clean_param_value(val):
    val = re.sub(r"</?[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|invoke|function_call|parameter|param)\b[^>]*>", "", val, flags=re.IGNORECASE)
    # [v33] same whitelist blind spot as _WRAPPER_TAG_RE, but here it is fatal:
    # a decorated closer left inside `command` reaches the client and the shell
    # dies with "unterminated quoted string". Marker wins, name is irrelevant.
    val = _DECOR_TAG_RE.sub("", val)
    val = _BARE_CLOSE_RE.sub("", val)
    val = _VALUE_RESIDUE_RE.sub("", val)
    # [v33] bare DSML marker anywhere inside the value ("echo a ｜｜DSML｜｜ b").
    # This construct is never legitimate content, so strip it unconditionally —
    # unlike the streaming path we need no keyword lookahead here because the
    # value is already finalized.
    # [v32] bare closer missing '</' stuck to the value tail (model writes
    # "标题｜｜DSML｜｜>" instead of "</｜｜DSML｜｜ parameter>") — drop any
    # trailing decoration tail that no legitimate value ends with.
    val = re.sub(r"[｜\| \t]{0,8}DSML[｜\| \t]{0,8}(?:\s*(?:parameter|invoke|calls?|tool_calls?)\b[^>\n]{0,40}>)?\s*$", "", val, flags=re.IGNORECASE).strip()
    return val.strip()


# [v66] 精确开标签：正文里存在 <tool_call>/<function_call> 时，DSML 搜刮器
# 必须让位（见 parse_tools 内注释）。think 区间内的引用不算数。
_EXACT_CALL_OPEN_RE = re.compile(r"<(?:tool_call|function_call)\s*>", re.IGNORECASE)
# [v66] think 块区间：模型的私有推理区，里面引用的任何标记原文都不参与解析。
_THINK_SPAN_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# [v75-a] DSML 参数外层包装名：模型模仿 OpenAI <tool_call> 结构时会把整个
# arguments 包成一个 parameter。见到这些名字且值为对象时合并解包，消除
# {"arguments": {...}} 双层嵌套（生产实证 2026-10-25 会话 e1751fe3）。
_DSML_ARGS_WRAPPERS = {"arguments", "args", "parameters", "params"}


# [v77 PR#70 移植] 当前请求的工具列表，由 handle_chat 设置，供 parse_tools
# 在模型丢掉工具名时按 schema 反推调用归属。
request_tools = contextvars.ContextVar("request_tools", default=None)


def _tool_schemas():
    schemas = {}
    for t in request_tools.get() or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        params = fn.get("parameters") or fn.get("input_schema") or {}
        if name and isinstance(params, dict):
            schemas[name] = params
    return schemas


def _infer_tool_name(args, schemas):
    """[v77 PR#70 移植] 按 schema 反推：唯一一个恰好接受这些参数名的工具。

    先要求必填字段全齐；如果模型连必填也丢了，仍命名唯一能接受这些
    参数的工具，让客户端报"缺字段"而不是"未知工具"。"""
    for need_required in (True, False):
        fits = []
        for name, params in schemas.items():
            props = set(params.get("properties") or {})
            required = set(params.get("required") or [])
            if set(args) <= props and (not need_required or required <= set(args)):
                fits.append((len(props - set(args)), name))
        fits.sort()
        if fits and (len(fits) == 1 or fits[0][0] < fits[1][0]):
            return fits[0][1]
    return None


def _repair_tool_names(tools):
    schemas = _tool_schemas()
    if not schemas:
        return tools
    for t in tools:
        if not isinstance(t, dict) or "function" not in t:
            continue
        fn = t["function"]
        if fn.get("name") in schemas:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            continue
        if isinstance(args, dict):
            inferred = _infer_tool_name(args, schemas)
            if inferred:
                fn["name"] = inferred
    return tools


# [v77 PR#72 移植] 当前请求的推理档位（thinking 开启时），build_prompt 转成
# 多思考指令。
request_effort = contextvars.ContextVar("request_effort", default=None)


def effort_level(body):
    """[v77 PR#72 移植] 客户端的 effort 字符串，无论方言放在哪个字段。"""
    cands = [body.get("reasoning_effort"), body.get("effort")]
    for key in ("reasoning", "output_config", "thinking"):
        v = body.get(key)
        if isinstance(v, dict):
            cands += [v.get("effort"), v.get("reasoning_effort"), v.get("level")]
        elif key == "thinking" and isinstance(v, str):
            cands.append(v)
    for c in cands:
        if isinstance(c, str) and c.strip():
            return c.strip().lower()
    return None


def parse_tools(text):
    tools = []
    clean_text = text
    fence_spans = _code_fence_spans(text)
    # [v66] think 块是私有推理，不是工具调用区。模型会在 think 里讨论/引用
    # 标记原文（实证：2026-10-25 会话 dee8cb2a，模型在 think 里写"收尾标签
    # 被打坏成 </｜｜DSML｜｜_call>"，DSML 搜刮器随即从这句引用里抠出残名
    # "_call"，产出 {"name": "_call"} 无效调用，真调用整个丢失）。所有扫描
    # 一律跳过 think 区间。
    think_spans = [m.span() for m in _THINK_SPAN_RE.finditer(text)]

    def fenced(pos):
        return any(s <= pos < e for s, e in fence_spans)

    def hidden(pos):
        if any(s <= pos < e for s, e in fence_spans):
            return True
        return any(s <= pos < e for s, e in think_spans)

    param_names = {"command", "description", "file_path", "content", "path", "prompt", "query", "subject", "old_string", "new_string", "url", "input"}
    tool_matches = list(re.finditer(r"<[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|invoke|function_calls?)\s+(?:name|tool)=[\x27\x22]([^\x27\x22]+)[\x27\x22][^>]*>", text, re.IGNORECASE))
    real_tool_matches = [tm for tm in tool_matches if not hidden(tm.start())]

    if real_tool_matches:
        for i, tm in enumerate(real_tool_matches):
            candidate_name = tm.group(1).strip()
            start_idx = tm.end()
            end_idx = real_tool_matches[i+1].start() if i + 1 < len(real_tool_matches) else len(text)
            body = text[start_idx:end_idx]
            args = {}
            p_matches = re.finditer(r"<[/]?[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:parameter|tool_call|param|invoke)\s+name=[\x27\x22]([^\x27\x22]+)[\x27\x22][^>]*>(.*?)(?:(?=<[/]?[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:parameter|tool_call|param|invoke)\b)|(?:(?<=^)|(?<=[^\n<｜\|/ \t]))[｜\| \t]{0,8}DSML[｜\| \t]{0,8}\s*>|</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:parameter|tool_call|param|invoke)\b[^>]*>|$)", body, flags=re.DOTALL | re.IGNORECASE)
            for pm in p_matches:
                p_name = pm.group(1).strip()
                p_val = _clean_param_value(pm.group(2))
                try:
                    val = json.loads(_repair_truncated_json(p_val, strict=True))
                except Exception:
                    val = p_val
                # [v75-a] 外层包装名解包：模型模仿 <tool_call> 结构时会写
                # parameter name="arguments" 把整个参数包起来，直接嵌套会产出
                # {"arguments": {...}} 双层结构，执行器找不到必填参数整条拒收
                # （生产实证 2026-10-25 18:24 会话 e1751fe3）。此处合并解包。
                if p_name in _DSML_ARGS_WRAPPERS:
                    if isinstance(val, dict):
                        args.update(val)
                        continue
                    if isinstance(val, str):
                        inner = None
                        try:
                            inner = json.loads(val)
                        except Exception:
                            # 尾部垃圾（模型多写的闭合括号等）：raw_decode 只取
                            # 开头完整的 JSON 值，其余忽略；非 JSON 文本直接失败
                            try:
                                inner, _end = json.JSONDecoder().raw_decode(val.lstrip())
                            except Exception:
                                inner = None
                        if isinstance(inner, dict):
                            args.update(inner)
                            continue
                    args[p_name] = val
                    continue
                # [v75-b] 容器类型还原为字符串：工具参数均为字符串类型（模型
                # 输出标注 string="true"）。JSON 文本（如 file_write 的
                # content payload）必须保持文本，禁止 json.loads 把字符串变成
                # 对象——客户端 file_write 只收字符串，会以"缺少 content"拒收
                # （生产实证 2026-10-25 当天连拒 3 次）。标量仍正常转换。
                # [v77 PR#70 移植] string="true" 标注 = 字面字符串："42"/"true"
                # 不得变成 JSON。放在包装名解包之后——历史实证（v75 Fix 1）：
                # 模型误用标注把 JSON 对象标成 string="true" 包在 arguments
                # 包装里，守卫若在解包前生效会把双层嵌套放回去。
                if re.search(r"string\s*=\s*[\x27\x22]true", pm.group(0), re.IGNORECASE):
                    args[p_name] = p_val
                    continue
                if isinstance(val, (dict, list)):
                    val = p_val
                args[p_name] = val
            tag_param_matches = re.finditer(r"<([A-Za-z0-9_\-]+)>(.*?)(?:</\1>|$)", body, flags=re.DOTALL | re.IGNORECASE)
            for pm in tag_param_matches:
                t_name = pm.group(1).strip().lower()
                if t_name in param_names:
                    t_val = _clean_param_value(pm.group(2))
                    if re.search(r"string\s*=\s*[\x27\x22]true", tm.group(0), re.IGNORECASE):
                        t_val_parsed = t_val
                    else:
                        try:
                            t_val_parsed = json.loads(_repair_truncated_json(t_val, strict=True))
                        except Exception:
                            t_val_parsed = t_val
                    if isinstance(t_val_parsed, (dict, list)):
                        t_val_parsed = t_val
                    args[t_name] = t_val_parsed
            if candidate_name:
                norm = normalize_tool_call(candidate_name, args)
                if norm:
                    tools.append(norm)

    if tools:
        clean_text = re.sub(r"<[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|calls)[^>]*>.*?(?:</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|calls)>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
        clean_text = re.sub(r"<[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:invoke|function_call)[^>]*>.*?(?:</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:invoke|function_call)>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE).strip()

    # [v66] 精确格式优先：正文（think 区间外）存在 <tool_call>/<function_call>
    # 精确开标签时，绝不进 DSML 搜刮器。搜刮器最后的无 '<' 前缀搜索会把
    # 打坏的收尾标签当成工具名，产出 {"name": "_call"} 无效调用，同时把
    # 旁边完好无损的真调用整个挤掉（生产实证：会话 dee8cb2a 连续 3 轮
    # file_write/shell_execute 全部变成 "_call 无效命令"）。
    _exact_openers = [m for m in _EXACT_CALL_OPEN_RE.finditer(text)
                      if not any(s <= m.start() < e for s, e in think_spans)]
    if not tools and "DSML" in text and not _exact_openers:
        dsml_block_pattern = re.compile(r"<[｜\|]{2}DSML[｜\|]{2}([A-Za-z0-9_]+)>(.*?)(?:</[｜\|]{2}DSML[｜\|]{2}\1>|$)", re.DOTALL | re.IGNORECASE)
        param_pattern_b = re.compile(r"<[｜\|]{2}DSML[｜\|]{2}B([A-Za-z0-9_]+)[^>]*>(.*?)(?:</[｜\|]{2}DSML[｜\|]{2}B.*?>|$)", re.DOTALL | re.IGNORECASE)
        for m in dsml_block_pattern.finditer(text):
            if hidden(m.start()):
                continue
            tool_name = m.group(1).strip()
            # [v66] 下划线开头的"名字"是打坏收尾标签的残片（如 _call），不是工具
            if tool_name.startswith("_"):
                continue
            body = m.group(2)
            args = {}
            for pm in param_pattern_b.finditer(body):
                p_name = pm.group(1).lower().strip()
                p_val = _clean_param_value(pm.group(2))
                if re.search(r"string\s*=\s*[\x27\x22]true", pm.group(0), re.IGNORECASE):
                    args[p_name] = p_val
                    continue
                try:
                    val = json.loads(_repair_truncated_json(p_val, strict=True))
                except Exception:
                    val = p_val
                # [v75-b] 同主分支：容器类型还原为字符串
                # [v77 PR#70 移植] string="true" 标注 = 字面字符串："42"/"true"
                # 不得变成 JSON。放在包装名解包之后——历史实证（v75 Fix 1）：
                # 模型误用标注把 JSON 对象标成 string="true" 包在 arguments
                # 包装里，守卫若在解包前生效会把双层嵌套放回去。
                if re.search(r"string\s*=\s*[\x27\x22]true", pm.group(0), re.IGNORECASE):
                    args[p_name] = p_val
                    continue
                if isinstance(val, (dict, list)):
                    val = p_val
                args[p_name] = val
            norm = normalize_tool_call(tool_name, args)
            if norm:
                tools.append(norm)
        if not tools:
            tool_match = re.search(r"[｜\|]{2}DSML[｜\|]{2}(Bash|Read|Write|Edit|Agent|TaskList|TaskCreate|WebSearch|[A-Za-z0-9_]+)", text, re.IGNORECASE)
            if tool_match and not hidden(tool_match.start()) and not tool_match.group(1).startswith("_"):
                candidate = tool_match.group(1).strip()
                tool_name = "Bash" if candidate.lower().startswith("b") and candidate.lower() not in ["bdescription", "bparam"] else candidate
                args = {}
                cmd_match = re.search(r"[｜\|]{2}B[\x22\x27]?command[\x22\x27]?[^>]*>(.*?)(?:</[｜\|]{2}B|$)", text, re.DOTALL | re.IGNORECASE)
                desc_match = re.search(r"[｜\|]{2}B[\x22\x27]?description[\x22\x27]?[^>]*>(.*?)(?:</[｜\|]{2}B|$)", text, re.DOTALL | re.IGNORECASE)
                if cmd_match:
                    clean_cmd = re.sub(r"</?[｜\|]{2}DSML[｜\|]{2}[^>]*>", "", cmd_match.group(1)).strip("\x22\x27() ")
                    args["command"] = clean_cmd
                if desc_match:
                    clean_desc = re.sub(r"</?[｜\|]{2}DSML[｜\|]{2}[^>]*>", "", desc_match.group(1)).strip("\x22\x27() ")
                    args["description"] = clean_desc
                norm = normalize_tool_call(tool_name, args)
                if norm:
                    tools.append(norm)
        # [v66] 无论是否搜刮出工具，DSML 残渣都不许进正文——打坏的收尾标签
        # 残留在 flush 尾巴里会原样漏给客户端显示。
        clean_text = re.sub(r"<[｜\|]{2}DSML[｜\|]{2}[^>]*>.*?(?:</[｜\|]{2}DSML[｜\|]{2}[^>]*>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
        clean_text = re.sub(r"</?[｜\|]{2}DSML[｜\|]{2}[^>]*>", "", clean_text, flags=re.IGNORECASE).strip()

    if not tools:
        fn_call_pattern = re.compile(r"<function_call>\s*<name>([^<]+)</name>\s*<arguments>(.*?)</arguments>\s*</function_call>", re.DOTALL | re.IGNORECASE)
        for m in fn_call_pattern.finditer(text):
            if hidden(m.start()):
                continue
            name = m.group(1).strip()
            args_raw = m.group(2).strip()
            try:
                args = json.loads(args_raw)
            except Exception:
                args = args_raw
            norm = normalize_tool_call(name, args)
            if norm:
                tools.append(norm)
        if tools:
            clean_text = re.sub(r"<function_call>.*?</function_call>", "", clean_text, flags=re.DOTALL | re.IGNORECASE).strip()

    if not tools:
        # [v76-a] 标签名补 call：模型会混用第三种写法 <call tool="名字">{...}</call>
        # （生产实证 2026-10-26 会话 3e4457c5，shell_execute 调用整段泄露进正文）。
        # 该路径本就支持 tool= 属性，只缺标签名。属性分支自动把 JSON 顶层键当参数。
        tag_regex = re.compile(r"<(?:tool_call|function_call|call)(?:\s+(?:name|tool|function)=[\x27\x22]([^\x27\x22]+)[\x27\x22])?\s*>", re.IGNORECASE)
        decoder = json.JSONDecoder()
        matches = [m for m in tag_regex.finditer(text) if not hidden(m.start())]
        if matches:
            for m in matches:
                tag_name = m.group(1)
                after_tag = text[m.end():]
                brace_pos = after_tag.find("{")
                if brace_pos != -1:
                    json_substr = after_tag[brace_pos:]
                    data = None
                    try:
                        data, _ = decoder.raw_decode(json_substr)
                    except Exception:
                        pass
                    if not data:
                        cleaned_json = re.sub(r"</?(?:tool_call|function_call|tool_calls|invoke|call(?=[\s>/]))[^>]*>.*", "", json_substr, flags=re.DOTALL).strip()
                        open_b = cleaned_json.count("{")
                        close_b = cleaned_json.count("}")
                        if open_b > close_b:
                            cleaned_json += "}" * (open_b - close_b)
                        try:
                            data = json.loads(cleaned_json)
                        except Exception:
                            pass
                    if isinstance(data, dict):
                        if tag_name:
                            name = tag_name
                            if "arguments" in data and isinstance(data["arguments"], dict):
                                args = data["arguments"]
                            elif "parameters" in data and isinstance(data["parameters"], dict):
                                args = data["parameters"]
                            elif "input" in data and isinstance(data["input"], dict):
                                args = data["input"]
                            else:
                                args = {k: v for k, v in data.items() if k not in ["name", "tool", "function"]}
                        else:
                            name = data.get("name") or data.get("tool") or data.get("tool_name") or data.get("function") or data.get("action")
                            args = data.get("arguments") or data.get("parameters") or data.get("input") or data.get("args") or data.get("params") or data.get("tool_input") or data.get("action_input")
                            if args is None:
                                args = {}
                        if name:
                            norm = normalize_tool_call(name, args)
                            if norm:
                                tools.append(norm)
            # [v76-a] call 后必须紧跟空白或 >（call(?=[\s>])）：防止 <call_call>
            # 这类畸形标签被当作 <call> 开标签、DOTALL 吞掉后续正文
            # （回放实证 1792866321_stream_ce93ffb8）。
            clean_text = re.sub(r"<(?:tool_call|function_call|call(?=[\s>]))[^>]*>.*?(?:</(?:tool_call|function_call|call)>|$)(?:\s*(?:参数|parameter|param)\s*>)?", "", text, flags=re.DOTALL).strip()

    if not tools:
        codeblock_pattern = r"```(?:tool_call|function_call)\s*(.*?)\s*```"
        cb_matches = list(re.finditer(codeblock_pattern, clean_text, flags=re.DOTALL))
        for m in cb_matches:
            cleaned = clean_json_str(m.group(1))
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    name = data.get("name") or data.get("tool") or data.get("function") or data.get("action")
                    args = data.get("arguments") or data.get("parameters") or data.get("input") or data.get("args") or {}
                    norm = normalize_tool_call(name, args)
                    if norm:
                        tools.append(norm)
            except Exception:
                pass
        if tools:
            clean_text = re.sub(codeblock_pattern, "", clean_text, flags=re.DOTALL).strip()

    if not tools:
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        json_matches = list(re.finditer(json_pattern, clean_text, flags=re.DOTALL))
        for m in json_matches:
            cleaned = clean_json_str(m.group(1))
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and ("name" in data or "tool" in data or "function" in data):
                    name = data.get("name") or data.get("tool") or data.get("function") or data.get("action")
                    args = data.get("arguments") or data.get("parameters") or data.get("input") or data.get("args") or {}
                    norm = normalize_tool_call(name, args)
                    if norm:
                        tools.append(norm)
            except Exception:
                pass
        if tools:
            clean_text = re.sub(json_pattern, "", clean_text, flags=re.DOTALL).strip()
    # Keep companion text: the tool-call blocks themselves were already removed
    # from clean_text above; only leftover bare tags are stripped here.
    clean_text = _strip_wrapper_tags(clean_text).strip()
    # [v77 PR#70 移植] 模型丢掉工具名时按请求的 tool schema 反推
    tools = _repair_tool_names(tools)
    return tools, clean_text


# Family-wide closer pattern for the tool-call wrapper family, including
# the |/｜-decorated variants that parse_tools accepts. Used by
# StreamToolParser so a mismatched closer closes the open block instead of
# hanging until flush(). [FIX 3]
_TOOL_END_TAG_RE = re.compile(
    r"</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|calls|invoke|function_calls?)\b[^>]*>",
    re.IGNORECASE,
)

# [FIX 4] Spaced-DSML dialect: deepseek-harness emits decorated tags with a
# space between the ｜｜DSML｜｜ marker and the tag name, e.g.
# "<｜｜DSML｜｜ invoke name=...>". Entry detection cannot rely on plain
# substring start tags; this regex accepts bars, the DSML marker and the
# space in any combination.
_STREAM_ENTRY_RE = re.compile(
    r"<[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(tool_calls?|calls|function_calls?|invoke)\b[^>]*>",
    re.IGNORECASE,
)

# Orphan closers trailing a block already closed by the per-tag or family
# fallback (e.g. "</｜｜DSML｜｜ calls>" after the inner invoke was flushed).
_ORPHAN_CLOSER_RE = re.compile(
    r"</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|calls|function_calls?|invoke)\b[^>]*>",
    re.IGNORECASE,
)

# [FIX 5] Bare DSML marker: a "<｜｜DSML｜｜>" / "</｜｜DSML｜｜>" whose tag-name
# slot is empty (nothing between the DSML marker and ">"). Every other cleanup
# regex in this module requires an actual tag name (tool_calls/calls/invoke/
# function_call/parameter), so these bare markers used to slip through and get
# streamed to the client as literal text. Note the tag-name slot is required to
# be EMPTY: "<｜｜DSML｜｜ invoke ...>" does not match and is left for the entry
# regex.
_BARE_DSML_MARKER_RE = re.compile(
    r"</?[｜\| \t]{0,8}DSML[｜\| \t]{0,8}\s*>(?!\s*(?:tool_calls?|calls|function_calls?|invoke|parameter)\b)",
    re.IGNORECASE,
)


# Full wrapper-tag family (open or close, with an optional tag name). Used on
# the non-streaming path and on flush() to strip wrapper noise; unlike
# _ORPHAN_CLOSER_RE it also removes stray OPEN tags.
_WRAPPER_TAG_RE = re.compile(
    r"</?[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|calls|function_calls?|invoke|parameter)[^>]*>",
    re.IGNORECASE,
)

# [v33] Decorated tag with an ARBITRARY tag name. The fixed keyword lists above
# only know tool_calls/calls/invoke/parameter, but the upstream harness also
# emits closing tags whose name slot holds a parameter key or attribute residue:
#     </｜｜DSML｜｜ tool_title" string="true">
# which used to survive every rule and land inside the client's Bash command
# (-> /bin/sh: unterminated quoted string). Anything carrying the DSML marker is
# protocol garbage regardless of the name that follows.
_DECOR_TAG_RE = re.compile(
    r"</?[｜\| \t]{0,8}DSML[｜\| \t]{0,8}\s*[^\s<>]{0,40}[^<\n]{0,120}>",
    re.IGNORECASE,
)

# [v33] A '<' / '</' left dangling at the end of a flush, with no tag body.
_DANGLING_LT_RE = re.compile(r"<[/｜\| \t]{0,8}$")

# [v33] Closer-only variant for the prose path: never touches an OPEN tag
# (mid-stream prose may legitimately contain one), but kills every closer that
# carries the DSML marker, whatever name sits in the tag-name slot.
_DECOR_CLOSER_RE = re.compile(
    r"</[｜\| \t]{0,8}DSML[｜\| \t]{0,8}\s*[^\s<>]{0,40}[^<\n]{0,120}>",
    re.IGNORECASE,
)

# [v33] A "</" that is not followed by an ASCII letter can never open a real
# closing tag. This is what leaked into Bash commands as a bare "</".
_BARE_CLOSE_RE = re.compile(r"</(?![A-Za-z])")


_DECOR_BLOCK_RE = re.compile(
    r"<[|｜ \t]{0,8}DSML[|｜ \t]{0,8}[^>\n]{0,200}>"
    r"[^\n]*?"
    r"(?:</?)?[|｜ \t]{0,8}DSML[|｜ \t]{0,8}[^>\n]{0,200}>",
    re.IGNORECASE,
)

_DECOR_OPEN_RE = re.compile(
    r"<[|｜ \t]{0,8}DSML[|｜ \t]{0,8}[^>\n]{0,200}>",
    re.IGNORECASE,
)


def _strip_stray_markers(text: str) -> str:
    """Prose path: drop bare DSML markers + orphan closers only.

    Open tags are left untouched here so literal prose mentioning e.g.
    "" is not silently deleted mid-stream.
    """
    if not text:
        return text
    # [v57 移植 v39] whole marked block on one line (open + content + close)
    text = _DECOR_BLOCK_RE.sub("", text)
    # [v57 移植 v39] lone marked open tag with any name
    text = _DECOR_OPEN_RE.sub("", text)
    text = _sub_bare_markers(text)
    text = _ORPHAN_CLOSER_RE.sub("", text)
    # [v33] closer with arbitrary tag name, then any impossible "</"
    text = _DECOR_CLOSER_RE.sub("", text)
    text = _BARE_CLOSE_RE.sub("", text)
    return text


def _sub_bare_markers(text: str) -> str:
    """Drop bare DSML markers, but NEVER when what follows could be the head
    of a tool tag. Streaming chunks arrive mid-tag (e.g. "<｜DSML｜>invok"),
    so a plain negative lookahead on complete keywords is not enough: the
    keyword itself may still be incomplete. Keep the marker if the following
    text is a prefix of, or starts with, any tool-tag keyword."""
    def _repl(m):
        tail = text[m.end():m.end() + 24].lstrip()
        low = tail.lower()
        if not low:
            # Nothing follows the marker: no tool head can start here — drop
            # it. (k.startswith("") is vacuously True and used to keep
            # end-of-buffer bare markers alive, leaking them to clients.)
            return ""
        for k in ("tool_call", "calls", "function_call", "invoke", "parameter"):
            if low.startswith(k) or k.startswith(low):
                return m.group(0)
        # [v24] the marker itself is garbage; a short trailing fragment the
        # model emitted on the same construct (e.g. "<｜｜DSML｜｜>参数>")
        # is garbage too — drop it together with the marker.
        return ""
    # [v24] marker+fragment PAIRS first (while the marker is still there to
    # anchor the fragment), then any remaining bare markers.
    _DECOR_FRAGMENT_RE = re.compile(r"</?[｜\| \t]{0,8}DSML[｜\| \t]{0,8}\s*>[^<\n]{0,12}>", re.IGNORECASE)
    bare = _DECOR_FRAGMENT_RE.sub("", text)
    return _BARE_DSML_MARKER_RE.sub(_repl, bare)


def _strip_wrapper_tags(text: str) -> str:
    """Wrapper path: drop the whole tag family (open or close) + bare markers."""
    if not text:
        return text
    text = _sub_bare_markers(text)
    text = _WRAPPER_TAG_RE.sub("", text)
    # [v33] whatever the keyword lists missed, the marker decides
    text = _DECOR_TAG_RE.sub("", text)
    text = _BARE_CLOSE_RE.sub("", text)
    return text


def _append_clean_text(results, text: str) -> None:
    """Append cleaned prose, skipping empty deltas after marker stripping."""
    cleaned = _strip_stray_markers(text)
    if cleaned:
        results.append({"text": cleaned})


# Tag names _STREAM_ENTRY_RE can open on.
_STREAM_ENTRY_TAGS = ("tool_calls", "tool_call", "function_calls", "function_call", "invoke", "calls")

def _skip_bars(text, pos):
    # Skip any mix of bars/spaces/tabs (decorated dialects interleave them,
    # e.g. "| | DSML | | >"); newlines are NOT decor — never cross lines.
    while pos < len(text) and text[pos] in "|｜ \t":
        pos += 1
    return pos

def _is_plausible_bare_opener(line):
    """True while `line` (no newline yet) could still become a bare opener
    (decorated DSML tag missing its leading '<'). Prefix-tolerant: '|',
    '| ', '| | D' all hold; '| a' (markdown table) does not."""
    s = line.lstrip()
    if not s or s.startswith("<"):
        return False
    m = re.match(r"^[｜\| \t]+", s)
    if not m:
        return False
    rest = s[m.end():]
    if not rest:
        return True  # pure decoration so far
    m2 = re.match(r"^D(?:S(?:M(?:L?)?)?)?", rest, re.IGNORECASE)
    if not m2:
        return False
    rest2 = rest[m2.end():]
    if not rest2:
        return True
    m3 = re.match(r"^[｜\| \t]+", rest2)
    if not m3:
        return False
    rest3 = rest2[m3.end():]
    if not rest3:
        return True
    low = rest3.lower()
    return any(k.startswith(low) for k in ("tool_call", "calls", "function_call", "invoke", "parameter"))


def _is_plausible_stream_entry_prefix(segment: str) -> bool:
    if not segment.startswith("<"):
        return False
    body = segment[1:]
    pos = _skip_bars(body, 0)
    length = len(body)
    if pos < length and body[pos] in "Dd":
        matched = 0
        while matched < 4 and pos < length and body[pos].upper() == "DSML"[matched]:
            pos += 1
            matched += 1
        if matched < 4:
            return ">" not in body   # partial "DSML" — only plausible while no stray '>'
        pos = _skip_bars(body, pos)
        # B-dialect decorated '>' ("<｜DSML｜>invoke ...") may arrive mid-entry
        # while the tag/attributes are still streaming in. It is part of the
        # decoration, not the entry terminator; skip it and keep holding.
        if pos < length and body[pos] == ">":
            pos += 1
    else:
        # No DSML decoration: any '>' here means the segment is not a
        # plausible incomplete entry prefix.
        if ">" in body:
            return False
    while pos < length and body[pos].isspace():
        pos += 1
    if pos == length:
        return True
    remainder = body[pos:]
    if ">" in remainder:
        return False
    remainder_lower = remainder.lower()
    for tag in _STREAM_ENTRY_TAGS:
        if tag.startswith(remainder_lower):
            return True
        if remainder_lower.startswith(tag):
            suffix = remainder[len(tag):]
            if not suffix or not (suffix[0].isalnum() or suffix[0] == "_"):
                return True
    return False

def _is_plausible_stream_closer_prefix(segment: str) -> bool:
    return segment.startswith("</") and _is_plausible_stream_entry_prefix("<" + segment[2:])


class StreamToolParser:
    def __init__(self):
        self.buffer = ""
        self.in_tool = False
        self.has_tool = False
        self.json_done = False
        self._end_re = None
        # [v59] 工具发射后吞掉后续纯空白正文。工具标记内部的换行
        # （模型在 </…invoke> 与 </…calls> 之间写的 "\n"）会被家族
        # 宽泛闭合 [FIX 3] 在 invoke 闭合处提前收块，尾部那个 "\n"
        # 于是当 prose 漏进 content——每个工具轮正文因此多一个孤
        # 换行（2026-10-24 用户报告"调用工具正文就多出换行"的另
        # 一半；前一半是桥自加的前导 "\n"，已在 app.py splitter
        # 剥掉）。真 prose（非纯空白）不受影响，照常放行并清标志。
        self._post_tool_ws = False

    def _emit_text(self, results, text: str) -> None:
        """追加正文；若上一个产出是工具调用且本文本纯空白，则吞掉。"""
        cleaned = _strip_stray_markers(text)
        if not cleaned:
            return
        if self._post_tool_ws and not cleaned.strip():
            return
        results.append({"text": cleaned})
        self._post_tool_ws = False

    def feed(self, chunk):
        self.buffer += chunk
        results = []
        while True:
            if self.in_tool:
                # [FIX 3] Family-wide closer fallback: accept ANY wrapper closer
                # the tool-call family can emit (including |/｜-decorated
                # variants parse_tools tolerates) instead of hanging an open
                # block until flush() when the model closes with a wrong tag.
                # [FIX 4] Prefer the closer for the tag that OPENED the block
                # so nested wrappers (<｜｜DSML｜｜ calls> wrapping invokes)
                # close as one unit; fall back to the family-wide closer for
                # mismatched or decorated closers [FIX 3].
                end_match = self._end_re.search(self.buffer) if self._end_re else None
                if end_match is None:
                    end_match = _TOOL_END_TAG_RE.search(self.buffer)
                if end_match:
                    if not self.json_done:
                        tool_xml = self.buffer[: end_match.end()]
                        parsed, _ = parse_tools(tool_xml)
                        for item in parsed:
                            if _valid_tool_call(item):
                                results.append({"tool": item})
                                self._post_tool_ws = True
                    self.buffer = self.buffer[end_match.end():]
                    self.in_tool = False
                    self.json_done = False
                    self._end_re = None
                    continue
                brace_idx = self.buffer.find("{")
                if brace_idx != -1 and not self.json_done:
                    decoder = json.JSONDecoder()
                    try:
                        data, consumed = decoder.raw_decode(self.buffer[brace_idx:])
                        norm = normalize_tool_call(data)
                        if norm and _valid_tool_call(norm):
                            results.append({"tool": norm})
                            self._post_tool_ws = True
                            self.json_done = True
                            self.buffer = self.buffer[brace_idx + consumed:]
                            continue
                    except Exception:
                        pass
                break
            else:
                # [FIX 6] repair bare decorated openers (missing '<') so the
                # entry regex can catch them instead of leaking as prose.
                self.buffer = _repair_bare_dsml_lines(self.buffer)
                # [FIX 4] Regex entry detection replaces plain substring finds
                # so decorated/spaced DSML openers enter tool mode.
                m = _STREAM_ENTRY_RE.search(self.buffer)
                if m:
                    start = m.start()
                    if start > 0:
                        self._emit_text(results, self.buffer[:start])
                    self.buffer = self.buffer[start:]
                    tag_name = m.group(1).lower()
                    self._end_re = re.compile(
                        r"</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*" + re.escape(tag_name) + r"[｜\| \t]{0,8}\s*>",
                        re.IGNORECASE,
                    )
                    self.in_tool = True
                    self.has_tool = True
                    continue
                # [FIX 8] A bare opener line (missing leading '<', e.g.
                # "| | DSML | | invoke ...") can only be repaired once its
                # full line is in the buffer (_repair_bare_dsml_lines is
                # line-anchored). While the LAST line is still streaming in
                # (prefix could still become decoration+DSML+tag, no newline
                # yet), hold it — but complete earlier lines are emitted.
                last_nl = self.buffer.rfind("\n")
                tail_line = self.buffer[last_nl + 1:]
                if tail_line and _is_plausible_bare_opener(tail_line):
                    if last_nl != -1:
                        self._emit_text(results, self.buffer[:last_nl + 1])
                        self.buffer = tail_line
                    break
                # [FIX 2] Hold an unclosed '<' tail only while it remains a
                # plausible partial match of _STREAM_ENTRY_RE or of a wrapper
                # closer ("</..."), so a split closer is never dumped as prose.
                # [FIX 7] The prefix predicates themselves decide whether an
                # embedded decorated '>' (B-dialect "<｜DSML｜>invoke ...") is
                # acceptable mid-entry — no blanket '" not in tail' short-circuit,
                # which used to dump the tail as prose the moment the dialect's
                # decoration '>' arrived but before the entry's own '>' did.
                last_lt = self.buffer.rfind("<")
                tail = self.buffer[last_lt:] if last_lt != -1 else ""
                hold = last_lt != -1 and (
                    _is_plausible_stream_entry_prefix(tail)
                    or _is_plausible_stream_closer_prefix(tail)
                )
                if last_lt != -1 and hold:
                    if last_lt > 0:
                        self._emit_text(results, self.buffer[:last_lt])
                    self.buffer = tail
                    break
                if self.buffer:
                    self._emit_text(results, self.buffer)
                self.buffer = ""
                break
        return results

    def flush(self):
        out = []
        if self.buffer and not self.in_tool:
            # [FIX 8] Last chance: the stream ended before the bare-opener
            # line got its newline — repair now (end-of-string closes the
            # line for the line-anchored regex), then re-dispatch.
            if True:
                repaired = _repair_bare_dsml_lines(self.buffer)
                m = _STREAM_ENTRY_RE.search(repaired)
                if m:
                    start = m.start()
                    pre = repaired[:start]
                    if pre.strip():
                        self._emit_text(out, pre)
                    parsed, clean_tail = parse_tools(repaired[start:])
                    for item in parsed:
                        if _valid_tool_call(item):
                            out.append({"tool": item})
                            self._post_tool_ws = True
                    if clean_tail.strip():
                        self._emit_text(out, clean_tail)
                    self.buffer = ""
                    return out
            self._emit_text(out, self.buffer)
        elif self.in_tool:
            # [FIX 1] Recover the tool before stripping: if the stream was cut
            # off before the closing tag arrived but the payload itself is
            # parseable, emit the tool call instead of dumping raw parameter
            # values into the chat text. Strip only when nothing parses.
            parsed, _ = parse_tools(self.buffer)
            if parsed:
                for item in parsed:
                    if _valid_tool_call(item):
                        out.append({"tool": item})
                        self._post_tool_ws = True
            elif not self.json_done:
                stripped = _strip_wrapper_tags(self.buffer).strip()
                if stripped:
                    out.append({"text": stripped})
            # With json_done set, whatever is left after the consumed JSON
            # payload is wrapper noise (partial closers / whitespace) and is
            # dropped instead of leaking into the chat text.
        self.buffer = ""
        self.in_tool = False
        self.json_done = False
        self._end_re = None
        return out


def summarize_messages(messages, max_tokens=500):
    recent = messages[-10:] if len(messages) > 10 else messages
    parts = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        if content:
            parts.append(f"{role}: {content[:200]}")
    summary = "\n".join(parts)
    tokens = count_tokens(summary)
    while tokens > max_tokens and len(parts) > 1:
        parts = parts[1:]
        summary = "\n".join(parts)
        tokens = count_tokens(summary)
    return summary


_pow_setup = None


def _get_pow_setup():
    global _pow_setup
    if _pow_setup is None:
        engine = wasmtime.Engine()
        with open(wasm_path, "rb") as f:
            module = wasmtime.Module(engine, f.read())
        _pow_setup = (engine, module, wasmtime.Linker(engine))
    return _pow_setup


def _find_pow_answer_blocking(challange_data):
    engine, module, linker = _get_pow_setup()
    store = wasmtime.Store(engine)
    instance = linker.instantiate(store, module)
    memory = instance.exports(store)["memory"]
    alloc_func = instance.exports(store)["alloc"]
    solve_func = instance.exports(store)["solve_pow"]
    ch_ptr, ch_len = write_string_pow(challange_data["challenge"], alloc_func, memory, store)
    salt_ptr, salt_len = write_string_pow(challange_data["salt"], alloc_func, memory, store)
    result = solve_func(store, ch_ptr, ch_len, salt_ptr, salt_len, challange_data["expire_at"], challange_data["difficulty"])
    if result < 0:
        result = result + 0x10000000000000000
    return result if result != 0xFFFFFFFFFFFFFFFF else None


async def create_challange_pow(target_path, auth_token):
    headers = get_headers(auth_token)
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    response = await post_with_failover(
        "/api/v0/chat/create_pow_challenge",
        headers=headers, json={"target_path": target_path},
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=20),
    )
    async with response:
        data = await response.json()
    return data["data"]["biz_data"]["challenge"]


def write_string_pow(text, alloc_func, memory, store):
    data = text.encode("utf-8")
    ptr = alloc_func(store, len(data))
    mem = memory.data_ptr(store)
    for i in range(len(data)):
        mem[ptr + i] = data[i]
    return ptr, len(data)


async def find_pow_answer(challange_data):
    return await asyncio.to_thread(_find_pow_answer_blocking, challange_data)


async def solve_create_pow(target_path, auth_token):
    pow = await create_challange_pow(target_path, auth_token)
    answer = await find_pow_answer(pow)
    if answer is None:
        raise Exception("PoW solve failed")
    json_data = {
        "algorithm": "DeepSeekHashV1",
        "challenge": pow["challenge"],
        "salt": pow["salt"],
        "answer": answer,
        "signature": pow["signature"],
        "target_path": target_path,
    }
    return base64.b64encode(json.dumps(json_data).encode()).decode()


# ── PoW prefetch pool (v26) ─────────────────────────────────────────────
# Every completion turn used to pay a full create_pow_challenge round-trip
# (~200-400ms) plus wasm solve CPU before the real request could fire. The
# challenge carries its own expire_at, so we pre-solve one in the background
# right after each turn completes and the next turn consumes it instantly.
# Request rate towards create_pow_challenge is unchanged (1 per turn), only
# its timing moves earlier — zero semantic change for the upstream.
_POW_PREFETCH: dict = {}
_POW_PREFETCH_MAX = 8


async def _prefetch_pow(auth_token, target_path="/api/v0/chat/completion"):
    try:
        p = await solve_create_pow(target_path, auth_token)
        try:
            data = json.loads(base64.b64decode(p))
            expire_at = float(data.get("expire_at") or 0)
        except Exception:
            expire_at = 0
        key = (auth_token, target_path)
        if len(_POW_PREFETCH) >= _POW_PREFETCH_MAX:
            _POW_PREFETCH.pop(next(iter(_POW_PREFETCH)), None)
        _POW_PREFETCH[key] = (p, expire_at)
    except Exception:
        pass


def _take_prefetched_pow(auth_token, target_path="/api/v0/chat/completion"):
    entry = _POW_PREFETCH.pop((auth_token, target_path), None)
    if not entry:
        return None
    p, expire_at = entry
    if expire_at and expire_at < time.time() + 30:
        return None  # too close to expiry — solve fresh
    return p


async def warmup_pool(max_tokens=3):
    """[v57 移植 v35] 启动预热：把冷启动 ~30s 的开销前移到启动阶段。

    三件事依次做完，第一个用户请求就不用再付：
      1. wasm PoW 模块编译（CPU 密集，走线程池）
      2. 与上游 chat.deepseek.com 建立 TLS 连接（跨洋握手最贵）
      3. 用 ACTIVE token 预解 PoW，填满 _POW_PREFETCH
    任何一步失败只记日志，绝不影响服务启动。
    """
    t0 = time.time()
    try:
        await asyncio.to_thread(_get_pow_setup)
    except Exception as exc:
        logger.warning("[warmup] wasm 预编译失败: %s", exc)
    try:
        await get_session()
    except Exception as exc:
        logger.warning("[warmup] 上游会话建立失败: %s", exc)
    try:
        toks = await asyncio.to_thread(get_tokens)
        actives = [t for t in toks if t.get("status") == "ACTIVE"][:max_tokens]
        for t in actives:
            try:
                await _prefetch_pow(t["token"], "/api/v0/chat/completion")
            except Exception:
                pass
        logger.info("[warmup] token=%d 预热完成", len(actives))
    except Exception as exc:
        logger.warning("[warmup] PoW 预热失败: %s", exc)
    logger.info("[warmup] 全部完成，用时 %.1fs，PoW 池 %d 条", time.time() - t0, len(_POW_PREFETCH))


async def create_new_chat(auth_token):
    headers = get_headers(auth_token)
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    response = await post_with_failover(
        "/api/v0/chat_session/create",
        headers=headers,
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=20),
    )
    async with response:
        data = await response.json()
    return data["data"]["biz_data"]["chat_session"]["id"]


_EMPTY_SSE_DEFAULT = (
    "Empty response from DeepSeek (no parseable SSE content): "
    "stream ended with no assistant fragments"
)
_EMPTY_SSE_CONTEXT = (
    "Empty response from DeepSeek (prompt may exceed the session context limit)"
)
_CONTEXT_LIMIT_HINT_RE = re.compile(
    r"context|token.?limit|maximum.?context|too.?long|exceed",
    re.IGNORECASE,
)
_SSE_RECENT_LINES = 8


class _SseStreamFinished(Exception):
    pass


class _SseStreamInterrupted(Exception):
    """上游流在收到 FINISHED 之前异常结束（断流/断连/上游提前关闭）。

    [v62] reason/detail 载体：让 send_message 的续写循环按中断原因选择
    重投提示词——畸形工具调用场景改用带诊断的提示词（告诉模型错在哪、
    附原始调用、给正确格式），其余场景仍用通用的 _CONTINUE_PROMPT。
    默认参数保证既有裸抛（stall/断流/no_body）零改动、行为不变。"""

    def __init__(self, reason="interrupted", detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class _RetryBoundary(str):
    """[v61] 续写轮边界哨兵：空串 str 子类，只在 send_message 的续写
    循环向客户端流里插入，告知 app.stream_response「新 attempt 从此开始，
    请重置 splitter」。

    为什么需要它：app._StreamTagSplitter 在第一次见到 </think> 后把
    think_done 闭锁（v46 起，为容忍模型正文里的字面 <think>），此后同一流内
    再出现 <think> 不再被识别——而自动续写轮恰以 _send_message_once 的
    "\\n<think>" 开头（functions.py 两处 yield），续写轮的整段思考会被
    splitter 当 content 放行，泄漏进正文（2026-10-24 21:25 线上实例
    1792849156：attempt-0 正常答完被上游断流打断，attempt-1 的
    “The user says 接着上文继续写...” 整段漏成正文）。
    v39/v40 曾把这个现象误判为「续写轮关闭 thinking 所致」（B3 思考泄漏），
    回退 thinking 参数并不能修它——根因是 splitter 的跨轮状态，与 thinking
    开关无关。

    空串子类对其他消费者透明：full_text += chunk 加空串无害、非流式 join
    无害、任何 len/真值判断视作空串；只有 stream_response 按 isinstance
    拦截重置 splitter。"""
    __slots__ = ()


def _redact_sse_line(line: str) -> str:
    if len(line) > 500:
        return line[:500] + "..."
    return line


def _remember_sse_line(recent_lines, line: str):
    recent_lines.append(_redact_sse_line(line))
    if len(recent_lines) > _SSE_RECENT_LINES:
        del recent_lines[0]


def _sse_context_limit_hint(recent_lines, parsed_events):
    for obj in parsed_events:
        try:
            blob = json.dumps(obj, ensure_ascii=False)
        except Exception:
            blob = str(obj)
        if _CONTEXT_LIMIT_HINT_RE.search(blob):
            return True
    for line in recent_lines:
        if _CONTEXT_LIMIT_HINT_RE.search(line):
            return True
    return False


def _scan_recent_sse_errors(parsed_events):
    """Return (http_status, message) if a recent parsed event is an upstream error."""
    for obj in reversed(parsed_events):
        if not isinstance(obj, dict) or obj.get("type") != "error":
            continue
        content = obj.get("content") or obj.get("message") or "Upstream error"
        finish = str(obj.get("finish_reason") or "")
        if finish == "rate_limit_reached" or "rate_limit" in finish.lower():
            return 429, content
        if finish in ("permission_denied", "forbidden"):
            return 403, content
        return 502, content
    return None


def _raise_empty_sse_response(recent_lines, parsed_events):
    # [v77 P0-4 移植] 信封内嵌请求关联 id（若中间件已注入），失败响应可与
    # 访问日志行对上。延迟 import 避免依赖方向问题。
    try:
        from middleware import request_id_var
        _rid = request_id_var.get() or "-"
    except Exception:
        _rid = "-"
    err = _scan_recent_sse_errors(parsed_events)
    if err:
        code, msg = err
        logger.warning(
            "DeepSeek SSE error event (not empty stream); last events: %s",
            recent_lines[-_SSE_RECENT_LINES:],
        )
        raise UpstreamError(code, f"{msg} [rid:{_rid}]")
    if _sse_context_limit_hint(recent_lines, parsed_events):
        msg = _EMPTY_SSE_CONTEXT
    else:
        msg = _EMPTY_SSE_DEFAULT
    logger.warning(
        "DeepSeek SSE ended without assistant output; last events: %s",
        recent_lines[-_SSE_RECENT_LINES:],
    )
    raise UpstreamError(502, f"{msg} [rid:{_rid}]")


def _raise_sse_error_event(data, recent_lines):
    content = data.get("content") or data.get("message") or "Upstream error"
    finish = str(data.get("finish_reason") or "")
    if finish == "rate_limit_reached" or "rate_limit" in finish.lower():
        logger.warning(
            "DeepSeek SSE error event (not empty stream); last events: %s",
            recent_lines[-_SSE_RECENT_LINES:],
        )
        raise UpstreamError(429, content)
    if finish in ("permission_denied", "forbidden"):
        raise UpstreamError(403, content)
    raise UpstreamError(502, content)


# [v60] 工具调用标记的结构化特征（原生 tool_call/function_call 家族 +
# DSML 装饰家族）。只匹配标签/装饰结构，不匹配散文里提到的 "DSML"
# 字样，避免把正常讨论误判成调用。用于识别「parser 把畸形调用整块吞掉」
# 的回合：标记在、parse_tools 却解析不出任何工具。
_TOOL_MARKUP_HINT_RE = re.compile(
    r"<\s*/?\s*(?:tool_calls?|function_calls?|invoke|parameter)\b"
    r"|[<｜\|][｜\|][｜\| \t]*DSML",
    re.IGNORECASE,
)


def _has_unswallowed_tool_markup(text: str) -> bool:
    """[v60] 正文中是否存在「未被解析成工具」的工具调用标记。

    覆盖 v58 _body_has_real_content 的两个盲区：
    1. 只认 DSML 标记——原生 <tool_call> 家族整个漏掉；
    2. 只抓「剥掉标记后正文全空」——「真人话 + 被吞的调用」漏掉。
       2026-10-24 12:27:42 线上实例（session bb59b5d3）：模型输出一句
       英文正文 + 丢了 name 字段的原生调用，parser 把调用静默吞掉，
       finish_reason=stop 收尾，下游以为正常答完，会话死掉，用户被迫
       手动发「继续」。

    代码围栏内的标记是示例/文档，不算。think 块里引用的标记原文也不算
    （[v66] 模型会在推理里讨论标记长什么样，实证 dee8cb2a）。能解析出
    工具的正常回合不算（工具走正常投递路径，由下游执行或按 schema 拒绝
    ——拒绝会带错误信息回给模型，模型能自纠，那是 v58 之前就设计好的
    容错路径）。
    """
    if not text:
        return False
    spans = _code_fence_spans(text)
    think_spans = [m.span() for m in _THINK_SPAN_RE.finditer(text)]
    hit = False
    for m in _TOOL_MARKUP_HINT_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        if any(s <= m.start() < e for s, e in think_spans):
            continue
        hit = True
        break
    if not hit:
        return False
    tools, _ = parse_tools(text)
    return not tools


def _body_has_real_content(text: str) -> bool:
    """FINISHED 时，正文剥掉工具调用标记后是否还有可读内容。

    [v58] 畸形 DSML 检测：上游偶发输出只有闭合 invoke 的工具块（开口
    <｜｜DSML｜｜ invoke name=...> 丢失），StreamToolParser 会按
    calls>..</calls> 整块吞掉且 parse_tools 解析不出任何工具。若此时
    FINISHED 正常到达，这一轮会被判"正常完成"，客户端只见思考、正文
    全空（finish_reason=stop）。此处把「纯标记且解析不出工具」按无正文
    处理，触发既有自动续写机制让上游重新产出。

    [v60] 盲区扩展：v58 的条件只覆盖 DSML 家族且「剥完标记正文全空」。
    模型把原生 <tool_call> 的 name 字段整个写丢时（有真人话 + 被吞的
    调用），两个条件都不满足，回合仍会被判正常完成——改用
    _has_unswallowed_tool_markup 统一识别「标记在但解析不出工具」，
    不限家族、不限周围有无正文。

    有可读正文、或标记块能解析出有效工具（下游要投递 tool_calls）时
    返回 True，行为与原先完全一致。
    """
    if not text or not text.strip():
        return False
    if _has_unswallowed_tool_markup(text):
        return False
    if "DSML" not in text:
        return True
    cleaned = re.sub(
        r"<[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|calls)[^>]*>"
        r".*?(?:</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:tool_calls?|calls)>|$)",
        "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(
        r"<[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:invoke|function_call)[^>]*>"
        r".*?(?:</[｜\| \t]{0,8}(?:DSML[｜\| \t]{0,8}>?)?\s*(?:invoke|function_call)>|$)",
        "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned)
    if cleaned.strip():
        return True
    tools, _ = parse_tools(text)
    return bool(tools)


async def _send_message_once(chat_id, auth_token, message, parent_message_id, thinking=False, search=False, file_ids_=None):
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    if parent_message_id == 0:
        parent_message_id = None
    file_ids = file_ids_ or []

    _pow = _take_prefetched_pow(auth_token, "/api/v0/chat/completion")
    if _pow is None:
        _pow = await solve_create_pow("/api/v0/chat/completion", auth_token)
    else:
        # consumed one — refill in the background for the next turn
        asyncio.create_task(_prefetch_pow(auth_token, "/api/v0/chat/completion"))
    headers = get_headers(auth_token, _pow)
    json_data = {
        "chat_session_id": chat_id,
        "parent_message_id": parent_message_id,
        "model_type": DEFAULT_MODEL_TYPE,
        "prompt": message,
        "ref_file_ids": file_ids,
        "thinking_enabled": thinking,
        "search_enabled": search,
        "preempt": False,
        "action": None,
    }

    think_open = False
    got_output = False
    got_body = False
    finished = False
    # [v77 PR#70 移植] DeepSeek 会以 status INCOMPLETE 结束被它截断的回复
    # （长上下文思考中可见）。还没有任何回答文本时这是失败轮，不是完成轮：
    # 空 "stop" 会让 agent 客户端误以为轮次正常结束。
    incomplete = False
    # [v58] 累积非思考正文，供 FINISHED 时判断是否只剩未解析的工具标记
    body_acc = []
    # [v63] 首字节预算提前到 POST 之前定义（S8）：该预算的设计意图本就是
    # "从发请求到见到第一个字节"（原注释：预填期上游"一个字都不吐"，连响应
    # 头都不发），但旧实现只把它用在 body 首字节上，响应头阶段裸奔、只能等
    # session sock_read=600s。提前后 POST 与后续读取共用同一预算：POST 的
    # 响应头等待受 header_timeout 约束，body 首字节受同一 deadline 约束，
    # 语义统一。env 键与默认值不变（DEEPSEEKER_FIRST_BYTE_TIMEOUT=120）。
    _first_byte_budget = float(os.getenv("DEEPSEEKER_FIRST_BYTE_TIMEOUT", "120"))
    _first_byte_deadline = time.time() + _first_byte_budget
    resp = await post_with_failover(
        "/api/v0/chat/completion",
        headers=headers, json=json_data,
        # [v63] S8：响应头阶段独立预算。上游"接了请求不回响应头"时最多
        # 挂起 120s 即取消并 failover（旧行为干等 600s）；正常流 POST
        # 毫秒返回，零开销。
        header_timeout=_first_byte_budget,
        # cookies=cookie,  # Backup WAF fallback
    )
    async with resp:
        if resp.status != 200:
            error_text = await resp.text()
            logger.warning("DeepSeek completion HTTP %d for chat %s: %s", resp.status, chat_id, error_text[:300])
            raise UpstreamError(resp.status, error_text)

        recent_lines = []
        parsed_events = []
        line_buf = b""

        async def _emit_from_event(data):
            nonlocal think_open, got_output, got_body, incomplete
            if isinstance(data, dict) and data.get("type") == "error":
                _raise_sse_error_event(data, recent_lines)

            if data.get("o") == "BATCH" and isinstance(data.get("v"), list):
                for op in data["v"]:
                    if not isinstance(op, dict):
                        continue
                    if op.get("p") == "quasi_status":
                        # [v77 PR#70 移植] 状态词永不属于内容；INCOMPLETE 记账
                        incomplete = incomplete or op.get("v") == "INCOMPLETE"
                        continue
                    async for chunk in _emit_from_event(op):
                        yield chunk
                return

            # [v77 PR#70 移植] 其他 status 字段（"response/status"、".../status"）
            # 携带 INCOMPLETE / WIP 等状态词，永不属于内容。
            _p_name, _p_val = str(data.get("p") or ""), data.get("v")
            if (
                _p_name.endswith("status")
                and isinstance(_p_val, str)
                and _p_val.isupper()
                and _p_val != "FINISHED"
            ):
                incomplete = incomplete or _p_val == "INCOMPLETE"
                return

            if data.get("p") == "response/status" and data.get("v") == "FINISHED":
                if think_open:
                    yield "</think>"
                # [v77 PR#70 移植适配] INCOMPLETE 且无任何回答 = 上游截断了
                # 回复。抛中断走既有 auto-continue 续写恢复（比上游的 502 更
                # 优：续写轮要求"继续"，模型从中断处接着产出）。
                if incomplete and not got_body:
                    logger.warning("[v77] FINISHED with status INCOMPLETE and no body -> continue")
                    raise _SseStreamInterrupted()
                if not got_output:
                    _raise_empty_sse_response(recent_lines, parsed_events)
                # [v57 移植 v39] FINISHED with thinking only, no body: upstream
                # closed the turn without ever producing an answer. Treat as
                # interrupted so the auto-continue path asks for the real reply.
                if not got_body:
                    logger.warning("[auto-continue] FINISHED but no body (thinking only) -> continue")
                    raise _SseStreamInterrupted()
                # [v58] 正文只剩未解析的工具调用标记时同样按中断处理：
                # 上游偶发输出畸形 DSML（有 </｜｜DSML｜｜ invoke> 闭合却没有
                # <｜｜DSML｜｜ invoke name=...> 开口）——2026-10-24 16:53:51
                # 线上实例（session 8e13b53e，WAL rowid 1541，5 个帧版本一致，
                # 1208 字符）：思考完整、FINISHED 正常，但 StreamToolParser 把
                # 整个 calls 块吞掉（parse_tools 解析不出工具），客户端只见
                # 思考、正文全空，用户被迫手动发「继续」。此处让既有自动续写
                # 接管，重新请求上游产出真正回复（续写轮仍带 thinking）。
                body_text = "".join(body_acc)
                if not _body_has_real_content(body_text):
                    logger.warning("[auto-continue] FINISHED but body is unparsed tool markup only -> continue")
                    # [v62] 分叉：因畸形工具调用触发时，把原始调用文本带进
                    # 异常，让续写轮用诊断提示词（模型知道错在哪才能重写），
                    # 而不是通用的"接着上文继续写"（模型只能掷硬币）。
                    if _has_unswallowed_tool_markup(body_text):
                        raise _SseStreamInterrupted(
                            "malformed_tool_call", _extract_malformed_call(body_text))
                    raise _SseStreamInterrupted("no_body", "")
                raise _SseStreamFinished()

            if "v" in data and isinstance(data["v"], dict) and "response" in data["v"]:
                fragments = data["v"]["response"].get("fragments")
                if fragments:
                    for fragment in fragments:
                        if fragment.get("type") == "THINK":
                            if not think_open:
                                yield "\n<think>"
                                think_open = True
                            got_output = True
                            yield fragment.get("content", "")
                        else:
                            if think_open:
                                yield "</think>"
                                think_open = False
                            got_output = True
                            got_body = True
                            _last_emit[0] = time.time()
                            body_acc.append(fragment.get("content", "") or "")
                            yield fragment.get("content", "")
                return

            if data.get("p") == "response/fragments" and data.get("o") == "APPEND":
                fragments = data.get("v")
                if isinstance(fragments, list):
                    for fragment in fragments:
                        if fragment.get("type") == "RESPONSE":
                            if think_open:
                                yield "</think>"
                                think_open = False
                            got_output = True
                            got_body = True
                            _last_emit[0] = time.time()
                            body_acc.append(fragment.get("content", "") or "")
                            yield fragment.get("content", "")
                        elif fragment.get("type") == "THINK":
                            if not think_open:
                                yield "\n<think>"
                                think_open = True
                            got_output = True
                            _last_emit[0] = time.time()
                            yield fragment.get("content", "")
                        else:
                            got_output = True
                            got_body = True
                            _last_emit[0] = time.time()
                            body_acc.append(fragment.get("content", "") or "")
                            yield fragment.get("content", "")
                return

            v = data.get("v")
            if isinstance(v, str) and v:
                got_output = True
                got_body = True
                _last_emit[0] = time.time()
                body_acc.append(v)
                yield v

        async def _consume_sse_line(decoded_line: str):
            if not decoded_line.startswith("data: "):
                return
            _remember_sse_line(recent_lines, decoded_line)
            try:
                data = json.loads(decoded_line[6:])
            except Exception:
                return
            if isinstance(data, dict):
                parsed_events.append(data)
                if len(parsed_events) > _SSE_RECENT_LINES:
                    del parsed_events[0]
            try:
                async for chunk in _emit_from_event(data):
                    yield chunk
            except _SseStreamFinished:
                raise

        # [v29] Upstream stall guard: an SSE stream that neither emits bytes
        # nor closes (thinking finished, answer pending, upstream hung) used
        # to block until the CLIENT timed out first — the user saw "thinking
        # done, no reply", and send_message's auto-continue never fired
        # because the stream never *ended*. Treat N seconds of total silence
        # as an interruption so the auto-continue path takes over.
        _aiter = resp.content.iter_any()
        _idle_timeout = float(os.getenv("DEEPSEEKER_STREAM_IDLE_TIMEOUT", "25"))
        _last_emit = [time.time()]
        # [v57 移植 v45] 首字节宽限：上游预填长上下文时一个字都不吐（实测
        # 120k 输入 18.7s、300k 输入 71s），而旧代码从发请求起就按 25s 空闲
        # 判定，长输入会在预填途中被误判挂起 -> 发续写 -> parent+2 -> 重新
        # 预填，反而更慢，且能造出"思考完没输出"。首字节到达前用独立预算，
        # 之后才启用空闲判定。（_first_byte_budget/_first_byte_deadline 已在
        # [v63] 提前至 POST 之前定义，与响应头阶段共用同一预算。）
        _got_first_byte = False
        while True:
            # [v30] Silence is measured on CONTENT EMISSION, not raw bytes:
            # the upstream may dribble keep-alive bytes/blank lines forever
            # while hanging, which would defeat a byte-level guard. remaining
            # budget shrinks until _emit_from_event actually yields content.
            if _got_first_byte:
                remaining = _idle_timeout - (time.time() - _last_emit[0])
            else:
                remaining = _first_byte_deadline - time.time()
            if remaining <= 0:
                # [v57 移植 v49] 静默一律按中断处理，不按 got_body 分流：
                # v45 曾在"正文已输出后静默"时改为正常收尾不续写，结果真实
                # 事故里回复断在半个代码块——收尾后客户端拿到残缺输出且不会
                # 再续写。FINISHED 才是"正常说完"的唯一凭证（同 v43 结论），
                # 没拿到就中断触发续写，由续写轮补全。
                logger.warning("[stream-stall] 上游流 %.0fs 无新内容（got_body=%s），按中断处理触发续写", _idle_timeout, got_body)
                if think_open:
                    yield "</think>"
                logger.warning("[stream-stall] 上游流 %.0fs 未产出任何正文（疑似挂起），主动中断以触发自动续写", _idle_timeout)
                raise _SseStreamInterrupted()
            try:
                chunk = await asyncio.wait_for(_aiter.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.warning("[stream-stall] 上游流 %.0fs 无新内容（got_body=%s），按中断处理触发续写", _idle_timeout, got_body)
                if think_open:
                    yield "</think>"
                logger.warning("[stream-stall] 上游流 %.0fs 未产出任何正文（疑似挂起），主动中断以触发自动续写", _idle_timeout)
                raise _SseStreamInterrupted()
            if not chunk:
                continue
            if not _got_first_byte:
                # 首字节到达才重置空闲计时：否则首字节耗时（长上下文预填
                # 可达 71s）会直接吃光空闲预算，第二圈就误判挂起。
                _got_first_byte = True
                _last_emit[0] = time.time()
            line_buf += chunk
            while b"\n" in line_buf:
                raw_line, line_buf = line_buf.split(b"\n", 1)
                decoded_line = raw_line.decode("utf-8", errors="replace").strip()
                if not decoded_line:
                    continue
                try:
                    async for out in _consume_sse_line(decoded_line):
                        yield out
                except _SseStreamFinished:
                    finished = True
                    return

        if line_buf.strip():
            decoded_line = line_buf.decode("utf-8", errors="replace").strip()
            if decoded_line:
                try:
                    async for out in _consume_sse_line(decoded_line):
                        yield out
                except _SseStreamFinished:
                    finished = True
                    return

        if think_open:
            yield "</think>"
        # [v77 PR#70 移植适配] 流尾同样检查 INCOMPLETE
        if incomplete and not got_body:
            logger.warning("[v77] stream ended with status INCOMPLETE and no body -> continue")
            raise _SseStreamInterrupted()
        if not got_output:
            _raise_empty_sse_response(recent_lines, parsed_events)
        if not finished:
            # [v57 移植 v43] FINISHED 是"正常说完"的唯一凭证。没收到就说明流被
            # 上游/网络掐断，哪怕已经出过正文也可能断在半句——按中断处理触发
            # 续写。v34 的 got_body 守卫在"说完后断流"上是对的，在"话说到
            # 一半被掐"上会静默截断输出（2026-10-23 现场：回复断在代码块中间）。
            logger.warning("[auto-continue] 未收到 FINISHED（got_body=%s），按中断处理触发续写", got_body)
            raise _SseStreamInterrupted()


_CONTINUE_PROMPT = (
    "接着上文继续写。不要重复已经写过的内容，不要任何开场白、解释或复述。"
    "如果上文已经完整写完了，只输出【END】，不要输出任何别的内容。"
)

# [v64] 零可见内容场景的续写提示词。与 _CONTINUE_PROMPT 的分工：
# 旧提示词的 【END】 逃逸口服务"已写出真实正文、只是被掐断"的场景——
# 模型说句 【END】 表示写完，过滤器砍掉避免重复追加。但当着整个回合
# 还没有发出过任何可见内容时（思考后上游直接结束回合），模型面对一个
# 空的助手回合会顺势只吐 【END】，被过滤器整段砍光 -> 客户端只收到
# 思考、finish_reason=stop，回合静默死亡（2026-10-25 01:04:11 线上实例
# session d4a7e123：517 字思考后零输出，前 12 轮均为工具调用轮）。
# 此场景明确告知"客户端什么都没收到"并禁用终止标记，要求直接产出。
_CONTINUE_PROMPT_EMPTY = (
    "【系统通知】你上一轮的回复只有思考过程，没有任何正文或工具调用，"
    "客户端什么都没有收到。请直接给出完整的回复内容，或重新发出工具调用。"
    "禁止输出【END】或其他终止标记，禁止复述思考内容，不要任何开场白、"
    "解释或自我评价。"
)

# [v62] 畸形工具调用场景的续写提示词。与 _CONTINUE_PROMPT 的分工：
# 通用提示词服务"断流/卡死/思考无输出"（模型不缺信息，只是被催续写）；
# 本提示词服务"工具调用格式错误被静默吞掉"（2026-10-24 12:27 线上实例
# session bb59b5d3：模型输出一句正文 + 缺 name 的原生调用，parser 吞块、
# parse_tools 解析不出、finish_reason=stop，会话死掉）。
# 措辞刻意不指死具体病因（畸形不止缺 name 一种：还有只有闭合、参数块破损），
# 统一说"格式错误"，附原始调用让模型自己定位，给格式示例降低再错率，
# 并明确两条出路防止它只会"写正文总结"。
_MALFORMED_TOOL_PROMPT = (
    "【系统通知】你上一轮发出的工具调用格式错误，服务端无法解析，已拒绝执行。"
    "被拒绝的原始调用如下：\n\n"
    "<rejected_call>\n{raw}\n</rejected_call>\n\n"
    "工具调用的正确格式为（name 与 arguments 两个字段都不可缺少）：\n"
    "<tool_call>{{\"name\": \"工具名\", \"arguments\": {{...}}}}</tool_call>\n\n"
    "请重新发出格式正确的工具调用；如果这个任务已不需要工具，"
    "直接给出最终答案，不要复述上文，不要解释这个错误。"
)


def _extract_malformed_call(text: str, limit: int = 400) -> str:
    """[v62] 从正文里提取畸形工具调用的原文（截断到 limit），供诊断提示词附载。

    策略：找到第一个工具标记（_TOOL_MARKUP_HINT_RE，代码围栏内的示例跳过），
    从该处起 600 字符内找最后一个 > 作为调用块结尾（覆盖原生跨行 JSON 与
    DSML 多行参数块）；取不到就退回标记后 300 字符。
    """
    if not text:
        return ""
    spans = _code_fence_spans(text)
    for m in _TOOL_MARKUP_HINT_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        tail = text[m.start():m.start() + 600]
        close = tail.rfind(">")
        raw = tail[:close + 1] if close >= 0 else tail[:300]
        return raw[:limit]
    return text[:limit]

_END_MARKERS = ("【END】", "[END]", "<END>")


class _ContinuationFilter:
    """续写轮的输出过滤器。

    第一轮原样透传，不做任何改写。续写轮把输出缓冲到 flush，做三件事：
    1. 砍掉终止标记之后的所有内容（模型用【END】表示"上文已完整"）
    2. 去掉前导空行/空白
    3. 与已发内容做尾部去重（模型常把最后一句原样重说一遍）

    去重要求重叠 >= 8 字符才削，避免"的"、"了"这类单字巧合造成误删。
    """

    def __init__(self, passthrough, tail=""):
        self.passthrough = passthrough
        self.tail = tail or ""
        self.buf = []
        self.done = False

    def feed(self, chunk):
        if not chunk:
            return []
        if self.passthrough:
            return [chunk]
        self.buf.append(chunk)
        return []

    def flush(self):
        if self.passthrough or self.done:
            return []
        self.done = True
        out = "".join(self.buf)
        self.buf = []
        if not out:
            return []
        cut = len(out)
        hit = False
        for m in _END_MARKERS:
            i = out.find(m)
            if 0 <= i < cut:
                cut = i
                hit = True
        out = out[:cut]
        if hit:
            out = out.rstrip()
        out = out.lstrip("\r\n \t")
        tail = self.tail[-400:]
        if tail and out:
            k = min(len(tail), len(out), 400)
            while k >= 8:
                if out.startswith(tail[-k:]):
                    out = out[k:]
                    break
                k -= 1
        return [out] if out else []


class _VisibleTracker:
    """[v64] 跨 chunk 判断客户端是否收到过"可见正文"（非思考内容）。

    喂入的必须是经过 _ContinuationFilter 之后真正 yield 给客户端的内容
    （续写轮被砍掉的 【END】 不会进来——客户端本就没看到它）。

    语义与 app._StreamTagSplitter 的 think_done 闭锁对齐：首个思考块
    闭合之后，后续内容一律算可见（splitter 也不再把它分流为 reasoning）；
    工具调用标记算可见——它对用户/下游是真实产出。

    前提契约：调用方保证 "<think>"/"</think>" 标签原子到达（_emit_from_event
    正是整串 yield；_ContinuationFilter 的 passthrough 与 flush 都不切分
    chunk），故无需 splitter 那样的标签前缀扣留逻辑。
    """

    __slots__ = ("in_think", "done", "seen")

    def __init__(self):
        self.in_think = False
        self.done = False
        self.seen = False

    def feed(self, s):
        if self.seen or not s:
            return
        i = 0
        n = len(s)
        while i < n:
            if self.done or not self.in_think:
                j = -1 if self.done else s.find("<think>", i)
                if j == -1:
                    if s[i:].strip():
                        self.seen = True
                    return
                if s[i:j].strip():
                    self.seen = True
                    return
                self.in_think = True
                i = j + 7
            else:
                j = s.find("</think>", i)
                if j < 0:
                    return
                self.in_think = False
                self.done = True
                i = j + 8


# [v77 PR#71 移植] DeepSeek 网页端每账号同时只能生成一条回复；第二个并发
# 发送会报 "A message is being generated" / "Messages too frequent"。改为
# 按账号排队发送。所有生成都走 send_message，因此覆盖全部调用方。
_send_locks = {}


def token_busy(auth_token):
    """[v77 PR#71 移植] 该账号正在生成（其发送锁被持有）时为 True。"""
    lock = _send_locks.get(auth_token)
    return bool(lock and lock.locked())


async def send_message(chat_id, auth_token, message, parent_message_id, thinking=False, search=False, file_ids_=None, max_continue=2):
    # [v77 PR#71 移植] 上游流由独立任务排空：即使消费方中途放弃这个
    # generator（客户端消失且未 aclose），锁也会在 DeepSeek 结束时释放——
    # 跨 yield 持锁会泄漏锁，把该账号之后的所有请求全部卡死。
    lock = _send_locks.setdefault(auth_token, asyncio.Lock())
    queue = asyncio.Queue()
    done = object()

    async def _pump():
        try:
            async with lock:
                gen = _send_message_locked(chat_id, auth_token, message, parent_message_id, thinking=thinking, search=search, file_ids_=file_ids_, max_continue=max_continue)
                try:
                    async for chunk in gen:
                        queue.put_nowait(chunk)
                finally:
                    await gen.aclose()
        except Exception as e:
            queue.put_nowait(e)
        finally:
            queue.put_nowait(done)

    task = asyncio.create_task(_pump())
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # 消费方关闭我们（断连、abort）时仍取消上游。
        task.cancel()


async def _send_message_locked(chat_id, auth_token, message, parent_message_id, thinking=False, search=False, file_ids_=None, max_continue=2):
    """带自动续写的对外入口。

    上游流在 FINISHED 之前异常中断（断流/断连/上游提前关闭）时，自动以
    『继续』在同会话内续写——parent 按 next_parent 语义（+2）链式前进，
    续写内容无缝拼进同一个客户端流。最多自动续写 max_continue 次。

    [v64] 两处加固（根因：2026-10-25 01:04 线上"思考完没输出"）：
    1. 提示词分流：本回合还没发出过任何可见正文时，续写不用带 【END】
       逃逸口的 _CONTINUE_PROMPT，换用明令禁止终止标记的
       _CONTINUE_PROMPT_EMPTY——否则模型对着空助手回合只吐 【END】，
       被过滤器砍光后客户端静默无所得。
    2. 收尾安全网：整个回合结束时可见正文仍为空（续写轮只吐终止标记），
       还有预算就换提示词再试，没预算给出可见提示而非静默 return。
    """
    parent = parent_message_id
    msg = message
    files = file_ids_
    emitted = ""
    # [v64] 客户端可见正文跟踪（跨 attempt 累积）；喂入的均为过滤后
    # 真正 yield 给客户端的内容。
    visible = _VisibleTracker()
    for attempt in range(max_continue + 1):
        flt = _ContinuationFilter(passthrough=(attempt == 0), tail=emitted)
        # [v57 移植 v50] 续写轮原样透传 thinking，不强制关闭：
        # v39/v40 曾实验 _thinking = thinking if attempt == 0 else False，
        # 实测第二轮上游会把新思考当正文吐出来（B3 思考泄漏），真终局
        # 已回退为该行。
        _thinking = thinking
        clean_end = False
        interrupt_reason = "interrupted"
        interrupt_detail = ""
        try:
            async for chunk in _send_message_once(chat_id, auth_token, msg, parent, _thinking, search, files):
                for out in flt.feed(chunk):
                    emitted = (emitted + out)[-400:]
                    visible.feed(out)
                    yield out
            for out in flt.flush():
                emitted = (emitted + out)[-400:]
                visible.feed(out)
                yield out
            clean_end = True  # 正常完成（上游发出 FINISHED）
        except _SseStreamFinished:
            for out in flt.flush():
                emitted = (emitted + out)[-400:]
                visible.feed(out)
                yield out
            clean_end = True
        except _SseStreamInterrupted as e:
            for out in flt.flush():
                emitted = (emitted + out)[-400:]
                visible.feed(out)
                yield out
            # except as 变量在块结束即解绑（PEP 3110），先取出再用
            interrupt_reason = getattr(e, "reason", "interrupted")
            interrupt_detail = getattr(e, "detail", "")
            pass
        if clean_end:
            asyncio.create_task(_prefetch_pow(auth_token, "/api/v0/chat/completion"))
            if visible.seen:
                return  # 正常完成（上游发出 FINISHED）
            # [v64] 收尾安全网：整个回合没有发出过任何可见正文——续写轮
            # 只吐了 【END】 被过滤器整段砍光（或反复只吐思考）。不能
            # 静默 return：客户端将只收到思考 + finish_reason=stop，回合
            # 静默死亡（01:04 故障形态）。还有预算就换提示词再试，没预算
            # 给出可见提示。
            if attempt >= max_continue:
                logger.warning("[auto-continue] 连续多轮无可见正文，给出可见提示收尾")
                yield chr(10) + chr(10) + (
                    "（上游连续多轮只返回思考过程、未产出任何正文，已停止自动重试。"
                    "请重新发送消息或稍后再试。）")
                return
            logger.warning("[auto-continue] 续写后仍无可见正文（疑似只吐终止标记），"
                           "换用禁 END 提示词重试 %d/%d", attempt + 1, max_continue)
            interrupt_reason = "empty_continuation"
        if attempt >= max_continue:
            logger.warning("[auto-continue] 自动续写已达上限(%d)，按现状收尾", max_continue)
            yield chr(10) + chr(10) + "（上游在思考后反复中断，已重试 %d 次仍未产出正文）" % max_continue
            return
        logger.warning("[auto-continue] 上游流未正常结束（reason=%s），自动续写第 %d/%d 次",
                       interrupt_reason, attempt + 1, max_continue)
        parent = parent + 2  # next_parent 语义：本次 exchange 上游产生 user+assistant 两条消息
        # [v62] 畸形工具调用场景换用诊断提示词：告诉模型错在哪、附原始调用、
        # 给正确格式、给两条出路（重写调用 or 直接作答）——模型知道病因才会
        # 重写正确调用，而不是对着通用的"继续写"掷硬币。其余场景行为不变。
        # [v64] 追加分流：本回合还没有任何可见正文时（visible.seen=False），
        # 旧提示词的 【END】 逃逸口会被模型当成"没啥可续"的出口——只吐
        # 【END】 被砍光 -> 静默死回合。换用禁终止标记的
        # _CONTINUE_PROMPT_EMPTY。已有可见正文时行为与 v62 完全一致。
        if interrupt_reason == "malformed_tool_call":
            msg = _MALFORMED_TOOL_PROMPT.format(raw=interrupt_detail[:400])
        elif interrupt_reason == "empty_continuation" or not visible.seen:
            msg = _CONTINUE_PROMPT_EMPTY
        else:
            msg = _CONTINUE_PROMPT
        files = []
        # [v61] 边界哨兵：让 app.stream_response 重置 splitter 的 think_done
        # 闭锁，使续写轮开头的 "\n<think>" 被重新识别为 reasoning 而不是
        # 漏进 content（见 _RetryBoundary 文档）。空串，对内容无贡献。
        yield _RetryBoundary()


async def upload_file(file_bytes, file_name, file_content_type, auth_token):
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    session = await get_session()
    url = "https://chat.deepseek.com/api/v0/file/upload_file"
    file_size = len(file_bytes)
    pow_response = await solve_create_pow("/api/v0/file/upload_file", auth_token)
    boundary = b"----WebKitFormBoundaryTB0pXOQR2RL219Hu"
    safe_name = re.sub(r"[^ -~]", "_", file_name).replace('"', "_") or "file.bin"
    body_parts = [
        b"--" + boundary + b"\r\n",
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'.encode("utf-8"),
        f"Content-Type: {file_content_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n--" + boundary + b"--\r\n",
    ]
    reconstructed_body = b"".join(body_parts)
    headers = get_headers(auth_token, pow_response)
    headers.update({
        "content-type": f"multipart/form-data; boundary={boundary.decode('utf-8')}",
        "x-file-size": str(file_size),
    })
    response = await post_with_failover(
        "/api/v0/file/upload_file",
        data=reconstructed_body, headers=headers,
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=120),
    )
    async with response:
        resp_json = await response.json()
    file_id = resp_json["data"]["biz_data"]["id"]
    yield ("uploaded", file_id)
    js_data = resp_json["data"]["biz_data"]
    status = js_data["status"]
    headers = get_headers(auth_token)
    deadline = time.time() + 300
    while status in ["PENDING", "PARSING"] and time.time() < deadline:
        yield ("uploaded", file_id)
        await asyncio.sleep(0.3)
        async with session.get(
            "https://chat.deepseek.com/api/v0/file/fetch_files?file_ids=" + file_id,
            headers=headers,
            # cookies=cookie,  # Backup WAF fallback
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            js_data = (await resp.json())["data"]["biz_data"]["files"][0]
        status = js_data["status"]
    if status == "SUCCESS":
        tp_data = datetime.fromtimestamp(js_data["updated_at"], timezone.utc)
        yield ("success", {
            "file_id": file_id,
            "openai_timestamp": int(js_data["updated_at"]),
            "size": js_data["file_size"],
            "anthropic_timestamp": tp_data.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    else:
        yield ("error", file_id)


async def get_file_content(auth_token, file_id):
    # Backup: WAF cookies not required with Android headers. Kept as fallback:
    # cookie = await get_cookies()
    session = await get_session()
    headers = get_headers(auth_token)
    async with session.get(
        "https://chat.deepseek.com/api/v0/file/fetch_files?file_ids=" + file_id,
        headers=headers,
        # cookies=cookie,  # Backup WAF fallback
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        resp_json = await resp.json()
    js_data = resp_json["data"]["biz_data"]["files"][0]
    yield mimetypes.guess_type(js_data["file_name"])[0]
    deadline = time.time() + 60
    while js_data.get("status") in ("PENDING", "PARSING") and time.time() < deadline and not js_data.get("signed_path"):
        await asyncio.sleep(0.5)
        async with session.get(
            "https://chat.deepseek.com/api/v0/file/fetch_files?file_ids=" + file_id,
            headers=headers,
            # cookies=cookie,  # Backup WAF fallback
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            js_data = (await resp.json())["data"]["biz_data"]["files"][0]
    if not js_data.get("signed_path"):
        return
    file_path = "https://files.deepseeksvc.com/api" + js_data["signed_path"] + "&ty=r"
    async with session.get(file_path) as data:
        async for chunk in data.content.iter_chunked(8192):
            if chunk:
                yield chunk
