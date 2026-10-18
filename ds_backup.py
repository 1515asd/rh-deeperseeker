"""
ds_backup.py — deeperseeker tokens 表备份/恢复模块
===================================================

背景
----
Belmo 免费档不承诺容器文件系统持久化（pricing 页免费档无磁盘配额；
重新部署 = 新容器，应用运行时写入的文件不保证存活）。deeperseeker 的
SQLite 中唯一不可重建的资产是 `tokens` 表（DeepSeek 令牌纯手动录入，
无自动抓取逻辑）；`sessions` / `session_map` / `files` / `chat_memory`
丢失只会造成客户端重连重建，无长期损失。

本模块只备份/恢复 tokens 表 + 关键配置，用 PBKDF2 派生 keystream 做
流加密后推送到外部目标，启动时若 tokens 表为空则自动恢复。

零第三方依赖（仅标准库）。

配置（环境变量）
----------------
DEEPSEEKER_BACKUP_SECRET   必填。备份加密密码（建议 24+ 字符随机串）。
                           未设置时本模块完全静默禁用，不影响主程序。
DEEPSEEKER_BACKUP_TARGET   telegram | gist | file | none（默认 none）
DEEPSEEKER_BACKUP_INTERVAL 后台备份间隔秒数（默认 21600 = 6 小时）
DEEPSEEKER_BACKUP_TG_BOT_TOKEN / DEEPSEEKER_BACKUP_TG_CHAT_ID
                           telegram 目标：Bot Token + 接收备份的 chat_id
DEEPSEEKER_BACKUP_GIST_TOKEN / DEEPSEEKER_BACKUP_GIST_ID
                           gist 目标：GitHub token + 既有私有 Gist 的 ID
                           （Gist 需先手动创建，之后本模块只做 update）
DEEPSEEKER_BACKUP_FILE_PATH
                           file 目标：备份文件本地路径

集成方式（app.py 顶部 __main__ 之前各加一行即可）
------------------------------------------------
    import ds_backup
    ds_backup.restore_if_needed()          # 启动恢复：tokens 空则拉备份
    asyncio.create_task(ds_backup.background_loop())  # 常驻定时备份

恢复逻辑
--------
只有 tokens 表为空（全新/丢库容器）时才触发恢复；表非空时绝不动本地
数据（本地优先，备份只作灾备，不会覆盖活跃数据）。
"""

import asyncio
import base64
import hashlib
import json
import os
import sqlite3
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

_SECRET = os.getenv("DEEPSEEKER_BACKUP_SECRET", "").strip()
_TARGET = os.getenv("DEEPSEEKER_BACKUP_TARGET", "none").strip().lower()
_INTERVAL = max(300, int(os.getenv("DEEPSEEKER_BACKUP_INTERVAL", "21600")))
_TG_TOKEN = os.getenv("DEEPSEEKER_BACKUP_TG_BOT_TOKEN", "").strip()
_TG_CHAT = os.getenv("DEEPSEEKER_BACKUP_TG_CHAT_ID", "").strip()
_GIST_TOKEN = os.getenv("DEEPSEEKER_BACKUP_GIST_TOKEN", "").strip()
_GIST_ID = os.getenv("DEEPSEEKER_BACKUP_GIST_ID", "").strip()
_FILE_PATH = os.getenv("DEEPSEEKER_BACKUP_FILE_PATH", "").strip()

_DB = os.getenv("DB_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "deeperseeker.db"
)
_SNAPSHOT_NAME = "deeperseeker-tokens-snapshot"

_TG_API = "https://api.telegram.org"
_GIST_API = "https://api.github.com/gists"


# ---------------------------------------------------------------------------
# 加密 / 解密（PBKDF2-SHA256 keystream XOR，零依赖流加密）
# ---------------------------------------------------------------------------

def _keystream(secret: str, salt: bytes, length: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < length:
        out += hashlib.pbkdf2_hmac(
            "sha256",
            secret.encode("utf-8"),
            salt + counter.to_bytes(4, "big"),
            100_000,
            dklen=32,
        )
        counter += 1
    return out[:length]


def encrypt(plaintext: bytes, secret: str = None) -> str:
    """加密为 base64(salt + ciphertext)。"""
    secret = secret or _SECRET
    salt = os.urandom(16)
    ks = _keystream(secret, salt, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, ks))
    return base64.b64encode(salt + ct).decode("ascii")


def decrypt(payload: str, secret: str = None) -> bytes:
    """解密 encrypt() 的输出。密码错误时数据将是乱码（由 JSON 校验兜底）。"""
    secret = secret or _SECRET
    raw = base64.b64decode(payload)
    salt, ct = raw[:16], raw[16:]
    ks = _keystream(secret, salt, len(ct))
    return bytes(a ^ b for a, b in zip(ct, ks))


# ---------------------------------------------------------------------------
# 快照导出 / 恢复（只动 tokens 表）
# ---------------------------------------------------------------------------

def _db_connect():
    conn = sqlite3.connect(_DB, timeout=10)
    return conn


def token_count() -> int:
    try:
        conn = _db_connect()
        n = conn.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return -1  # 表不存在或库不可读


def export_snapshot() -> dict:
    """导出 tokens 表 + 元信息。"""
    conn = _db_connect()
    rows = conn.execute(
        "SELECT id, alias, token, status, rate_limited_until, last_used FROM tokens"
    ).fetchall()
    conn.close()
    return {
        "v": 1,
        "created_at": int(time.time()),
        "table": "tokens",
        "tokens": [
            {
                "id": r[0],
                "alias": r[1],
                "token": r[2],
                "status": r[3],
                "rate_limited_until": r[4],
                "last_used": r[5],
            }
            for r in rows
        ],
    }


def restore_snapshot(snapshot: dict) -> int:
    """把快照写回 tokens 表（假设表已存在；不存在则先建表）。返回恢复行数。"""
    conn = _db_connect()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT,
            token TEXT,
            status TEXT DEFAULT 'ACTIVE',
            rate_limited_until REAL,
            last_used REAL
        )"""
    )
    n = 0
    for t in snapshot.get("tokens", []):
        conn.execute(
            "INSERT OR REPLACE INTO tokens (id, alias, token, status, rate_limited_until, last_used)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                t.get("id"),
                t.get("alias"),
                t.get("token"),
                t.get("status") or "ACTIVE",
                t.get("rate_limited_until"),
                t.get("last_used"),
            ),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def restore_if_needed() -> str:
    """tokens 表为空时自动从备份恢复。返回动作描述（用于日志）。"""
    if not _SECRET or _TARGET in ("", "none"):
        return "backup disabled"
    n = token_count()
    if n != 0:
        return f"local tokens={n}, skip restore"
    try:
        payload = _download()
        snapshot = json.loads(decrypt(payload))
        if snapshot.get("table") != "tokens":
            return "snapshot table mismatch, abort"
        restored = restore_snapshot(snapshot)
        return f"restored {restored} tokens from {_TARGET} snapshot (age {int(time.time()) - snapshot.get('created_at', 0)}s)"
    except Exception as e:
        return f"restore failed: {e}"


# ---------------------------------------------------------------------------
# 传输后端
# ---------------------------------------------------------------------------

def _post_json(url: str, body: dict, headers: dict = None, timeout: int = 30):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, headers: dict = None, timeout: int = 30):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _upload_tg(payload: str) -> str:
    """通过 Telegram Bot 的 sendDocument 上传（文本当 .enc 文件）。"""
    boundary = "----dsb" + hashlib.md5(str(time.time()).encode()).hexdigest()
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{_TG_CHAT}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"{_SNAPSHOT_NAME}.enc\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(payload.encode("utf-8"))
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{_TG_API}/bot{_TG_TOKEN}/sendDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"telegram send failed: {data}")
    return f"telegram message_id={data['result']['message_id']}"


def _download_tg() -> str:
    """从 chat 历史里找最新的 snapshot 文件并下载。"""
    updates = _get_json(
        f"{_TG_API}/bot{_TG_TOKEN}/getUpdates?limit=100",
        headers={},
    )
    best = None  # (date, file_id)
    for upd in updates.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        doc = msg.get("document") or {}
        fname = doc.get("file_name") or ""
        if doc.get("file_id") and _SNAPSHOT_NAME in fname:
            d = msg.get("date", 0)
            if best is None or d > best[0]:
                best = (d, doc["file_id"])
    if not best:
        raise RuntimeError("no snapshot document found in telegram chat")
    file = _get_json(f"{_TG_API}/bot{_TG_TOKEN}/getFile?file_id={best[1]}")
    fpath = file["result"]["file_path"]
    with urllib.request.urlopen(
        f"{_TG_API}/file/bot{_TG_TOKEN}/{fpath}", timeout=60
    ) as resp:
        return resp.read().decode("utf-8")


def _upload_gist(payload: str) -> str:
    """更新既有私有 Gist 的单个文件（需 GIST_ID；未配置则创建）。"""
    body = {"files": {f"{_SNAPSHOT_NAME}.enc": {"content": payload}}}
    if _GIST_ID:
        req = urllib.request.Request(
            f"{_GIST_API}/{_GIST_ID}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {_GIST_TOKEN}",
                "Content-Type": "application/json",
            },
            method="PATCH",
        )
    else:
        req = urllib.request.Request(
            _GIST_API,
            data=json.dumps({**body, "public": False}).encode("utf-8"),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {_GIST_TOKEN}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    gid = data.get("id", "")
    return f"gist id={gid}"


def _download_gist() -> str:
    gist = _get_json(
        f"{_GIST_API}/{_GIST_ID}",
        headers={"Authorization": f"Bearer {_GIST_TOKEN}"},
    )
    f = gist.get("files", {}).get(f"{_SNAPSHOT_NAME}.enc")
    if not f:
        raise RuntimeError("snapshot file not found in gist")
    if f.get("truncated"):
        with urllib.request.urlopen(f["raw_url"], timeout=60) as resp:
            return resp.read().decode("utf-8")
    return f.get("content", "")


def _upload_file(payload: str) -> str:
    with open(_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(payload)
    return f"file {_FILE_PATH}"


def _download_file() -> str:
    with open(_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


_BACKENDS = {
    "telegram": (_upload_tg, _download_tg),
    "gist": (_upload_gist, _download_gist),
    "file": (_upload_file, _download_file),
}


def _upload(payload: str) -> str:
    up, _ = _BACKENDS[_TARGET]
    return up(payload)


def _download() -> str:
    _, down = _BACKENDS[_TARGET]
    return down()


# ---------------------------------------------------------------------------
# 对外主流程
# ---------------------------------------------------------------------------

def backup_now() -> str:
    """立即执行一次备份。返回描述（成功/禁用/失败原因）。"""
    if not _SECRET or _TARGET not in _BACKENDS:
        return "backup disabled"
    n = token_count()
    if n <= 0:
        return "no tokens to back up"
    snapshot = export_snapshot()
    payload = encrypt(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))
    try:
        info = _upload(payload)
        return f"backup ok: {n} tokens -> {info}"
    except Exception as e:
        return f"backup failed: {e}"


async def background_loop():
    """常规定时备份协程（由 app.py 用 create_task 启动）。"""
    while True:
        await asyncio.sleep(_INTERVAL)
        try:
            msg = backup_now()
        except Exception as e:
            msg = f"backup loop error: {e}"
        print(f"[ds_backup] {msg}", flush=True)


if __name__ == "__main__":
    print(restore_if_needed())
    print(backup_now())
