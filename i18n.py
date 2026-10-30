"""Minimal i18n for the DeeperSeeker UI.

Locale resolution order (all pre-login, no auth required):
  1. ``?lang=<code>`` query parameter        — explicit per-request override
  2. ``lang`` cookie                          — set by /set-lang/{code}
  3. ``Accept-Language`` header               — automatic browser-locale detect
  4. ``DEEPSEEKER_DEFAULT_LANG`` env or "en"  — final fallback

Templates live in per-locale directories: templates/{en,ru,zh}/ — each an
independent copy so translators edit whole pages, not key fragments.
"""

import os
from typing import Dict

from fastapi import Request
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUPPORTED_LANGS = ("en", "ru", "zh")
DEFAULT_LANG = os.getenv("DEEPSEEKER_DEFAULT_LANG", "en")
if DEFAULT_LANG not in SUPPORTED_LANGS:
    DEFAULT_LANG = "en"

COOKIE_NAME = "lang"
QUERY_NAME = "lang"

_templates: Dict[str, Jinja2Templates] = {
    lang: Jinja2Templates(directory=os.path.join(BASE_DIR, "templates", lang))
    for lang in SUPPORTED_LANGS
}

# Server-side strings that Jinja templates cannot carry (route logic in app.py).
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "too_many_attempts": {
        "en": "Too many attempts. Try again later.",
        "ru": "Слишком много попыток. Попробуйте позже.",
        "zh": "尝试次数过多，请稍后再试。",
    },
    "invalid_credentials": {
        "en": "Invalid username or password",
        "ru": "Неверное имя пользователя или пароль",
        "zh": "用户名或密码错误",
    },
}


def resolve_locale(request: Request) -> str:
    """Best-language for this request: query param → cookie → Accept-Language → default."""
    q = request.query_params.get(QUERY_NAME)
    if q in SUPPORTED_LANGS:
        return q
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie in SUPPORTED_LANGS:
        return cookie
    accept = request.headers.get("accept-language", "")
    for part in accept.split(","):
        # "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7" → take the primary subtag.
        code = part.split(";", 1)[0].strip().lower().split("-", 1)[0]
        if code in SUPPORTED_LANGS:
            return code
    return DEFAULT_LANG


def msg(key: str, lang: str) -> str:
    """Server-side message by key. Falls back through lang → default → key."""
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang) or entry.get(DEFAULT_LANG) or key


def page(request: Request, template: str, context: dict, lang: str = None):
    """Render a UI template in the request's locale, injecting `lang`."""
    lang = lang or resolve_locale(request)
    context = dict(context)
    context.setdefault("lang", lang)
    context.setdefault("supported_langs", SUPPORTED_LANGS)
    return _templates[lang].TemplateResponse(request, template, context)
