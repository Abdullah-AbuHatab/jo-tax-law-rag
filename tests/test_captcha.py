"""Optional Cloudflare Turnstile on /ask: off by default, fail closed when on (siteverify mocked)."""

import json
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.captcha import SITEVERIFY_URL, TurnstileVerifier
from app.config import Settings
from app.llm import FakeLLM
from app.main import create_app
from tests.conftest import fake_settings

QUESTION = "ما هي نسبة الضريبة العامة على المبيعات؟"
SITE_KEY = "0x4AAAAAAAtestsitekey"
KEYS = {"turnstile_site_key": SITE_KEY, "turnstile_secret_key": "test-secret"}


class Siteverify:
    """A mocked https://challenges.cloudflare.com/turnstile/v0/siteverify."""

    def __init__(self, reply=None):
        self.requests: list[httpx.Request] = []
        self.reply = reply or (lambda token: httpx.Response(200, json={"success": token == "good-token"}))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        token = parse_qs(request.content.decode())["response"][0]
        return self.reply(token)


def build(laws, embedder, filled_store, siteverify=None, llm=None, **settings):
    verifier = None
    if siteverify is not None:
        verifier = TurnstileVerifier("test-secret", client=httpx.Client(transport=httpx.MockTransport(siteverify)))
    app = create_app(
        fake_settings(**settings),
        laws=laws,
        embedder=embedder,
        store=filled_store,
        llm=llm or FakeLLM(),
        captcha=verifier,
    )
    return TestClient(app)


def ask(client, token=None, **headers):
    if token is not None:
        headers["CF-Turnstile-Response"] = token
    return client.post("/ask", json={"question": QUESTION}, headers=headers)


def test_turnstile_is_off_by_default(laws, embedder, filled_store):
    client = build(laws, embedder, filled_store)
    assert client.get("/config").json()["turnstile_site_key"] is None
    assert ask(client).status_code == 200  # no token needed
    assert "challenges.cloudflare.com" not in client.get("/").headers["content-security-policy"]


def test_settings_need_both_turnstile_keys_or_neither():
    assert Settings.model_validate({}).turnstile_enabled is False
    assert Settings.from_env({"TURNSTILE_SITE_KEY": "a", "TURNSTILE_SECRET_KEY": "b"}).turnstile_enabled is True
    for partial in ({"TURNSTILE_SITE_KEY": "a"}, {"TURNSTILE_SECRET_KEY": "b"}):
        with pytest.raises(ValidationError):
            Settings.from_env(partial)


def test_turnstile_on_requires_a_valid_token(laws, embedder, filled_store):
    siteverify, llm = Siteverify(), FakeLLM()
    client = build(laws, embedder, filled_store, siteverify, llm=llm, **KEYS)
    assert client.get("/config").json()["turnstile_site_key"] == SITE_KEY

    missing = ask(client)
    assert missing.status_code == 403 and missing.json()["error"]["code"] == "CAPTCHA_REQUIRED"
    assert siteverify.requests == []  # nothing to verify: Cloudflare is not called
    bad = ask(client, "bad-token")
    assert bad.status_code == 403 and bad.json()["error"]["code"] == "CAPTCHA_FAILED"
    too_long = ask(client, "x" * 2049)
    assert too_long.status_code == 403 and too_long.json()["error"]["code"] == "CAPTCHA_FAILED"
    assert len(siteverify.requests) == 1  # the oversized token was refused locally
    assert llm.calls == []

    good = ask(client, "good-token")
    assert good.status_code == 200 and good.json()["refused"] is False
    request = siteverify.requests[-1]
    assert str(request.url) == SITEVERIFY_URL and request.method == "POST"
    assert parse_qs(request.content.decode()) == {"secret": ["test-secret"], "response": ["good-token"]}
    assert request.extensions["timeout"]["read"] == 5.0


@pytest.mark.parametrize(
    "reply",
    [
        lambda token: (_ for _ in ()).throw(httpx.ConnectError("down")),
        lambda token: httpx.Response(500, text="oops"),
        lambda token: httpx.Response(200, text="not json"),
        lambda token: httpx.Response(200, json={"success": "yes"}),  # only a real true passes
    ],
)
def test_turnstile_fails_closed(laws, embedder, filled_store, reply):
    client = build(laws, embedder, filled_store, Siteverify(reply), **KEYS)
    response = ask(client, "good-token")
    assert response.json()["error"]["code"] == "CAPTCHA_FAILED"
    assert response.status_code in (403, 503)


def test_captcha_failures_do_not_use_the_daily_count(laws, embedder, filled_store):
    client = build(laws, embedder, filled_store, Siteverify(), daily_ask_limit_per_client=1, **KEYS)
    assert ask(client).status_code == 403
    assert ask(client, "bad-token").status_code == 403
    assert ask(client, "good-token").status_code == 200


def test_turnstile_on_extends_only_the_page_csp(laws, embedder, filled_store):
    client = build(laws, embedder, filled_store, Siteverify(), **KEYS)
    page_csp = client.get("/").headers["content-security-policy"]
    assert page_csp == (
        "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; "
        "frame-src https://challenges.cloudflare.com; style-src 'self'; img-src 'self'; object-src 'none'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    assert client.get("/laws").headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert json.loads(client.get("/openapi.json").text)["paths"]["/config"]


def test_ui_renders_turnstile_only_when_the_server_has_a_site_key():
    from app.main import STATIC_DIR

    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'fetchJson("/config")' in script
    assert "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit" in script
    assert '"CF-Turnstile-Response"' in script and "turnstile.reset(" in script
    assert '<div id="captcha" class="captcha" hidden></div>' in page
