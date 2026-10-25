"""Optional Cloudflare Turnstile check for /ask (off unless both keys are set).

Server-side validation contract (https://developers.cloudflare.com/turnstile/get-started/server-side-validation/):
POST form fields ``secret`` and ``response`` to the siteverify URL; the JSON reply
has ``success: true`` for a valid, unused token. Tokens are single use and at most
2048 characters. Any failure to get a clear yes is a no (fail closed).
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)
SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
MAX_TOKEN_CHARS = 2048


class CaptchaUnavailable(Exception):
    """Siteverify could not be reached or gave no usable answer."""


class TurnstileVerifier:
    def __init__(self, secret_key: str, client: httpx.Client | None = None, timeout_seconds: float = 5.0) -> None:
        if not secret_key:
            raise ValueError("TURNSTILE_SECRET_KEY is not set")
        self._secret = secret_key
        self._client = client or httpx.Client()
        self._timeout = timeout_seconds

    def verify(self, token: str) -> bool:
        """True only for a token Cloudflare confirms; raises CaptchaUnavailable when it cannot tell."""
        if not token or len(token) > MAX_TOKEN_CHARS:
            return False
        try:
            response = self._client.post(
                SITEVERIFY_URL, data={"secret": self._secret, "response": token}, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            logger.warning("turnstile siteverify failed", extra={"error_type": type(exc).__name__})
            raise CaptchaUnavailable from exc
        if response.status_code != 200:
            logger.warning("turnstile siteverify returned an error", extra={"status": response.status_code})
            raise CaptchaUnavailable
        try:
            payload = response.json()
        except ValueError as exc:
            raise CaptchaUnavailable from exc
        return isinstance(payload, dict) and payload.get("success") is True
