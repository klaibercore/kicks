"""Token verification and mode handling for the optional Supabase auth layer."""

import json
import time

import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from kicks.api.auth import Auth, AuthSettings, TokenVerifier

SECRET = "test-secret-with-at-least-32-characters!!"


def _request(token: str | None = None) -> Request:
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    return Request({"type": "http", "headers": headers, "method": "GET", "path": "/", "query_string": b""})


def _token(**overrides) -> str:
    claims = {
        "sub": "0d7d6a2b-2a6f-4bbd-9f2e-3a4e1d0e5c11",
        "email": "someone@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 60,
    }
    claims.update(overrides)
    return jwt.encode(claims, SECRET, algorithm="HS256")


def _settings(mode: str = "required", service_key: str | None = None) -> AuthSettings:
    return AuthSettings(url="https://example.supabase.co", service_key=service_key, jwt_secret=SECRET, mode=mode)


def test_hs256_token_is_accepted():
    user = TokenVerifier(_settings()).verify(_token())
    assert user.id == "0d7d6a2b-2a6f-4bbd-9f2e-3a4e1d0e5c11"
    assert user.email == "someone@example.com"
    assert user.role == "authenticated"


@pytest.mark.parametrize(
    "bad",
    [
        _token(exp=int(time.time()) - 10),          # expired
        _token(aud="anon"),                         # wrong audience
        jwt.encode({"sub": "x", "aud": "authenticated"}, "other-secret", algorithm="HS256"),
        "not-a-jwt",
    ],
)
def test_bad_tokens_are_401(bad):
    with pytest.raises(HTTPException) as exc:
        TokenVerifier(_settings()).verify(bad)
    assert exc.value.status_code == 401


def test_required_mode_rejects_anonymous():
    auth = Auth(_settings("required"))
    with pytest.raises(HTTPException) as exc:
        auth.current_user(_request())
    assert exc.value.status_code == 401
    assert auth.current_user(_request(_token())).id.startswith("0d7d6a2b")


def test_optional_mode_allows_anonymous_but_still_verifies():
    auth = Auth(_settings("optional"))
    assert auth.current_user(_request()) is None
    with pytest.raises(HTTPException):
        auth.current_user(_request("garbage"))


def test_off_mode_has_no_accounts():
    auth = Auth(AuthSettings(url=None, service_key=None, jwt_secret=None, mode="off"))
    assert auth.current_user(_request()) is None
    assert auth.credits is None
    with pytest.raises(HTTPException) as exc:
        auth.require_user(_request())
    assert exc.value.status_code == 503


def test_billing_needs_service_key():
    assert Auth(_settings()).credits is None
    assert Auth(_settings(service_key="service")).credits is not None


def test_settings_from_env(monkeypatch):
    monkeypatch.delenv("KICKS_SUPABASE_URL", raising=False)
    monkeypatch.delenv("KICKS_AUTH_MODE", raising=False)
    assert AuthSettings.from_env().mode == "off"

    monkeypatch.setenv("KICKS_SUPABASE_URL", "https://x.supabase.co/")
    s = AuthSettings.from_env()
    assert s.mode == "required" and s.url == "https://x.supabase.co"

    monkeypatch.setenv("KICKS_AUTH_MODE", "sometimes")
    with pytest.raises(ValueError):
        AuthSettings.from_env()


# -- credits client ----------------------------------------------------------

import httpx  # noqa: E402

from kicks.api.auth import CreditsClient, InsufficientCredits  # noqa: E402


def _credits_client(handler) -> CreditsClient:
    client = CreditsClient(_settings(service_key="service-key"))
    client._client = httpx.AsyncClient(  # swap the transport; same base_url and headers
        base_url="https://example.supabase.co/rest/v1",
        headers=client._client.headers,
        transport=httpx.MockTransport(handler),
    )
    return client


@pytest.mark.anyio
async def test_charge_export_uses_service_role_and_parses_rows():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json=[{"balance": 4, "generation_id": "gen-1"}])

    balance, generation_id = await _credits_client(handler).charge_export(
        "user-1", "export-1", "kick", {"sliders": {"sub": 0.5}},
    )
    assert (balance, generation_id) == (4, "gen-1")
    assert seen["url"].endswith("/rpc/charge_export")
    assert seen["auth"] == "Bearer service-key"
    assert json.loads(seen["body"])["p_export_id"] == "export-1"


@pytest.mark.anyio
async def test_insufficient_credits_is_its_own_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "P0001", "message": "insufficient_credits"})

    with pytest.raises(InsufficientCredits):
        await _credits_client(handler).charge_export("user-1", "export-1", "kick", {})


@pytest.mark.anyio
async def test_other_backend_errors_become_502():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(HTTPException) as exc:
        await _credits_client(handler).balance("user-1")
    assert exc.value.status_code == 502
