"""Optional Supabase-backed authentication and credit accounting.

The API works without any of this: run it locally and every endpoint is open.
Set ``KICKS_SUPABASE_URL`` and ``KICKS_SUPABASE_SERVICE_KEY`` and two things
change — the synthesis endpoints start requiring a Supabase session token, and
``POST /export`` charges the caller one credit through the database's
``charge_export`` function before it renders.

Tokens are verified locally (no round trip per request). Supabase projects
created since 2025 sign with an asymmetric key published at
``/auth/v1/.well-known/jwks.json``; older projects use a shared HS256 secret,
which can be supplied as ``KICKS_SUPABASE_JWT_SECRET``. Credits are never
touched with the user's own token: only the service role may call the ledger
functions, and the service key never leaves this process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from fastapi import HTTPException, Request

#: How strict the API is about a session token.
#:   required  — every synthesis endpoint needs a valid token (default when configured)
#:   optional  — tokens are verified when present; anonymous calls are allowed
#:   off       — no verification; ``/export`` is unavailable
AuthMode = str


@dataclass(frozen=True)
class AuthSettings:
    url: str | None
    service_key: str | None
    jwt_secret: str | None
    mode: AuthMode

    @classmethod
    def from_env(cls) -> "AuthSettings":
        url = os.environ.get("KICKS_SUPABASE_URL") or None
        service_key = os.environ.get("KICKS_SUPABASE_SERVICE_KEY") or None
        mode = os.environ.get("KICKS_AUTH_MODE") or ("required" if url else "off")
        if mode not in ("required", "optional", "off"):
            raise ValueError(f"KICKS_AUTH_MODE must be required, optional or off, not {mode!r}")
        if mode != "off" and not url:
            raise ValueError("KICKS_AUTH_MODE is set but KICKS_SUPABASE_URL is not")
        return cls(
            url=url.rstrip("/") if url else None,
            service_key=service_key,
            jwt_secret=os.environ.get("KICKS_SUPABASE_JWT_SECRET") or None,
            mode=mode,
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def billing_enabled(self) -> bool:
        """Credits can only be charged with a service key."""
        return self.enabled and bool(self.service_key)


@dataclass(frozen=True)
class User:
    id: str
    email: str | None
    role: str


class TokenVerifier:
    """Verifies Supabase access tokens issued for this project."""

    def __init__(self, settings: AuthSettings):
        self._settings = settings
        self._jwks: jwt.PyJWKClient | None = None
        if settings.url and not settings.jwt_secret:
            self._jwks = jwt.PyJWKClient(
                f"{settings.url}/auth/v1/.well-known/jwks.json",
                cache_keys=True, lifespan=3600,
            )

    def verify(self, token: str) -> User:
        try:
            if self._settings.jwt_secret:
                claims = jwt.decode(
                    token, self._settings.jwt_secret,
                    algorithms=["HS256"], audience="authenticated",
                )
            else:
                assert self._jwks is not None
                key = self._jwks.get_signing_key_from_jwt(token)
                claims = jwt.decode(
                    token, key.key,
                    algorithms=["ES256", "RS256"], audience="authenticated",
                )
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid session token: {exc}") from exc
        sub = claims.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Session token has no subject")
        return User(id=sub, email=claims.get("email"), role=claims.get("role", "authenticated"))


class InsufficientCredits(Exception):
    pass


class CreditsClient:
    """Calls the ledger functions in Postgres through PostgREST as the service role."""

    def __init__(self, settings: AuthSettings):
        if not settings.billing_enabled:
            raise RuntimeError("credits need KICKS_SUPABASE_URL and KICKS_SUPABASE_SERVICE_KEY")
        assert settings.url and settings.service_key
        self._client = httpx.AsyncClient(
            base_url=f"{settings.url}/rest/v1",
            headers={
                "apikey": settings.service_key,
                "Authorization": f"Bearer {settings.service_key}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )

    async def _rpc(self, fn: str, **params: Any) -> Any:
        response = await self._client.post(f"/rpc/{fn}", json=params)
        if response.status_code >= 400:
            detail = response.text
            try:
                detail = response.json().get("message", detail)
            except ValueError:
                pass
            if "insufficient_credits" in detail:
                raise InsufficientCredits(detail)
            raise HTTPException(status_code=502, detail=f"credits backend: {detail}")
        return response.json()

    async def balance(self, user_id: str) -> int:
        return int(await self._rpc("credit_balance", p_user_id=user_id))

    async def charge_export(
        self, user_id: str, export_id: str, instrument: str, params: dict[str, Any],
    ) -> tuple[int, str]:
        """Spend one credit for an export. Idempotent on ``export_id``.

        Returns ``(remaining_balance, generation_id)``.
        """
        rows = await self._rpc(
            "charge_export",
            p_user_id=user_id, p_export_id=export_id,
            p_instrument=instrument, p_params=params,
        )
        row = rows[0] if isinstance(rows, list) else rows
        return int(row["balance"]), str(row["generation_id"])

    async def refund_export(self, export_id: str) -> None:
        """Give the credit back when rendering failed after it was charged."""
        await self._rpc("refund_export", p_export_id=export_id)

    async def aclose(self) -> None:
        await self._client.aclose()


class Auth:
    """Request-level entry point: resolves the caller for the current mode."""

    def __init__(self, settings: AuthSettings | None = None):
        self.settings = settings or AuthSettings.from_env()
        self._verifier = TokenVerifier(self.settings) if self.settings.enabled else None
        self.credits = CreditsClient(self.settings) if self.settings.billing_enabled else None

    def _token(self, request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        return token.strip() if scheme.lower() == "bearer" and token else None

    def current_user(self, request: Request) -> User | None:
        """FastAPI dependency: the caller, or ``None`` when anonymous is allowed."""
        if not self._verifier:
            return None
        token = self._token(request)
        if token is None:
            if self.settings.mode == "required":
                raise HTTPException(status_code=401, detail="Sign in to use the synthesizer")
            return None
        return self._verifier.verify(token)

    def require_user(self, request: Request) -> User:
        """FastAPI dependency: a signed-in caller, regardless of mode."""
        if not self._verifier:
            raise HTTPException(status_code=503, detail="Accounts are not enabled on this server")
        token = self._token(request)
        if token is None:
            raise HTTPException(status_code=401, detail="Sign in to continue")
        return self._verifier.verify(token)

    def describe(self) -> dict[str, Any]:
        return {"mode": self.settings.mode, "billing": self.settings.billing_enabled}
