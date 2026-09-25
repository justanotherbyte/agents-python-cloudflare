from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit

from .storage import MCPKeyValueStorage
from .types import MCPAuthorizationRequired

OAUTH_STATE_TTL_MS = 10 * 60 * 1_000


@dataclass(frozen=True, slots=True)
class _ParsedState:
    nonce: str
    server_id: str


class DurableOAuthProvider:
    """Durable OAuth state and credential slots scoped to one MCP server."""

    def __init__(
        self,
        storage: MCPKeyValueStorage,
        client_name: str,
        redirect_url: str,
        *,
        server_id: str,
        client_id: str | None = None,
        clock_ms: Callable[[], int] | None = None,
    ):
        self.storage = storage
        self.client_name = client_name
        self.redirect_url = redirect_url
        self.server_id = server_id
        self.client_id = client_id
        self.auth_url: str | None = None
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    @property
    def client_metadata(self) -> dict[str, Any]:
        hostname = urlsplit(self.redirect_url).hostname
        local = (
            hostname == "localhost"
            or hostname == "::1"
            or bool(hostname and hostname.startswith("127."))
        )
        return {
            "application_type": "native" if local else "web",
            "client_name": self.client_name,
            "client_uri": _origin(self.redirect_url),
            "grant_types": ["authorization_code", "refresh_token"],
            "redirect_uris": [self.redirect_url],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }

    async def state(self) -> str:
        nonce = secrets.token_urlsafe(16)
        value = f"{nonce}.{self.server_id}"
        await self.storage.put(
            self._state_key(nonce),
            {
                "nonce": nonce,
                "serverId": self.server_id,
                "createdAt": self._clock_ms(),
            },
        )
        return value

    async def check_state(self, state: str) -> tuple[bool, str | None]:
        parsed = _parse_state(state)
        if parsed is None:
            return False, "Invalid state format"
        stored = await self.storage.get(self._state_key(parsed.nonce))
        if not isinstance(stored, Mapping):
            return False, "State not found or already used"
        if stored.get("serverId") != parsed.server_id:
            await self.storage.delete(self._state_key(parsed.nonce))
            return False, "State serverId mismatch"
        created_at = stored.get("createdAt")
        if type(created_at) is not int or self._expired(created_at):
            await self.storage.delete(self._state_key(parsed.nonce))
            return False, "State expired"
        return True, None

    async def consume_state(self, state: str) -> None:
        parsed = _parse_state(state)
        if parsed is not None:
            await self.storage.delete(self._state_key(parsed.nonce))

    async def save_client_information(self, value: Mapping[str, Any]) -> None:
        client_id = value.get("client_id")
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("OAuth client information requires client_id")
        self.client_id = client_id
        await self.storage.put(self._client_info_key(client_id), dict(value))

    async def client_information(self) -> Mapping[str, Any] | None:
        if self.client_id is None:
            return None
        value = await self.storage.get(self._client_info_key(self.client_id))
        return value if isinstance(value, Mapping) else None

    async def save_tokens(self, value: Mapping[str, Any]) -> None:
        if self.client_id is None:
            raise ValueError("OAuth tokens require client_id")
        await self.storage.put(self._token_key(self.client_id), dict(value))
        await self.storage.delete(self._discovery_key())

    async def tokens(self) -> Mapping[str, Any] | None:
        if self.client_id is None:
            return None
        value = await self.storage.get(self._token_key(self.client_id))
        return value if isinstance(value, Mapping) else None

    async def save_discovery_state(self, value: Mapping[str, Any]) -> None:
        await self.storage.put(self._discovery_key(), dict(value))

    async def discovery_state(self) -> Mapping[str, Any] | None:
        value = await self.storage.get(self._discovery_key())
        return value if isinstance(value, Mapping) else None

    async def save_code_verifier(self, state: str, verifier: str) -> None:
        parsed = _parse_state(state)
        if parsed is None or parsed.server_id != self.server_id:
            raise ValueError("invalid OAuth state for code verifier")
        await self.storage.put(
            self._verifier_key(parsed.nonce),
            {"verifier": verifier, "createdAt": self._clock_ms()},
        )

    async def code_verifier(self, state: str) -> str:
        parsed = _parse_state(state)
        if parsed is None:
            raise ValueError("Invalid state format")
        value = await self.storage.get(self._verifier_key(parsed.nonce))
        if not isinstance(value, Mapping):
            raise ValueError("No code verifier found for OAuth state")
        created_at = value.get("createdAt")
        verifier = value.get("verifier")
        if (
            type(created_at) is not int
            or self._expired(created_at)
            or not isinstance(verifier, str)
        ):
            await self.storage.delete(self._verifier_key(parsed.nonce))
            raise ValueError("Code verifier expired")
        return verifier

    async def delete_code_verifier(self, state: str) -> None:
        parsed = _parse_state(state)
        if parsed is not None:
            await self.storage.delete(self._verifier_key(parsed.nonce))

    async def invalidate_credentials(self, scope: str = "all") -> None:
        keys: list[str] = []
        if scope in ("all", "discovery"):
            keys.append(self._discovery_key())
        if self.client_id is not None:
            if scope in ("all", "client"):
                keys.append(self._client_info_key(self.client_id))
            if scope in ("all", "tokens"):
                keys.append(self._token_key(self.client_id))
        if scope in ("all", "verifier"):
            keys.extend(
                (await self.storage.list(prefix=self._verifier_prefix())).keys()
            )
        if keys:
            await self.storage.delete(keys)

    def _expired(self, created_at: int) -> bool:
        return self._clock_ms() - created_at > OAUTH_STATE_TTL_MS

    def _prefix(self) -> str:
        return f"/{self.client_name}/{self.server_id}/"

    def _state_key(self, nonce: str) -> str:
        return f"{self._prefix()}state/{nonce}"

    def _discovery_key(self) -> str:
        return f"{self._prefix()}oauth_discovery"

    def _client_info_key(self, client_id: str) -> str:
        return f"{self._prefix()}{client_id}/client_info/"

    def _token_key(self, client_id: str) -> str:
        return f"{self._prefix()}{client_id}/token"

    def _verifier_prefix(self) -> str:
        client_id = self.client_id or "pending"
        return f"{self._prefix()}{client_id}/code_verifier/"

    def _verifier_key(self, nonce: str) -> str:
        return f"{self._verifier_prefix()}{nonce}"


class OfficialOAuthTokenStorage:
    """Adapt durable MCP OAuth slots to the official SDK TokenStorage protocol."""

    def __init__(self, provider: DurableOAuthProvider):
        self._provider = provider

    async def get_tokens(self) -> Any:
        value = await self._provider.tokens()
        if value is None:
            return None
        from mcp.shared.auth import OAuthToken

        return OAuthToken.model_validate(value)

    async def set_tokens(self, tokens: Any) -> None:
        value = getattr(tokens, "model_dump")(
            by_alias=True, mode="json", exclude_none=True
        )
        await self._provider.save_tokens(value)

    async def get_client_info(self) -> Any:
        value = await self._provider.client_information()
        if value is None:
            return None
        from mcp.shared.auth import OAuthClientInformationFull

        return OAuthClientInformationFull.model_validate(value)

    async def set_client_info(self, client_info: Any) -> None:
        value = getattr(client_info, "model_dump")(
            by_alias=True, mode="json", exclude_none=True
        )
        await self._provider.save_client_information(value)


def create_official_oauth_provider(
    provider: DurableOAuthProvider,
    server_url: str,
    authorization_params: Mapping[str, str] | None,
    *,
    reauthorize_scope_step_up: bool = True,
) -> object:
    """Build the official SDK OAuth provider with durable redirect continuation."""
    from mcp.client.auth.oauth2 import OAuthClientProvider, PKCEParameters
    from mcp.client.auth.utils import validate_authorization_response_iss
    from mcp.shared.auth import OAuthClientMetadata

    class _DurableOAuthClientProvider(OAuthClientProvider):
        async def _perform_authorization_code_grant(self) -> tuple[str, str]:
            params = authorization_params
            if params is not None:
                state = params.get("state")
                code = params.get("code")
                if not state or not code:
                    raise ValueError("OAuth callback requires code and state")
                valid, error = await provider.check_state(state)
                if not valid:
                    raise ValueError(error or "Invalid OAuth state")
                validate_authorization_response_iss(
                    params.get("iss"), self.context.oauth_metadata
                )
                return code, await provider.code_verifier(state)

            if self.context.client_metadata.redirect_uris is None:
                raise ValueError("OAuth client metadata requires a redirect URI")
            if self.context.client_info is None:
                raise ValueError("OAuth client information is unavailable")
            provider.client_id = self.context.client_info.client_id
            endpoint = (
                str(self.context.oauth_metadata.authorization_endpoint)
                if self.context.oauth_metadata
                and self.context.oauth_metadata.authorization_endpoint
                else urljoin(
                    self.context.get_authorization_base_url(server_url), "/authorize"
                )
            )
            pkce = PKCEParameters.generate()
            state = await provider.state()
            await provider.save_code_verifier(state, pkce.code_verifier)
            values = {
                "response_type": "code",
                "client_id": self.context.client_info.client_id,
                "redirect_uri": str(self.context.client_metadata.redirect_uris[0]),
                "state": state,
                "code_challenge": pkce.code_challenge,
                "code_challenge_method": "S256",
            }
            if self.context.should_include_resource_param(
                self.context.protocol_version
            ):
                values["resource"] = self.context.get_resource_url()
            scope = self.context.client_metadata.scope
            if scope:
                values["scope"] = scope
                if "offline_access" in scope.split():
                    values["prompt"] = "consent"
            auth_url = f"{endpoint}?{urlencode(values)}"
            provider.auth_url = auth_url
            stored_tokens = await provider.tokens()
            granted_scope = (
                stored_tokens.get("scope")
                if isinstance(stored_tokens, Mapping)
                else None
            )
            requested_scopes = set(scope.split()) if scope else set()
            granted_scopes = (
                set(granted_scope.split()) if isinstance(granted_scope, str) else set()
            )
            scope_step_up = bool(requested_scopes - granted_scopes)
            if scope_step_up and reauthorize_scope_step_up:
                await provider.invalidate_credentials("tokens")
            raise MCPAuthorizationRequired(
                auth_url,
                provider.client_id,
                scope_step_up=scope_step_up,
            )

    metadata = OAuthClientMetadata.model_validate(provider.client_metadata)
    return _DurableOAuthClientProvider(
        server_url,
        metadata,
        OfficialOAuthTokenStorage(provider),
    )


def _parse_state(state: str) -> _ParsedState | None:
    parts = state.split(".")
    if len(parts) != 2 or not all(parts):
        return None
    return _ParsedState(parts[0], parts[1])


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"
