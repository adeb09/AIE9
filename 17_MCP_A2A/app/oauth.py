import json
import secrets
import time

import aiosqlite

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .db import init_db


class CatShopOAuthProvider(OAuthAuthorizationServerProvider):
    _db: aiosqlite.Connection | None = None

    def __init__(self, issuer_url: str):
        self.issuer_url = issuer_url

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect("catshop.db")
            await init_db(self._db)
        return self._db

    # -- clients --

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT client_info_json FROM oauth_clients WHERE client_id = ?",
            (client_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return OAuthClientInformationFull.model_validate_json(row[0])

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        db = await self._get_db()
        await db.execute(
            "INSERT OR REPLACE INTO oauth_clients (client_id, client_info_json) VALUES (?, ?)",
            (client_info.client_id, client_info.model_dump_json()),
        )
        await db.commit()

    # -- authorize --

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        db = await self._get_db()
        request_id = secrets.token_hex(16)
        scopes = params.scopes or ["read", "write"]

        await db.execute(
            """INSERT INTO pending_authorizations
               (request_id, client_id, scopes_json, code_challenge,
                redirect_uri, redirect_uri_provided_explicitly, resource, state, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                client.client_id,
                json.dumps(scopes),
                params.code_challenge,
                str(params.redirect_uri),
                int(params.redirect_uri_provided_explicitly),
                str(params.resource) if params.resource else None,
                params.state,
                time.time() + 600,
            ),
        )
        await db.commit()

        return f"{self.issuer_url}/login?req={request_id}"

    # -- authorization codes --

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM authorization_codes WHERE code = ? AND client_id = ?",
            (authorization_code, client.client_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return AuthorizationCode(
            code=row[0],
            client_id=row[1],
            scopes=json.loads(row[2]),
            expires_at=row[3],
            code_challenge=row[4],
            redirect_uri=row[5],
            redirect_uri_provided_explicitly=bool(row[6]),
            resource=row[7],
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        db = await self._get_db()

        cursor = await db.execute(
            "SELECT username FROM authorization_codes WHERE code = ?",
            (authorization_code.code,),
        )
        row = await cursor.fetchone()
        username = row[0] if row else "unknown"

        await db.execute(
            "DELETE FROM authorization_codes WHERE code = ?",
            (authorization_code.code,),
        )

        access_token = secrets.token_hex(32)
        refresh_token = secrets.token_hex(32)
        expires_in = 3600
        now = time.time()

        await db.execute(
            "INSERT INTO access_tokens (token, client_id, scopes_json, expires_at, resource) VALUES (?, ?, ?, ?, ?)",
            (
                access_token,
                client.client_id,
                json.dumps(authorization_code.scopes),
                now + expires_in,
                authorization_code.resource,
            ),
        )
        await db.execute(
            "INSERT INTO refresh_tokens (token, client_id, scopes_json, expires_at) VALUES (?, ?, ?, ?)",
            (
                refresh_token,
                client.client_id,
                json.dumps(authorization_code.scopes),
                now + 86400,
            ),
        )
        await db.execute(
            "INSERT INTO token_users (token, username) VALUES (?, ?)",
            (access_token, username),
        )
        await db.execute(
            "INSERT INTO token_users (token, username) VALUES (?, ?)",
            (refresh_token, username),
        )
        await db.commit()

        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in,
            scope=" ".join(authorization_code.scopes),
            refresh_token=refresh_token,
        )

    # -- refresh tokens --

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM refresh_tokens WHERE token = ? AND client_id = ?",
            (refresh_token, client.client_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return RefreshToken(
            token=row[0],
            client_id=row[1],
            scopes=json.loads(row[2]),
            expires_at=int(row[3]) if row[3] else None,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        db = await self._get_db()

        cursor = await db.execute(
            "SELECT username FROM token_users WHERE token = ?",
            (refresh_token.token,),
        )
        row = await cursor.fetchone()
        username = row[0] if row else "unknown"

        await db.execute(
            "DELETE FROM refresh_tokens WHERE token = ?", (refresh_token.token,)
        )
        await db.execute(
            "DELETE FROM token_users WHERE token = ?", (refresh_token.token,)
        )

        new_access = secrets.token_hex(32)
        new_refresh = secrets.token_hex(32)
        expires_in = 3600
        now = time.time()
        effective_scopes = scopes or refresh_token.scopes

        await db.execute(
            "INSERT INTO access_tokens (token, client_id, scopes_json, expires_at) VALUES (?, ?, ?, ?)",
            (new_access, client.client_id, json.dumps(effective_scopes), now + expires_in),
        )
        await db.execute(
            "INSERT INTO refresh_tokens (token, client_id, scopes_json, expires_at) VALUES (?, ?, ?, ?)",
            (new_refresh, client.client_id, json.dumps(effective_scopes), now + 86400),
        )
        await db.execute(
            "INSERT INTO token_users (token, username) VALUES (?, ?)",
            (new_access, username),
        )
        await db.execute(
            "INSERT INTO token_users (token, username) VALUES (?, ?)",
            (new_refresh, username),
        )
        await db.commit()

        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=expires_in,
            scope=" ".join(effective_scopes),
            refresh_token=new_refresh,
        )

    # -- access tokens --

    async def load_access_token(self, token: str) -> AccessToken | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM access_tokens WHERE token = ?", (token,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        expires_at = row[3]
        if expires_at and time.time() > expires_at:
            await db.execute("DELETE FROM access_tokens WHERE token = ?", (token,))
            await db.execute("DELETE FROM token_users WHERE token = ?", (token,))
            await db.commit()
            return None
        return AccessToken(
            token=row[0],
            client_id=row[1],
            scopes=json.loads(row[2]),
            expires_at=int(expires_at) if expires_at else None,
            resource=row[4],
        )

    # -- revocation --

    async def revoke_token(
        self, token: AccessToken | RefreshToken
    ) -> None:
        db = await self._get_db()
        if isinstance(token, AccessToken):
            await db.execute(
                "DELETE FROM access_tokens WHERE token = ?", (token.token,)
            )
        else:
            await db.execute(
                "DELETE FROM refresh_tokens WHERE token = ?", (token.token,)
            )
        await db.execute("DELETE FROM token_users WHERE token = ?", (token.token,))
        await db.commit()

    # -- helper to get username from access token --

    async def get_username_for_token(self, token: str) -> str | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT username FROM token_users WHERE token = ?", (token,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None
