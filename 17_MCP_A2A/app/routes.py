import json
import secrets
import time
from html import escape

from mcp.server.auth.provider import construct_redirect_uri
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from .server import mcp, oauth_provider

LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Cat Shop - Login</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
            background: #000;
            color: #ededed;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .card {{
            width: 100%;
            max-width: 360px;
            padding: 0 24px;
        }}
        .logo {{
            font-size: 40px;
            margin-bottom: 24px;
            text-align: center;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 700;
            color: #ededed;
            text-align: center;
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            color: #888;
            font-size: 14px;
            text-align: center;
            margin-top: 8px;
            margin-bottom: 32px;
            line-height: 1.5;
        }}
        label {{
            display: block;
            font-size: 13px;
            font-weight: 500;
            color: #a1a1a1;
            margin-bottom: 8px;
        }}
        input[type="text"] {{
            width: 100%;
            padding: 10px 12px;
            background: transparent;
            border: 1px solid #333;
            border-radius: 6px;
            font-size: 14px;
            color: #ededed;
            outline: none;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        input[type="text"]::placeholder {{ color: #555; }}
        input[type="text"]:focus {{
            border-color: #ededed;
            box-shadow: 0 0 0 1px #ededed;
        }}
        button {{
            width: 100%;
            padding: 10px 16px;
            background: #ededed;
            color: #000;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            margin-top: 20px;
            transition: background 0.15s ease;
        }}
        button:hover {{ background: #fff; }}
        button:active {{ background: #ccc; }}
        .divider {{
            display: flex;
            align-items: center;
            margin-top: 24px;
        }}
        .divider::before, .divider::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: #222;
        }}
        .divider span {{
            font-size: 12px;
            color: #555;
            padding: 0 12px;
        }}
        .hint {{
            color: #555;
            font-size: 13px;
            text-align: center;
            margin-top: 12px;
        }}
        .error {{
            background: rgba(255, 50, 50, 0.08);
            border: 1px solid rgba(255, 50, 50, 0.2);
            color: #ff6166;
            font-size: 13px;
            padding: 10px 12px;
            border-radius: 6px;
            margin-bottom: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="logo">🐱</div>
        <h1>Cat Shop</h1>
        <p class="subtitle">Sign in to your account to continue</p>
        {error}
        <form method="POST" action="/login?req={req}">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" placeholder="Enter your username" required
                   minlength="2" maxlength="30" pattern="[a-zA-Z0-9_]+"
                   title="Letters, numbers, and underscores only" autofocus>
            <button type="submit">Continue</button>
        </form>
        <div class="divider"><span>or</span></div>
        <p class="hint">New here? Just pick a username.</p>
    </div>
</body>
</html>"""


@mcp.custom_route("/login", methods=["GET", "POST"])
async def login_page(request: Request) -> Response:
    req_id = request.query_params.get("req", "")
    db = await oauth_provider._get_db()

    cursor = await db.execute(
        "SELECT * FROM pending_authorizations WHERE request_id = ? AND expires_at > ?",
        (req_id, time.time()),
    )
    pending = await cursor.fetchone()
    if pending is None:
        return HTMLResponse(
            '<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>Error</title>'
            "<style>*{margin:0;padding:0;box-sizing:border-box}"
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "background:#000;color:#ededed;min-height:100vh;display:flex;align-items:center;"
            "justify-content:center;text-align:center}"
            ".wrap{max-width:400px;padding:0 24px}"
            "h1{font-size:48px;font-weight:700;color:#333;margin-bottom:16px}"
            "p{color:#888;font-size:14px;line-height:1.6}"
            "</style></head><body><div class='wrap'>"
            "<h1>400</h1><p>This login request is invalid or has expired.<br>"
            "Please go back and try again.</p></div></body></html>",
            status_code=400,
        )

    if request.method == "GET":
        html = LOGIN_PAGE_HTML.format(req=escape(req_id), error="")
        return HTMLResponse(html)

    # POST - process login
    form = await request.form()
    username = form.get("username", "").strip()

    if not username or len(username) < 2 or len(username) > 30:
        html = LOGIN_PAGE_HTML.format(
            req=escape(req_id),
            error='<p class="error">Username must be 2-30 characters.</p>',
        )
        return HTMLResponse(html, status_code=400)

    # Create user if not exists
    await db.execute(
        "INSERT OR IGNORE INTO users (username, created_at) VALUES (?, ?)",
        (username, time.time()),
    )

    # Generate auth code linked to this user
    code = secrets.token_hex(32)
    scopes = json.loads(pending[2])  # scopes_json

    await db.execute(
        """INSERT INTO authorization_codes
           (code, client_id, scopes_json, expires_at, code_challenge,
            redirect_uri, redirect_uri_provided_explicitly, resource, username)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            code,
            pending[1],  # client_id
            pending[2],  # scopes_json
            time.time() + 300,
            pending[3],  # code_challenge
            pending[4],  # redirect_uri
            pending[5],  # redirect_uri_provided_explicitly
            pending[6],  # resource
            username,
        ),
    )

    # Clean up pending request
    await db.execute(
        "DELETE FROM pending_authorizations WHERE request_id = ?", (req_id,)
    )
    await db.commit()

    # Redirect back to the OAuth client with the code
    redirect_uri = construct_redirect_uri(pending[4], code=code, state=pending[7])
    return RedirectResponse(url=redirect_uri, status_code=302)
