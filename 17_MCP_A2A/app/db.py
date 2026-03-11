import aiosqlite

PRODUCTS = [
    ("Whisker Wand", "Interactive feather toy on a flexible wand", 9.99, "toys"),
    ("Catnip Mouse", "Organic catnip-stuffed plush mouse", 4.99, "toys"),
    ("Laser Pointer Pro", "Red-dot laser with adjustable patterns", 12.99, "toys"),
    ("Cozy Cat Bed", "Soft donut-shaped bed for curling up", 29.99, "beds"),
    ("Window Hammock", "Suction-cup window perch with fleece lining", 24.99, "beds"),
    ("Salmon Treats", "Freeze-dried wild salmon bites, 100g", 7.99, "food"),
    ("Tuna Crunchies", "Crunchy tuna-flavored dental treats, 80g", 5.99, "food"),
    ("Scratching Post Tower", "3-tier sisal scratching post with platforms", 49.99, "furniture"),
]


async def init_db(db: aiosqlite.Connection):
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS oauth_clients (
            client_id TEXT PRIMARY KEY,
            client_info_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS authorization_codes (
            code TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            scopes_json TEXT NOT NULL,
            expires_at REAL NOT NULL,
            code_challenge TEXT NOT NULL,
            redirect_uri TEXT NOT NULL,
            redirect_uri_provided_explicitly INTEGER NOT NULL,
            resource TEXT,
            username TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS access_tokens (
            token TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            scopes_json TEXT NOT NULL,
            expires_at REAL,
            resource TEXT
        );
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            scopes_json TEXT NOT NULL,
            expires_at REAL
        );
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cart_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            UNIQUE(username, product_id)
        );
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pending_authorizations (
            request_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            scopes_json TEXT NOT NULL,
            code_challenge TEXT NOT NULL,
            redirect_uri TEXT NOT NULL,
            redirect_uri_provided_explicitly INTEGER NOT NULL,
            resource TEXT,
            state TEXT,
            expires_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS token_users (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL
        );
        """
    )

    # Seed products if empty
    cursor = await db.execute("SELECT COUNT(*) FROM products")
    (count,) = await cursor.fetchone()
    if count == 0:
        await db.executemany(
            "INSERT INTO products (name, description, price, category) VALUES (?, ?, ?, ?)",
            PRODUCTS,
        )
    await db.commit()
