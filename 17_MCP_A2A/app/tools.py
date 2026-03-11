import secrets

from mcp.server.auth.middleware.auth_context import get_access_token

from .server import mcp, oauth_provider


async def _get_username() -> str:
    token = get_access_token()
    if token is None:
        raise ValueError("Not authenticated")
    username = await oauth_provider.get_username_for_token(token.token)
    if username is None:
        raise ValueError("User not found for token")
    return username


@mcp.tool()
async def list_products(category: str | None = None) -> list[dict]:
    """Browse the cat shop catalog. Optionally filter by category (toys, beds, food, furniture)."""
    db = await oauth_provider._get_db()
    if category:
        cursor = await db.execute(
            "SELECT id, name, description, price, category FROM products WHERE category = ?",
            (category,),
        )
    else:
        cursor = await db.execute(
            "SELECT id, name, description, price, category FROM products"
        )
    rows = await cursor.fetchall()
    return [
        {"id": r[0], "name": r[1], "description": r[2], "price": r[3], "category": r[4]}
        for r in rows
    ]


@mcp.tool()
async def get_product(product_id: int) -> dict:
    """Get full details of a single product by its ID."""
    db = await oauth_provider._get_db()
    cursor = await db.execute(
        "SELECT id, name, description, price, category FROM products WHERE id = ?",
        (product_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return {"error": "Product not found"}
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "price": row[3],
        "category": row[4],
    }


@mcp.tool()
async def add_to_cart(product_id: int, quantity: int = 1) -> dict:
    """Add a product to your shopping cart. If already in cart, quantity is increased."""
    username = await _get_username()
    db = await oauth_provider._get_db()

    cursor = await db.execute("SELECT name FROM products WHERE id = ?", (product_id,))
    product = await cursor.fetchone()
    if product is None:
        return {"error": "Product not found"}

    await db.execute(
        """INSERT INTO cart_items (username, product_id, quantity)
           VALUES (?, ?, ?)
           ON CONFLICT(username, product_id)
           DO UPDATE SET quantity = quantity + excluded.quantity""",
        (username, product_id, quantity),
    )
    await db.commit()
    return {"success": True, "message": f"Added {quantity}x {product[0]} to your cart"}


@mcp.tool()
async def view_cart() -> dict:
    """View everything in your shopping cart with quantities and totals."""
    username = await _get_username()
    db = await oauth_provider._get_db()
    cursor = await db.execute(
        """SELECT p.id, p.name, p.price, c.quantity
           FROM cart_items c JOIN products p ON c.product_id = p.id
           WHERE c.username = ?""",
        (username,),
    )
    rows = await cursor.fetchall()
    items = [
        {
            "product_id": r[0],
            "name": r[1],
            "price": r[2],
            "quantity": r[3],
            "subtotal": round(r[2] * r[3], 2),
        }
        for r in rows
    ]
    total = round(sum(i["subtotal"] for i in items), 2)
    return {"items": items, "total": total, "item_count": len(items)}


@mcp.tool()
async def remove_from_cart(product_id: int) -> dict:
    """Remove a product from your shopping cart."""
    username = await _get_username()
    db = await oauth_provider._get_db()
    cursor = await db.execute(
        "DELETE FROM cart_items WHERE username = ? AND product_id = ?",
        (username, product_id),
    )
    await db.commit()
    if cursor.rowcount == 0:
        return {"error": "Item not in cart"}
    return {"success": True, "message": "Item removed from cart"}


@mcp.tool()
async def checkout() -> dict:
    """Complete your purchase. Shows order summary and clears the cart."""
    username = await _get_username()
    db = await oauth_provider._get_db()

    cart = await view_cart()
    if not cart["items"]:
        return {"error": "Your cart is empty"}

    await db.execute("DELETE FROM cart_items WHERE username = ?", (username,))
    await db.commit()

    order_id = secrets.token_hex(8).upper()
    return {
        "order_id": order_id,
        "status": "confirmed",
        "items": cart["items"],
        "total": cart["total"],
        "message": f"Order {order_id} confirmed! Thanks {username}, your cats will love their new goodies!",
    }
