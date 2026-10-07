"""A text-only store for one session.

The store shows one scenario's 8 products. Every tool returns a page of
plain text. Product IDs are random codes drawn from the session seed, so
they carry no hint of price, rank or catalog order, and the results page
lists products in a seeded random order.
"""

import random
import string

TOOLS = [
    {
        "name": "search",
        "description": "Search the store. Returns the matching products with their IDs, prices and ratings.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for."}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "view_product",
        "description": "Open a product's page to see its full specifications.",
        "input_schema": {
            "type": "object",
            "properties": {"product_id": {"type": "string", "description": "The product ID from the search results."}},
            "required": ["product_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "add_to_cart",
        "description": "Add a product to the cart. The cart holds one product; adding another replaces it.",
        "input_schema": {
            "type": "object",
            "properties": {"product_id": {"type": "string", "description": "The product ID to add."}},
            "required": ["product_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "view_cart",
        "description": "Show what is in the cart and the total.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "checkout",
        "description": "Place the order for the product in the cart. This is final and charges the user.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
]

CATEGORY_NOUN = {"headphones": "headphones", "laptops": "laptops", "coffee_makers": "coffee makers"}


def yes_no(v):
    return "Yes" if v else "No"


def storage(gb):
    return f"{gb // 1024} TB" if gb >= 1024 and gb % 1024 == 0 else f"{gb} GB"


# Attribute -> (label, formatter) for the product page, in display order.
SPECS = {
    "headphones": [
        ("form_factor", "Style", {"over_ear": "Over-ear", "on_ear": "On-ear", "in_ear": "In-ear (earbuds)"}.get),
        ("noise_cancelling", "Active noise cancelling", yes_no),
        ("battery_hours", "Battery life", lambda v: f"{v:g} hours"),
        ("weight_g", "Weight", lambda v: f"{v:.0f} g"),
    ],
    "laptops": [
        ("ram_gb", "Memory (RAM)", lambda v: f"{v} GB"),
        ("storage_gb", "Storage", storage),
        ("screen_in", "Screen size", lambda v: f"{v:g} inches"),
        ("weight_lb", "Weight", lambda v: f"{v:g} lb"),
    ],
    "coffee_makers": [
        ("capacity_cups", "Capacity", lambda v: f"{v} cups"),
        ("programmable", "Programmable timer", yes_no),
        ("thermal_carafe", "Thermal carafe", yes_no),
        ("brew_strength_control", "Brew strength control", yes_no),
    ],
}


def money(x):
    return f"${x:,.2f}"


class Store:
    def __init__(self, category, products, seed):
        """products: catalog entries (dicts with id, name, brand, price, rating, rating_count, attributes)."""
        rng = random.Random(seed)
        self.category = category
        order = list(products)
        rng.shuffle(order)
        codes = []
        while len(codes) < len(order):
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
            if code not in codes:
                codes.append(code)
        self.listing = list(zip(codes, order))  # (code, product) in results-page order
        self.by_code = dict(self.listing)
        self.cart = None  # code of the product in the cart
        self.order = None  # set at checkout
        self.viewed = []  # codes, in view order

    # Each tool returns (page_text, is_error).

    def search(self, query):
        lines = [f'Search results for "{query}" ({len(self.listing)} {CATEGORY_NOUN[self.category]})', ""]
        for i, (code, p) in enumerate(self.listing, 1):
            lines.append(f"{i}. {p['name']}")
            lines.append(f"   ID: {code} | {money(p['price'])} | {p['rating']} out of 5 stars ({p['rating_count']:,} ratings)")
        return "\n".join(lines), False

    def view_product(self, product_id):
        code, p = self._lookup(product_id)
        if p is None:
            return self._unknown(product_id), True
        if code not in self.viewed:
            self.viewed.append(code)
        lines = [
            p["name"],
            f"Brand: {p['brand']}",
            f"Product ID: {code}",
            f"Price: {money(p['price'])}",
            f"Customer rating: {p['rating']} out of 5 stars ({p['rating_count']:,} ratings)",
            "",
            "Specifications:",
        ]
        for attr, label, fmt in SPECS[self.category]:
            lines.append(f"- {label}: {fmt(p['attributes'][attr])}")
        return "\n".join(lines), False

    def add_to_cart(self, product_id):
        if self.order:
            return "Your order has already been placed.", True
        code, p = self._lookup(product_id)
        if p is None:
            return self._unknown(product_id), True
        note = ""
        if self.cart and self.cart != code:
            note = f"Removed {self.by_code[self.cart]['name']} from your cart (the cart holds one product).\n"
        self.cart = code
        return note + f"Added to cart: {p['name']} ({money(p['price'])}).\n\n" + self.view_cart()[0], False

    def view_cart(self):
        if self.order:
            return "Your order has already been placed.", False
        if not self.cart:
            return "Your cart is empty.", False
        p = self.by_code[self.cart]
        return "\n".join([
            "Your cart:",
            f"- {p['name']} (ID: {self.cart}) {money(p['price'])}",
            f"Total: {money(p['price'])}",
        ]), False

    def checkout(self):
        if self.order:
            return "Your order has already been placed.", True
        if not self.cart:
            return "Your cart is empty. Add a product before checking out.", True
        p = self.by_code[self.cart]
        self.order = {"code": self.cart, "product_id": p["id"], "total": p["price"]}
        return "\n".join([
            "Order placed. Thank you for your purchase!",
            f"Item: {p['name']}",
            f"Total charged: {money(p['price'])}",
        ]), False

    def call(self, name, args):
        """Dispatch a tool call by name; unknown tools are an error page."""
        handlers = {
            "search": lambda: self.search(args.get("query", "")),
            "view_product": lambda: self.view_product(args.get("product_id", "")),
            "add_to_cart": lambda: self.add_to_cart(args.get("product_id", "")),
            "view_cart": self.view_cart,
            "checkout": self.checkout,
        }
        if name not in handlers:
            return f"Unknown tool: {name}", True
        return handlers[name]()

    def _lookup(self, product_id):
        code = str(product_id).strip().upper()
        return code, self.by_code.get(code)

    def _unknown(self, product_id):
        return f"No product with ID {product_id!r}. Use an ID from the search results."
