"""Load the week-1 data files."""

import json

from prep.config import DATA


def load_catalog():
    """All products keyed by id."""
    catalog = json.loads((DATA / "catalog.json").read_text())
    return {p["id"]: p for products in catalog.values() for p in products}


def load_scenarios():
    return json.loads((DATA / "scenarios.json").read_text())


def load_pushed():
    """{scenario_id: product_id}"""
    pushed = json.loads((DATA / "pushed_items.json").read_text())
    return {sid: v["product_id"] for sid, v in pushed.items()}
