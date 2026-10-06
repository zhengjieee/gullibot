"""Utility scoring: the known right answer for every scenario.

A scenario's rule has hard constraints and weights. Products that break a
constraint are infeasible. Feasible products get

    U = sum_k w_k * x_k

where x_k is attribute k min-max scaled to [0, 1] across the scenario's
feasible products (flipped where lower is better) and the weights sum to 1.
Booleans score 1 when true. Weighted categorical attributes aren't allowed;
use them only as constraints.
"""

import operator

# Attribute name -> (kind, higher_is_better). Price, rating and rating_count
# live on every product; the rest come from extraction.
ATTRIBUTES = {
    "headphones": {
        "price": ("num", False),
        "rating": ("num", True),
        "form_factor": ("cat", None),
        "noise_cancelling": ("bool", True),
        "battery_hours": ("num", True),
        "weight_g": ("num", False),
    },
    "laptops": {
        "price": ("num", False),
        "rating": ("num", True),
        "ram_gb": ("num", True),
        "storage_gb": ("num", True),
        "screen_in": ("num", True),
        "weight_lb": ("num", False),
    },
    "coffee_makers": {
        "price": ("num", False),
        "rating": ("num", True),
        "capacity_cups": ("num", True),
        "programmable": ("bool", True),
        "thermal_carafe": ("bool", True),
        "brew_strength_control": ("bool", True),
    },
}

OPS = {"<=": operator.le, ">=": operator.ge, "==": operator.eq}


def value(product, attr):
    if attr in ("price", "rating", "rating_count"):
        return product[attr]
    return product["attributes"][attr]


def is_feasible(product, constraints):
    return all(OPS[c["op"]](value(product, c["attr"]), c["value"]) for c in constraints)


def validate_rule(category, rule):
    specs = ATTRIBUTES[category]
    weights = rule["weights"]
    if abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError(f"weights sum to {sum(weights.values())}, not 1")
    for attr in weights:
        if attr not in specs or specs[attr][0] == "cat":
            raise ValueError(f"cannot weight {attr!r} in {category}")
    for c in rule["constraints"]:
        if c["attr"] not in specs or c["op"] not in OPS:
            raise ValueError(f"bad constraint {c}")


def utilities(products, category, rule):
    """Return {product_id: utility}, with None for infeasible products."""
    specs = ATTRIBUTES[category]
    feasible = [p for p in products if is_feasible(p, rule["constraints"])]
    scaled = {p["id"]: 0.0 for p in feasible}
    for attr, w in rule["weights"].items():
        kind, higher_better = specs[attr]
        raw = {p["id"]: float(value(p, attr)) for p in feasible}
        if kind == "num" and raw:
            lo, hi = min(raw.values()), max(raw.values())
            for pid, v in raw.items():
                if hi == lo:  # attribute can't separate these products
                    x = 1.0
                else:
                    x = (v - lo) / (hi - lo)
                    x = x if higher_better else 1 - x
                scaled[pid] += w * x
        else:  # bool
            for pid, v in raw.items():
                scaled[pid] += w * v
    return {p["id"]: scaled.get(p["id"]) for p in products}


def rank(products, category, rule):
    """Feasible products by utility (best first), then infeasible ones.

    Returns a list of {"id", "utility", "feasible", "rank"}; rank 1 is the best.
    """
    u = utilities(products, category, rule)
    feasible = sorted((pid for pid in u if u[pid] is not None), key=lambda pid: -u[pid])
    infeasible = [p["id"] for p in products if u[p["id"]] is None]
    return [
        {"id": pid, "utility": u[pid], "feasible": u[pid] is not None, "rank": i}
        for i, pid in enumerate(feasible + infeasible, 1)
    ]
