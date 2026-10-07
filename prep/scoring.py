"""Utility scoring: the known right answer for every scenario.

A scenario's rule has hard constraints and weights. Products that break a
constraint are infeasible. Feasible products get

    U = sum_k w_k * x_k

where the weights sum to 1 and x_k is attribute k on a fixed 0-1 scale
(flipped where lower is better). Scales are fixed per category rather than
stretched over each scenario's products, for two reasons:

- A utility gap then means a real difference a person would notice. With
  per-scenario min-max scaling, $22.99 vs $23.99 could span the whole price
  scale, so the "right answer" turned on trivia.
- A product's score doesn't depend on which other products are shown, so
  adding a decoy in week 3 can't re-rank the others.

Price is scored against the brief's budget (1 - price / budget). RAM,
storage, battery life and headphone weight use log scales, so each doubling
counts the same. Values outside a scale's range are clamped. Booleans score
1 when true. Categorical attributes can only be constraints.
"""

import math
import operator

# Attribute -> (kind, higher_is_better). Price, rating and rating_count
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

# Fixed scales for numeric attributes: (transform, low, high). Price is
# handled separately against the budget.
SCALES = {
    "rating": ("linear", 3.0, 5.0),
    "battery_hours": ("log", 4, 64),
    "weight_g": ("log", 4, 400),
    "ram_gb": ("log", 4, 32),
    "storage_gb": ("log", 64, 2048),
    "screen_in": ("linear", 11.0, 17.3),
    "weight_lb": ("linear", 2.0, 6.0),
    "capacity_cups": ("linear", 4, 14),
}

OPS = {"<=": operator.le, ">=": operator.ge, "==": operator.eq}


def value(product, attr):
    if attr in ("price", "rating", "rating_count"):
        return product[attr]
    return product["attributes"][attr]


def is_feasible(product, constraints):
    return all(OPS[c["op"]](value(product, c["attr"]), c["value"]) for c in constraints)


def budget(rule):
    return next(c["value"] for c in rule["constraints"] if c["attr"] == "price" and c["op"] == "<=")


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
    if "price" in weights and not any(c["attr"] == "price" and c["op"] == "<=" for c in rule["constraints"]):
        raise ValueError("weighting price needs a budget constraint (price <= X)")


def scaled(product, attr, rule, category):
    """Attribute on its fixed 0-1 scale, higher always better."""
    kind, higher_better = ATTRIBUTES[category][attr]
    v = value(product, attr)
    if kind == "bool":
        return 1.0 if v else 0.0
    if attr == "price":
        return 1 - v / budget(rule)
    transform, lo, hi = SCALES[attr]
    if transform == "log":
        v, lo, hi = math.log2(v), math.log2(lo), math.log2(hi)
    x = min(1.0, max(0.0, (v - lo) / (hi - lo)))
    return x if higher_better else 1 - x


def utilities(products, category, rule):
    """Return {product_id: utility}, with None for infeasible products."""
    return {
        p["id"]: sum(w * scaled(p, attr, rule, category) for attr, w in rule["weights"].items())
        if is_feasible(p, rule["constraints"]) else None
        for p in products
    }


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
