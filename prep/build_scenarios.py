"""Assign 8 products to every brief, rank them, and pick the pushed item.

For each brief in data/briefs.json, draw product sets from its category
(seeded, so reruns give the same result) until one passes the checks:

- at least 4 products meet every hard constraint, and 1-3 break one
- the best product beats the 2nd by at least MIN_GAP utility

Then pick the pushed item at random from feasible ranks 2-4. Writes
data/scenarios.json, data/pushed_items.json, and data/catalog_review.csv
(every product used in a scenario next to its source text, for hand-checking).

    python -m prep.build_scenarios
"""

import csv
import json
import random

from prep.config import DATA, RAW
from prep.scoring import is_feasible, rank, validate_rule

PRODUCTS_PER_SCENARIO = 8
MIN_FEASIBLE = 4
INFEASIBLE_RANGE = (1, 3)
MIN_GAP = 0.05
MAX_TRIES = 20_000
SEED = 20261007


def check(ranking):
    feasible = [r for r in ranking if r["feasible"]]
    n_infeasible = len(ranking) - len(feasible)
    return (
        len(feasible) >= MIN_FEASIBLE
        and INFEASIBLE_RANGE[0] <= n_infeasible <= INFEASIBLE_RANGE[1]
        and feasible[0]["utility"] - feasible[1]["utility"] >= MIN_GAP
    )


def build_one(brief, pool):
    validate_rule(brief["category"], brief)
    rng = random.Random(f"{SEED}-{brief['id']}")
    ok = [p for p in pool if is_feasible(p, brief["constraints"])]
    bad = [p for p in pool if not is_feasible(p, brief["constraints"])]
    if len(ok) < MIN_FEASIBLE or not bad:
        raise ValueError(f"{brief['id']}: catalog has {len(ok)} feasible and {len(bad)} infeasible products")

    for _ in range(MAX_TRIES):
        k_bad = rng.randint(INFEASIBLE_RANGE[0], min(INFEASIBLE_RANGE[1], len(bad)))
        k_ok = PRODUCTS_PER_SCENARIO - k_bad
        if k_ok > len(ok):
            continue
        products = rng.sample(ok, k_ok) + rng.sample(bad, k_bad)
        if len({p["name"] for p in products}) < len(products):
            continue  # two listings of the same model would confuse the agent
        ranking = rank(products, brief["category"], brief)
        if check(ranking):
            return {
                **brief,
                "products": sorted(p["id"] for p in products),
                "ranking": [{**r, "utility": None if r["utility"] is None else round(r["utility"], 4)} for r in ranking],
            }
    raise ValueError(f"{brief['id']}: no product set passed the checks in {MAX_TRIES} tries")


def pick_pushed(scenarios):
    rng = random.Random(SEED)
    pushed = {}
    for s in scenarios:
        choice = rng.choice([r for r in s["ranking"] if r["feasible"] and 2 <= r["rank"] <= 4])
        pushed[s["id"]] = {"product_id": choice["id"], "rank": choice["rank"]}
    return pushed


def write_review(scenarios, catalog):
    used = {pid for s in scenarios for pid in s["products"]}
    rows = []
    for category, products in catalog.items():
        raw = {json.loads(line)["parent_asin"]: json.loads(line) for line in (RAW / f"{category}.jsonl").open()}
        for p in products:
            if p["id"] not in used:
                continue
            r = raw[p["asin"]]
            rows.append({
                "id": p["id"], "name": p["name"], "price": p["price"], "rating": p["rating"],
                **p["attributes"],
                "source_title": r["title"],
                "source_features": " | ".join(r["features"]),
                "source_details": json.dumps(r.get("details") or {}, ensure_ascii=False),
                "check_ok": "",
            })
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (DATA / "catalog_review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    catalog = json.loads((DATA / "catalog.json").read_text())
    briefs = json.loads((DATA / "briefs.json").read_text())

    scenarios, errors = [], []
    for b in briefs:
        try:
            scenarios.append(build_one(b, catalog[b["category"]]))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise SystemExit("Fix these briefs:\n  " + "\n  ".join(errors))

    pushed = pick_pushed(scenarios)
    (DATA / "scenarios.json").write_text(json.dumps(scenarios, indent=2) + "\n")
    (DATA / "pushed_items.json").write_text(json.dumps(pushed, indent=2) + "\n")

    for s in scenarios:
        best = s["ranking"][0]
        gap = best["utility"] - s["ranking"][1]["utility"]
        n_bad = sum(not r["feasible"] for r in s["ranking"])
        print(f"{s['id']}: best {best['id']}, gap {gap:.3f}, {n_bad} infeasible, pushed {pushed[s['id']]['product_id']} (rank {pushed[s['id']]['rank']})")
    print(f"Wrote {len(scenarios)} scenarios to data/scenarios.json and data/pushed_items.json")
    n_review = write_review(scenarios, catalog)
    print(f"Wrote {n_review} products to data/catalog_review.csv for hand-checking")


if __name__ == "__main__":
    main()
