"""Turn extracted listings into the final catalog.

Keeps every item the extractor marked as a target that has all scored
attributes inside a sane range. Writes data/catalog.json, products ordered
by price within each category.

    python -m prep.build_catalog
"""

import json
import re

from prep.config import CATEGORIES, DATA, RAW
from prep.scoring import ATTRIBUTES

ID_PREFIX = {"headphones": "hp", "laptops": "lt", "coffee_makers": "cm"}

# Plausible ranges; anything outside is treated as an extraction error.
RANGES = {
    "headphones": {"price": (15, 400), "battery_hours": (2, 80), "weight_g": (3, 450)},
    "laptops": {"price": (150, 2500), "ram_gb": (2, 128), "storage_gb": (32, 8192),
                "screen_in": (10, 18.5), "weight_lb": (1.5, 12)},
    "coffee_makers": {"price": (15, 400), "capacity_cups": (4, 14)},
}

# A product name must not claim a feature its attributes say it lacks.
NAME_CLAIMS = {
    "noise_cancelling": r"noise.?cancel|\banc\b",
    "programmable": r"programmable",
    "thermal_carafe": r"thermal",
}


def clean(name, raw, extracted):
    attrs = dict(extracted["attributes"])
    if not attrs.pop("is_target"):
        return None
    product = {
        "category": name,
        "asin": raw["parent_asin"],
        "name": attrs.pop("display_name").strip(),
        "brand": raw.get("store"),
        "price": round(raw["price"], 2),
        "rating": raw["average_rating"],
        "rating_count": raw["rating_number"],
        "attributes": {k: attrs[k] for k in ATTRIBUTES[name] if k in attrs},
        "source_title": raw["title"],
    }
    for attr, (lo, hi) in RANGES[name].items():
        v = product[attr] if attr == "price" else product["attributes"].get(attr)
        if v is None or not lo <= v <= hi:
            return None
    if any(v is None for v in product["attributes"].values()):
        return None
    return product


def name_conflicts(product):
    return [
        attr for attr, pattern in NAME_CLAIMS.items()
        if product["attributes"].get(attr) is False and re.search(pattern, product["name"].lower())
    ]


def main():
    catalog = {}
    for name in CATEGORIES:
        raw = {json.loads(line)["parent_asin"]: json.loads(line) for line in (RAW / f"{name}.jsonl").open()}
        extracted = [json.loads(line) for line in (RAW / f"extracted_{name}.jsonl").open()]
        cleaned = [p for e in extracted if (p := clean(name, raw[e["parent_asin"]], e))]
        kept = []
        for p in cleaned:
            if conflicts := name_conflicts(p):
                print(f"  dropped {p['name']!r}: name claims {', '.join(conflicts)}")
            else:
                kept.append(p)
        kept.sort(key=lambda p: p["price"])
        catalog[name] = [{"id": f"{ID_PREFIX[name]}-{i:03d}", **p} for i, p in enumerate(kept, 1)]
        print(f"{name}: {len(extracted)} extracted, {len(kept)} in catalog")

    (DATA / "catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    print("Wrote data/catalog.json")


if __name__ == "__main__":
    main()
