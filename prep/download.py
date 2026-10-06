"""Stream Amazon Reviews 2023 item metadata and keep candidate products.

The source files are 5-12 GB, so we read them line by line over HTTP and
stop as soon as every category has enough candidates. Output goes to
data/raw/<category>.jsonl.

    python -m prep.download
"""

import json
import urllib.request
from collections import defaultdict

from prep.config import CATEGORIES, HF_BASE, MAX_CANDIDATES, MIN_FEATURES, MIN_RATINGS, RAW

KEEP_FIELDS = [
    "parent_asin", "title", "store", "price", "average_rating",
    "rating_number", "features", "description", "details", "categories",
]


def match_category(item, names):
    """Return the first category name whose leaves appear in the item's category path."""
    path = set(item.get("categories") or [])
    for name in names:
        if path & CATEGORIES[name]["leaves"]:
            return name
    return None


def is_candidate(item):
    price = item.get("price")
    return (
        isinstance(price, (int, float))
        and price > 0
        and len(item.get("features") or []) >= MIN_FEATURES
        and (item.get("rating_number") or 0) >= MIN_RATINGS
        and item.get("title")
    )


def stream_source(source, names):
    url = HF_BASE.format(source)
    kept = defaultdict(list)
    print(f"Streaming {source} for {', '.join(names)}")
    with urllib.request.urlopen(url) as resp:
        for n, line in enumerate(resp, 1):
            item = json.loads(line)
            name = match_category(item, names)
            if name and len(kept[name]) < MAX_CANDIDATES and is_candidate(item):
                kept[name].append({k: item.get(k) for k in KEEP_FIELDS})
            if n % 200_000 == 0:
                counts = ", ".join(f"{k}={len(kept[k])}" for k in names)
                print(f"  {n:,} lines read: {counts}")
            if all(len(kept[k]) >= MAX_CANDIDATES for k in names):
                break
    return kept


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    by_source = defaultdict(list)
    for name, spec in CATEGORIES.items():
        by_source[spec["source"]].append(name)

    for source, names in by_source.items():
        kept = stream_source(source, names)
        for name in names:
            path = RAW / f"{name}.jsonl"
            with path.open("w") as f:
                for item in kept[name]:
                    f.write(json.dumps(item) + "\n")
            print(f"Wrote {len(kept[name])} candidates to {path.relative_to(RAW.parent.parent)}")


if __name__ == "__main__":
    main()
