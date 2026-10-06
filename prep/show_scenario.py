"""Print a scenario for hand-checking.

Shows the brief and its 8 products without the answer, so you can rank them
yourself first. Add --answer to see the script's ranking and the pushed item.

    python -m prep.show_scenario S01
    python -m prep.show_scenario S01 --answer
"""

import json
import sys

from prep.config import DATA
from prep.scoring import ATTRIBUTES


def main():
    sid, show_answer = sys.argv[1], "--answer" in sys.argv
    scenario = next(s for s in json.loads((DATA / "scenarios.json").read_text()) if s["id"] == sid)
    catalog = {p["id"]: p for p in json.loads((DATA / "catalog.json").read_text())[scenario["category"]]}
    attrs = [a for a in ATTRIBUTES[scenario["category"]] if a not in ("price", "rating")]

    print(f"{sid} ({scenario['category']})\n{scenario['brief']}\n")
    header = ["id", "name", "price", "rating"] + attrs
    rows = []
    for pid in scenario["products"]:
        p = catalog[pid]
        rows.append([pid, p["name"][:34], f"${p['price']:.2f}", str(p["rating"])] + [str(p["attributes"][a]) for a in attrs])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    for row in [header] + rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))

    if show_answer:
        pushed = json.loads((DATA / "pushed_items.json").read_text())[sid]
        print("\nRanking:")
        for r in scenario["ranking"]:
            u = "infeasible" if r["utility"] is None else f"{r['utility']:.3f}"
            mark = "  <- pushed" if r["id"] == pushed["product_id"] else ""
            print(f"  {r['rank']}. {r['id']}  {u}{mark}")


if __name__ == "__main__":
    main()
