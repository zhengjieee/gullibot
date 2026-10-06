"""Extract structured attributes from candidate listings with Claude Haiku 4.5.

Reads data/raw/<category>.jsonl, pre-filters to the product type we want,
picks a price-spread sample, and asks the model to fill a fixed schema per
category. Results are cached in data/raw/extracted_<category>.jsonl, so a
rerun only pays for items not yet extracted.

    python -m prep.extract            # all categories
    python -m prep.extract laptops    # one category
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from prep.config import CATEGORIES, RAW, ROOT

EXTRACT_MODEL = "claude-haiku-4-5"
SAMPLE_PER_CATEGORY = 100  # price-spread sample per category, plus BOOST items
MAX_PER_BRAND = 6
WORKERS = 8

DISPLAY_NAME = Field(
    description="Short product name: brand plus model, at most 8 words. "
    "No specs, numbers of hours/GB/cups, or marketing claims."
)
IS_TARGET = "True only if the listing is a single new {what}, not an accessory, part, bundle or multi-pack."


class Headphones(BaseModel):
    is_target: bool = Field(description=IS_TARGET.format(what="pair of wireless headphones or earbuds"))
    display_name: str = DISPLAY_NAME
    form_factor: Literal["over_ear", "on_ear", "in_ear"]
    noise_cancelling: bool = Field(description="True only if the listing says it has active noise cancellation (ANC).")
    battery_hours: Optional[float] = Field(
        description="Playback hours on one full charge of the headphones themselves, not counting a charging case. Null if not stated."
    )
    weight_g: Optional[float] = Field(description="Weight of the headphones in grams (convert from ounces). Null if not stated.")


class Laptop(BaseModel):
    is_target: bool = Field(description=IS_TARGET.format(what="laptop") + " Refurbished or renewed laptops are not targets.")
    display_name: str = DISPLAY_NAME
    ram_gb: Optional[int] = Field(description="Installed RAM in GB. Null if not stated.")
    storage_gb: Optional[int] = Field(description="Main storage in GB (1 TB = 1024). Null if not stated.")
    screen_in: Optional[float] = Field(description="Screen diagonal in inches. Null if not stated.")
    weight_lb: Optional[float] = Field(description="Laptop weight in pounds. Null if not stated.")


class CoffeeMaker(BaseModel):
    is_target: bool = Field(
        description=IS_TARGET.format(what="drip coffee maker that brews a carafe")
        + " Espresso machines, pod/single-serve brewers, pour-overs, French presses and percolators are not targets."
    )
    display_name: str = DISPLAY_NAME
    capacity_cups: Optional[int] = Field(description="Carafe capacity in cups. Null if not stated.")
    programmable: bool = Field(description="True if it has a programmable timer or delayed brew.")
    thermal_carafe: bool = Field(description="True if the carafe is thermal or insulated stainless steel, false for glass.")
    brew_strength_control: bool = Field(description="True if the user can choose brew strength (e.g. regular/bold).")


SCHEMAS = {"headphones": Headphones, "laptops": Laptop, "coffee_makers": CoffeeMaker}

SYSTEM = (
    "You extract product specifications from Amazon listings. Use only facts stated in the listing. "
    "When a number is not stated, return null rather than guessing. A boolean is true only when the "
    "listing clearly says the product has that feature."
)

DETAIL_SKIP = {"Best Sellers Rank", "Date First Available", "Is Discontinued By Manufacturer", "Item model number"}


def prefilter(name, items):
    """Cheap title/details rules that drop obvious non-targets before paying for extraction."""
    out = []
    for it in items:
        title = it["title"].lower()
        details = it.get("details") or {}
        if name == "headphones":
            conn = str(details.get("Connectivity Technology", "")).lower()
            wireless = "wireless" in conn or "bluetooth" in conn or "wireless" in title or "bluetooth" in title
            if not wireless or it["price"] < 15 or "kids" in title:
                continue
        elif name == "laptops":
            if re.search(r"renewed|refurbished", title):
                continue
        elif name == "coffee_makers":
            if not re.search(r"\bdrip\b|coffee maker|coffee machine", title):
                continue
            if re.search(r"espresso|pour.?over|french press|percolat|cold brew|k-cup|single.?serve|\bpods?\b|moka|turkish", title):
                continue
        out.append(it)
    return out


# Product types the briefs need but a price-spread sample under-covers
# (most cheap listings are earbuds and glass-carafe machines). Every
# candidate matching these is extracted on top of the sample.
BOOST = {
    "headphones": lambda it: bool(set(it["categories"]) & {"Over-Ear Headphones", "On-Ear Headphones"}),
    "coffee_makers": lambda it: bool(re.search(r"thermal|insulated", (it["title"] + " ".join(it["features"])).lower())),
}


def price_spread_sample(items, n):
    """Take n items spread evenly over the price range, at most MAX_PER_BRAND per brand, no duplicate titles."""
    items = sorted(items, key=lambda it: it["price"])
    seen_titles, per_brand, picked = set(), {}, []
    step = max(1, len(items) / n)
    # first pass walks evenly spaced positions, second pass fills any gaps in order
    order = [items[int(i * step)] for i in range(min(n, len(items)))] + items
    for it in order:
        key = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:40]
        brand = (it.get("store") or "").lower()
        if key in seen_titles or per_brand.get(brand, 0) >= MAX_PER_BRAND:
            continue
        seen_titles.add(key)
        per_brand[brand] = per_brand.get(brand, 0) + 1
        picked.append(it)
        if len(picked) == n:
            break
    return picked


def listing_text(it):
    details = {k: v for k, v in (it.get("details") or {}).items() if k not in DETAIL_SKIP}
    description = " ".join(it.get("description") or [])[:1500]
    return "\n".join([
        f"Title: {it['title']}",
        f"Brand: {it.get('store')}",
        "Features:",
        *[f"- {f}" for f in it["features"]],
        f"Description: {description}",
        f"Details: {json.dumps(details, ensure_ascii=False)}",
    ])


def extract_one(client, name, it):
    response = client.messages.parse(
        model=EXTRACT_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": listing_text(it)}],
        output_format=SCHEMAS[name],
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"{it['parent_asin']}: stop_reason={response.stop_reason}")
    return {
        "parent_asin": it["parent_asin"],
        "attributes": response.parsed_output.model_dump(),
        "usage": {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
    }


def run_category(client, name):
    items = [json.loads(line) for line in (RAW / f"{name}.jsonl").open()]
    candidates = prefilter(name, items)
    sample = price_spread_sample(candidates, SAMPLE_PER_CATEGORY)
    in_sample = {it["parent_asin"] for it in sample}
    boost = BOOST.get(name, lambda it: False)
    sample += [it for it in candidates if boost(it) and it["parent_asin"] not in in_sample]

    out_path = RAW / f"extracted_{name}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["parent_asin"] for line in out_path.open()}
    todo = [it for it in sample if it["parent_asin"] not in done]
    print(f"{name}: {len(items)} candidates, {len(sample)} sampled, {len(todo)} to extract")

    tokens_in = tokens_out = failures = 0
    with out_path.open("a") as f, ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(extract_one, client, name, it) for it in todo]
        for fut in futures:
            try:
                row = fut.result()
            except (anthropic.APIError, RuntimeError) as e:
                failures += 1
                print(f"  failed: {e}")
                continue
            f.write(json.dumps(row) + "\n")
            tokens_in += row["usage"]["input"]
            tokens_out += row["usage"]["output"]
    cost = tokens_in / 1e6 * 1.00 + tokens_out / 1e6 * 5.00  # Haiku 4.5 list price
    print(f"  done: {tokens_in:,} input / {tokens_out:,} output tokens, about ${cost:.2f}, {failures} failures")


def main():
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic(max_retries=4)
    names = sys.argv[1:] or list(CATEGORIES)
    for name in names:
        run_category(client, name)


if __name__ == "__main__":
    main()
