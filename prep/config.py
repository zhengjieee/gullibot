"""Shared paths and category definitions for the data-prep scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"  # streamed candidates, not committed

HF_BASE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_{}.jsonl"
)

# Which Amazon file each category comes from, and which category-path
# entries mark an item as belonging to it.
CATEGORIES = {
    "headphones": {
        "source": "Electronics",
        "leaves": {"Earbud Headphones", "Over-Ear Headphones", "On-Ear Headphones"},
    },
    "laptops": {
        "source": "Electronics",
        "leaves": {"Traditional Laptops", "2 in 1 Laptops"},
    },
    "coffee_makers": {
        "source": "Home_and_Kitchen",
        "leaves": {"Coffee Machines"},
    },
}

# Candidate filters applied while streaming.
MIN_FEATURES = 4
MIN_RATINGS = 20
MAX_CANDIDATES = 600  # per category; streaming stops once every category has this many

