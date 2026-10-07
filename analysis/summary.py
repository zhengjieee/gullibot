"""Summarize a session log: competence, outcomes, process and cost.

    python -m analysis.summary logs/control-claude-haiku-4-5.jsonl
"""

import json
import math
import sys

import pandas as pd

CHANCE = 1 / 8


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load(path):
    """One row per session; a session rerun after an error keeps only its latest line."""
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "transcript"} for r in rows])
    return df.drop_duplicates("session_id", keep="last")


def rate(df, col):
    k, n = int(df[col].sum()), len(df)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} (95% CI {lo:.0%}-{hi:.0%})"


def print_summary(path):
    df = load(path)
    errors = df[df["end_reason"] == "error"]
    df = df[df["end_reason"] != "error"].copy()
    print(f"\n== {path} ==")
    print(f"sessions: {len(df)} ok, {len(errors)} errors; model version(s): {', '.join(df['model_version'].dropna().unique())}")
    if df.empty:
        return

    print(f"\nCompetence (bought the best item; chance = {CHANCE:.1%}): {rate(df, 'chose_best')}")
    for cat, g in df.groupby("category"):
        print(f"  {cat:14s} {rate(g, 'chose_best')}")

    bought = df["purchased_id"].notna()
    print("\nOutcomes:")
    print(f"  purchased {bought.sum()}, no purchase {(~bought).sum()} (end reasons: {df.loc[~bought, 'end_reason'].value_counts().to_dict()})")
    print(f"  rank bought: {df['purchased_rank'].value_counts().sort_index().to_dict()}")
    print(f"  infeasible purchases: {int((df['purchased_feasible'] == False).sum())}")  # noqa: E712
    print(f"  bought pushed item (control baseline): {rate(df, 'chose_pushed')}")
    print(f"  mean regret: {df['regret'].mean():.3f}")

    print("\nProcess and cost (mean per session):")
    print(f"  steps {df['steps'].mean():.1f}, products viewed {df['products_viewed'].mean():.1f}, "
          f"API calls {df['api_calls'].mean():.1f}, {df['duration_s'].mean():.0f} s")
    print(f"  cost ${df['cost_usd'].mean():.4f} per session, ${df['cost_usd'].sum():.2f} total")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print_summary(p)
