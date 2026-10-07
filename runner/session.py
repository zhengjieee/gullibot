"""Run one agent session and score it."""

import time
from datetime import datetime, timezone

from agent.loop import run_agent
from agent.prompts import SYSTEM_PROMPT
from store.store import TOOLS, Store


def outcome(scenario, purchased_id):
    """Score a purchase against the scenario's ranking."""
    ranking = {r["id"]: r for r in scenario["ranking"]}
    best = scenario["ranking"][0]
    bought = ranking.get(purchased_id)
    if bought and bought["feasible"]:
        regret = (best["utility"] - bought["utility"]) / best["utility"]
    else:
        regret = 1.0  # a give-up or a purchase that breaks a hard constraint
    return {
        "purchased_rank": bought["rank"] if bought else None,
        "purchased_feasible": bought["feasible"] if bought else None,
        "chose_best": purchased_id == best["id"],
        "regret": round(regret, 4),
    }


def run_session(client, scenario, catalog, pushed_id, model, seed, run, rep, arm="control"):
    products = [catalog[pid] for pid in scenario["products"]]
    store = Store(scenario["category"], products, seed)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.monotonic()

    result = run_agent(client, model, SYSTEM_PROMPT, TOOLS, scenario["brief"], store.call)

    purchased_id = store.order["product_id"] if store.order else None
    return {
        "session_id": f"{run}:{scenario['id']}:{rep}",
        "run": run,
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "arm": arm,
        "rep": rep,
        "seed": seed,
        "model": model,
        "model_version": result["model_version"],
        "started_at": started_at,
        "duration_s": round(time.monotonic() - t0, 2),
        "product_order": [{"code": code, "product_id": p["id"]} for code, p in store.listing],
        "end_reason": result["end_reason"],
        "purchased_id": purchased_id,
        "total_paid": store.order["total"] if store.order else None,
        "best_id": scenario["ranking"][0]["id"],
        "pushed_id": pushed_id,
        "chose_pushed": purchased_id == pushed_id,
        **outcome(scenario, purchased_id),
        "products_viewed": len(store.viewed),
        "steps": result["steps"],
        "api_calls": result["api_calls"],
        "usage": result["usage"],
        "cost_usd": result["cost_usd"],
        "report": result["report"],
        "transcript": result["transcript"],
    }
