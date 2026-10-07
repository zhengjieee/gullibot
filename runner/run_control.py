"""Week 2: run control sessions (plain store, no dark patterns).

Every scenario runs `--reps` times on one model. Sessions run in a shuffled
order and are appended to logs/<run>.jsonl, one line per session; a rerun
skips sessions already logged, so an interrupted batch can be resumed.

    python -m runner.run_control --model claude-haiku-4-5
    python -m runner.run_control --model claude-sonnet-5 --scenarios S01 S11
"""

import argparse
import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import httpx2
from dotenv import load_dotenv

from agent.models import MODELS
from analysis.summary import print_summary
from prep.config import ROOT
from runner.data import load_catalog, load_pushed, load_scenarios
from runner.session import run_session

LOGS = ROOT / "logs"


def make_client():
    """Some requests hung with no response until the timeout, then succeeded on retry.

    Turning off connection reuse avoids sending on a stale pooled connection,
    and a 60-second timeout (agent turns normally take under 15) retries any
    remaining hang quickly instead of after the SDK's 10-minute default.
    """
    http = anthropic.DefaultHttpxClient(limits=httpx2.Limits(max_connections=50, max_keepalive_connections=0))
    return anthropic.Anthropic(max_retries=5, timeout=anthropic.Timeout(60.0, connect=10.0), http_client=http)


def session_seed(scenario_id, rep):
    """Same seed for a scenario and repetition on every model and arm, so their product orders match."""
    return int(hashlib.sha256(f"{scenario_id}-{rep}".encode()).hexdigest()[:8], 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5", choices=sorted(MODELS))
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run", help="run name; default control-<model>")
    parser.add_argument("--scenarios", nargs="*", help="only these scenario ids")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = make_client()
    run = args.run or f"control-{args.model}"
    out_path = LOGS / f"{run}.jsonl"
    LOGS.mkdir(exist_ok=True)

    catalog, pushed = load_catalog(), load_pushed()
    scenarios = [s for s in load_scenarios() if not args.scenarios or s["id"] in args.scenarios]
    done = set()  # sessions that finished; errored ones run again and their new line supersedes the old
    if out_path.exists():
        done = {r["session_id"] for r in map(json.loads, out_path.open()) if r["end_reason"] != "error"}

    jobs = [(s, rep) for s in scenarios for rep in range(args.reps) if f"{run}:{s['id']}:{rep}" not in done]
    random.Random(run).shuffle(jobs)
    print(f"{run}: {len(jobs)} sessions to run ({len(done)} already logged) -> {out_path.relative_to(ROOT)}")

    lock = threading.Lock()
    out_of_credit = threading.Event()

    def work(scenario, rep):
        if out_of_credit.is_set():
            return None
        try:
            return run_session(client, scenario, catalog, pushed[scenario["id"]], args.model,
                               session_seed(scenario["id"], rep), run, rep)
        except anthropic.APIError as e:
            if "credit balance" in str(e):
                out_of_credit.set()  # every later request would fail the same way
            return {"session_id": f"{run}:{scenario['id']}:{rep}", "run": run, "scenario_id": scenario["id"],
                    "rep": rep, "model": args.model, "end_reason": "error", "error": repr(e)}

    with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, s, rep) for s, rep in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is None:
                continue  # skipped after running out of credit
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if rec["end_reason"] == "error":
                print(f"  {rec['session_id']}: ERROR {rec['error'][:120]}")
            else:
                bought = f"bought rank {rec['purchased_rank']}" if rec["purchased_id"] else "no purchase"
                print(f"  {rec['session_id']}: {bought}, {rec['steps']} steps, ${rec['cost_usd']:.3f}, {rec['end_reason']}")

    if out_of_credit.is_set():
        print("\nStopped: the API account is out of credit. Top up at console.anthropic.com, then rerun this "
              "command to finish the remaining sessions.")
    print_summary(out_path)


if __name__ == "__main__":
    main()
