#!/usr/bin/env python3
"""
ShrimPK Sustained Daemon Test -- Hebbian + Consolidation Effects

Previous benchmarks restart the daemon between every question, which means:
  - Consolidation never runs (needs 5+ min idle)
  - Hebbian co-activation graph is always empty
  - We've been testing 6/8 features, missing the two that compound over time

This test stores ALL 10 questions' data into ONE daemon instance, lets Hebbian
build and consolidation run, then queries -- measuring the compound effect.

Flow:
  1. Restart daemon ONCE with threshold=0.10
  2. Store ALL turn-pairs from ALL 10 questions into the SAME daemon
  3. Trigger manual consolidation
  4. Pass 1: echo all 10 questions (baseline with consolidation)
  5. Pass 2: echo all 10 questions again (Hebbian should boost co-activated)
  6. Compare Pass 1 vs Pass 2 vs threshold_sweep baseline (7/10 at t=0.10)

No Ollama needed -- pure retrieval quality test.

Usage:
  python sustained_daemon_test.py
"""

import json
import os
import subprocess
import sys
import time
import requests

DAEMON_URL = "http://127.0.0.1:11435"

TEST_IDS = [
    "e47becba", "118b2229", "51a45a95", "6ade9755", "c5e8278d",
    "6f9b354f", "58ef2f1c", "f8c5f88b", "5d3d2817", "7527f7e2",
]

BASELINE_FOUND = 7   # threshold_sweep at t=0.10: 7/10
BASELINE_RATE = 0.70


# ---------------------------------------------------------------------------
# Daemon helpers
# ---------------------------------------------------------------------------

def daemon_healthy():
    try:
        return requests.get(f"{DAEMON_URL}/health", timeout=2).status_code == 200
    except:
        return False


def store_memory(text, source="sustained"):
    try:
        return requests.post(
            f"{DAEMON_URL}/api/store",
            json={"text": text, "source": source},
            timeout=10,
        ).status_code == 200
    except:
        return False


def echo_query(query, max_results=20):
    try:
        r = requests.post(
            f"{DAEMON_URL}/api/echo",
            json={"query": query, "max_results": max_results},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        return {"results": [], "count": 0}
    except:
        return {"results": [], "count": 0}


def trigger_consolidation():
    """Trigger manual consolidation and return the response."""
    try:
        r = requests.post(f"{DAEMON_URL}/api/consolidate", timeout=30)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def get_memory_count():
    """Get total memory count from dump endpoint."""
    try:
        r = requests.get(f"{DAEMON_URL}/api/dump", timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return len(data)
            return data.get("count", data.get("total", 0))
        return -1
    except:
        return -1


def restart_daemon(threshold=0.10):
    """Kill daemon, delete data, restart with specified threshold."""
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = os.path.join(os.path.dirname(__file__), "..", "target", "release", "shrimpk-daemon.exe" if os.name == "nt" else "shrimpk-daemon")

    if not os.path.exists(daemon_bin):
        print(f"  ERROR: daemon binary not found at {daemon_bin}")
        return False

    # Kill existing daemon
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "shrimpk-daemon.exe"],
            capture_output=True, timeout=5,
        )
    except:
        pass
    time.sleep(0.3)

    # Delete data file for clean start
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
        except:
            pass

    # Start daemon with threshold env var
    env = os.environ.copy()
    env["SHRIMPK_SIMILARITY_THRESHOLD"] = str(threshold)
    try:
        subprocess.Popen([daemon_bin], creationflags=0x08000000, env=env)
    except Exception as e:
        print(f"  ERROR starting daemon: {e}")
        return False

    for _ in range(20):
        time.sleep(0.3)
        if daemon_healthy():
            return True
    print("  ERROR: daemon did not start within 6 seconds")
    return False


# ---------------------------------------------------------------------------
# Turn-pair storage (same as v2/threshold_sweep)
# ---------------------------------------------------------------------------

def store_turn_pairs(session, date=None, session_idx=0):
    """Split a session into user+assistant turn-pairs and store each."""
    stored = 0
    turns = list(session)
    i = 0
    p = 0
    while i < len(turns):
        parts = []
        if date:
            parts.append(f"[{date}]")
        turn = turns[i]
        parts.append(f"{turn['role']}: {turn['content']}")
        if turn["role"] == "user" and i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
            parts.append(f"assistant: {turns[i + 1]['content']}")
            i += 2
        else:
            i += 1
        text = "\n".join(parts)
        if len(text.strip()) < 20:
            continue
        if store_memory(text, f"s{session_idx}-p{p}"):
            stored += 1
        p += 1
    return stored


# ---------------------------------------------------------------------------
# Echo pass: query all 10 questions, collect results
# ---------------------------------------------------------------------------

def run_echo_pass(test_items, pass_label):
    """Run echo for all 10 questions, return list of per-question results."""
    results = []
    for qid, item in test_items:
        question = item["question"]
        answer = item["answer"]

        echo_result = echo_query(question, max_results=20)
        raw_results = echo_result.get("results", [])
        count = echo_result.get("count", 0)

        # Check if answer appears in returned results
        all_text = " ".join(r.get("content", "") for r in raw_results)
        answer_found = answer.lower() in all_text.lower()

        # Similarity scores
        sims = [r.get("similarity", 0) for r in raw_results]
        top_sim = sims[0] if sims else 0.0

        # Find rank of the result containing the answer
        rank = -1
        if answer_found:
            for idx, r in enumerate(raw_results):
                if answer.lower() in r.get("content", "").lower():
                    rank = idx + 1
                    break

        results.append({
            "qid": qid,
            "question": question,
            "answer": answer,
            "echo_count": count,
            "answer_found": answer_found,
            "top_sim": top_sim,
            "rank": rank,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_path = "LongMemEval/data/longmemeval_s_cleaned.json"

    # Check dataset exists (try relative and absolute)
    if not os.path.exists(dataset_path):
        alt = os.path.join(os.path.dirname(__file__), "LongMemEval", "data", "longmemeval_s_cleaned.json")
        if os.path.exists(alt):
            dataset_path = alt
        else:
            print(f"ERROR: dataset not found at {dataset_path}")
            sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}

    test_items = [(qid, ref_map[qid]) for qid in TEST_IDS if qid in ref_map]
    missing = [qid for qid in TEST_IDS if qid not in ref_map]
    if missing:
        print(f"WARNING: {len(missing)} question IDs not found: {missing}")
    if not test_items:
        print("ERROR: no test questions found in dataset")
        sys.exit(1)

    # ------------------------------------------------------------------
    print("=" * 72)
    print("SUSTAINED DAEMON TEST -- Hebbian + Consolidation Effects")
    print("=" * 72)
    print(f"Questions: {len(test_items)}")
    print(f"Threshold: 0.10")
    print(f"Baseline (threshold_sweep t=0.10): {BASELINE_FOUND}/10 found ({BASELINE_RATE:.0%})")
    print()

    # ------------------------------------------------------------------
    # Step 1: Restart daemon ONCE
    # ------------------------------------------------------------------
    print("[1/6] Restarting daemon with threshold=0.10 ...")
    if not restart_daemon(threshold=0.10):
        print("FATAL: could not start daemon")
        sys.exit(1)
    print("  Daemon is healthy.")

    # ------------------------------------------------------------------
    # Step 2: Store ALL turn-pairs from ALL 10 questions
    # ------------------------------------------------------------------
    print()
    print("[2/6] Storing ALL turn-pairs from all 10 questions ...")
    total_stored = 0
    total_sessions = 0
    t_store_start = time.time()

    for qid, item in test_items:
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))
        q_stored = 0
        for j, (session, date) in enumerate(zip(sessions, dates)):
            q_stored += store_turn_pairs(session, date, session_idx=total_sessions + j)
        total_sessions += len(sessions)
        total_stored += q_stored
        print(f"  {qid}: {q_stored} turn-pairs from {len(sessions)} sessions")

    t_store_elapsed = time.time() - t_store_start
    print(f"  Total stored: {total_stored} turn-pairs from {total_sessions} sessions "
          f"in {t_store_elapsed:.1f}s")

    # Brief wait for indexing
    time.sleep(1.0)

    # Verify count
    mem_count = get_memory_count()
    print(f"  Memory count from daemon: {mem_count}")

    # ------------------------------------------------------------------
    # Step 3: Trigger consolidation
    # ------------------------------------------------------------------
    print()
    print("[3/6] Triggering manual consolidation ...")
    t_consol_start = time.time()
    consol_result = trigger_consolidation()
    t_consol_elapsed = time.time() - t_consol_start
    print(f"  Consolidation result ({t_consol_elapsed:.1f}s): {json.dumps(consol_result)}")

    # Check memory count after consolidation
    mem_count_after = get_memory_count()
    if mem_count_after >= 0 and mem_count >= 0:
        delta = mem_count - mem_count_after
        print(f"  Memories before: {mem_count}, after: {mem_count_after}, "
              f"delta: {delta}")

    # ------------------------------------------------------------------
    # Step 4: Pass 1 -- echo all 10 questions
    # ------------------------------------------------------------------
    print()
    print("[4/6] PASS 1 -- Echo all 10 questions (post-consolidation) ...")
    pass1 = run_echo_pass(test_items, "Pass 1")

    print()
    print("PASS 1 RESULTS (post-consolidation, before Hebbian co-activation):")
    print("-" * 72)
    print(f"  {'QID':<10} | {'Echo':>4} | {'Found':>5} | {'TopSim':>7} | {'Rank':>4} | Question")
    print(f"  {'-'*10} | {'-'*4} | {'-'*5} | {'-'*7} | {'-'*4} | {'-'*30}")
    for r in pass1:
        found_str = "Y" if r["answer_found"] else "N"
        rank_str = f"#{r['rank']}" if r["rank"] > 0 else "-"
        q_short = r["question"][:30]
        print(f"  {r['qid']:<10} | {r['echo_count']:>4} | {found_str:>5} | "
              f"{r['top_sim']:>7.3f} | {rank_str:>4} | {q_short}")

    pass1_found = sum(1 for r in pass1 if r["answer_found"])
    print(f"\n  Pass 1 found rate: {pass1_found}/{len(pass1)}")

    # ------------------------------------------------------------------
    # Step 5: Pass 2 -- echo again (Hebbian should boost co-activated)
    # ------------------------------------------------------------------
    print()
    print("[5/6] PASS 2 -- Echo all 10 questions again (Hebbian co-activation) ...")
    pass2 = run_echo_pass(test_items, "Pass 2")

    print()
    print("PASS 2 RESULTS (after Hebbian co-activation from Pass 1 queries):")
    print("-" * 72)
    print(f"  {'QID':<10} | {'Echo':>4} | {'Found':>5} | {'TopSim':>7} | {'Rank':>4} | Question")
    print(f"  {'-'*10} | {'-'*4} | {'-'*5} | {'-'*7} | {'-'*4} | {'-'*30}")
    for r in pass2:
        found_str = "Y" if r["answer_found"] else "N"
        rank_str = f"#{r['rank']}" if r["rank"] > 0 else "-"
        q_short = r["question"][:30]
        print(f"  {r['qid']:<10} | {r['echo_count']:>4} | {found_str:>5} | "
              f"{r['top_sim']:>7.3f} | {rank_str:>4} | {q_short}")

    pass2_found = sum(1 for r in pass2 if r["answer_found"])
    print(f"\n  Pass 2 found rate: {pass2_found}/{len(pass2)}")

    # ------------------------------------------------------------------
    # Step 6: Comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("[6/6] COMPARISON")
    print("=" * 72)

    # Per-question diff
    print()
    print(f"  {'QID':<10} | {'P1 Found':>8} | {'P2 Found':>8} | "
          f"{'P1 TopSim':>9} | {'P2 TopSim':>9} | {'P1 Rank':>7} | {'P2 Rank':>7} | Delta")
    print(f"  {'-'*10} | {'-'*8} | {'-'*8} | "
          f"{'-'*9} | {'-'*9} | {'-'*7} | {'-'*7} | {'-'*12}")

    improvements = 0
    regressions = 0
    sim_deltas = []

    for r1, r2 in zip(pass1, pass2):
        f1 = "Y" if r1["answer_found"] else "N"
        f2 = "Y" if r2["answer_found"] else "N"
        rk1 = f"#{r1['rank']}" if r1["rank"] > 0 else "-"
        rk2 = f"#{r2['rank']}" if r2["rank"] > 0 else "-"
        sim_delta = r2["top_sim"] - r1["top_sim"]
        sim_deltas.append(sim_delta)

        # Determine delta label
        if r2["answer_found"] and not r1["answer_found"]:
            delta = "+FOUND"
            improvements += 1
        elif r1["answer_found"] and not r2["answer_found"]:
            delta = "-LOST"
            regressions += 1
        elif r2["rank"] > 0 and r1["rank"] > 0 and r2["rank"] < r1["rank"]:
            delta = f"+RANK({r1['rank']}->{r2['rank']})"
        elif r2["rank"] > 0 and r1["rank"] > 0 and r2["rank"] > r1["rank"]:
            delta = f"-RANK({r1['rank']}->{r2['rank']})"
        elif abs(sim_delta) > 0.001:
            delta = f"sim {sim_delta:+.3f}"
        else:
            delta = "="

        print(f"  {r1['qid']:<10} | {f1:>8} | {f2:>8} | "
              f"{r1['top_sim']:>9.3f} | {r2['top_sim']:>9.3f} | "
              f"{rk1:>7} | {rk2:>7} | {delta}")

    # Summary
    avg_sim_delta = sum(sim_deltas) / len(sim_deltas) if sim_deltas else 0
    hebbian_improvement = pass2_found - pass1_found
    sustained_vs_baseline = pass1_found - BASELINE_FOUND

    print()
    print("-" * 72)
    print("SUMMARY:")
    print(f"  Total memories stored:            {total_stored} turn-pairs")
    print(f"  Memories after consolidation:     {mem_count_after}")
    print()
    print(f"  Pass 1 found rate:                {pass1_found}/{len(pass1)} "
          f"({100*pass1_found/len(pass1):.0f}%)")
    print(f"  Pass 2 found rate:                {pass2_found}/{len(pass2)} "
          f"({100*pass2_found/len(pass2):.0f}%)")
    print(f"  Baseline (threshold_sweep t=0.10): {BASELINE_FOUND}/10 "
          f"({BASELINE_RATE:.0%})")
    print()
    print(f"  Consolidation effect (P1 vs baseline): "
          f"{sustained_vs_baseline:+d} answers")
    print(f"  Hebbian effect (P2 vs P1):             "
          f"{hebbian_improvement:+d} answers")
    print(f"  Combined effect (P2 vs baseline):      "
          f"{pass2_found - BASELINE_FOUND:+d} answers")
    print(f"  Avg similarity delta (P2 - P1):        "
          f"{avg_sim_delta:+.4f}")
    print(f"  Questions improved P1->P2:              {improvements}")
    print(f"  Questions regressed P1->P2:             {regressions}")
    print()

    if pass2_found > BASELINE_FOUND:
        print(f"  VERDICT: Sustained daemon IMPROVES retrieval "
              f"({pass2_found}/10 vs {BASELINE_FOUND}/10 baseline)")
    elif pass2_found == BASELINE_FOUND:
        print(f"  VERDICT: Sustained daemon matches baseline "
              f"({pass2_found}/10 = {BASELINE_FOUND}/10)")
    else:
        print(f"  VERDICT: Sustained daemon UNDERPERFORMS baseline "
              f"({pass2_found}/10 vs {BASELINE_FOUND}/10)")
        print(f"  NOTE: Cross-question interference may be hurting retrieval")

    if hebbian_improvement > 0:
        print(f"  HEBBIAN: Co-activation boosted {hebbian_improvement} additional answer(s)")
    elif hebbian_improvement == 0:
        print(f"  HEBBIAN: No measurable effect between passes")
    else:
        print(f"  HEBBIAN: Negative effect ({hebbian_improvement}) -- investigate")

    # Save results
    os.makedirs("results", exist_ok=True)
    results_data = {
        "test": "sustained_daemon",
        "threshold": 0.10,
        "total_stored": total_stored,
        "mem_after_consolidation": mem_count_after,
        "consolidation_result": consol_result,
        "pass1": pass1,
        "pass2": pass2,
        "pass1_found": pass1_found,
        "pass2_found": pass2_found,
        "baseline_found": BASELINE_FOUND,
        "hebbian_improvement": hebbian_improvement,
        "avg_sim_delta": avg_sim_delta,
    }
    results_path = "results/sustained_daemon_test.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
