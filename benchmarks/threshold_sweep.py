#!/usr/bin/env python3
"""
ShrimPK Cosine Threshold Sweep — find the optimal similarity_threshold.

Tests a range of thresholds [0.05 - 0.45] against 10 questions.
No Ollama needed — this is pure retrieval quality testing.

For each threshold:
  1. Restart daemon with SHRIMPK_SIMILARITY_THRESHOLD=X
  2. Store sessions, run echo, measure retrieval quality

Usage:
  python threshold_sweep.py
"""

import json
import os
import subprocess
import sys
import time
import requests

DAEMON_URL = "http://127.0.0.1:11435"

THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.14, 0.18, 0.22, 0.28, 0.35, 0.45]

TEST_IDS = [
    "e47becba", "118b2229", "51a45a95", "6ade9755", "c5e8278d",
    "6f9b354f", "58ef2f1c", "f8c5f88b", "5d3d2817", "7527f7e2",
]


def daemon_healthy():
    try:
        return requests.get(f"{DAEMON_URL}/health", timeout=2).status_code == 200
    except:
        return False


def store_memory(text, source="sweep"):
    try:
        return requests.post(f"{DAEMON_URL}/api/store", json={"text": text, "source": source}, timeout=10).status_code == 200
    except:
        return False


def echo_query(query, max_results=20):
    try:
        r = requests.post(f"{DAEMON_URL}/api/echo", json={"query": query, "max_results": max_results}, timeout=10)
        return r.json() if r.status_code == 200 else {"results": [], "count": 0}
    except:
        return {"results": [], "count": 0}


def restart_daemon_with_threshold(threshold):
    """Kill daemon, delete data, restart with new threshold via env var."""
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = os.path.join(os.path.dirname(__file__), "..", "target", "release", "shrimpk-daemon.exe" if os.name == "nt" else "shrimpk-daemon")

    # Kill
    try:
        subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"], capture_output=True, timeout=5)
    except:
        pass
    time.sleep(0.3)

    # Delete data
    if os.path.exists(data_file):
        try: os.remove(data_file)
        except: pass

    # Restart with threshold env var
    env = os.environ.copy()
    env["SHRIMPK_SIMILARITY_THRESHOLD"] = str(threshold)
    try:
        if sys.platform == "win32":
            subprocess.Popen([daemon_bin], creationflags=0x08000000, env=env)
        else:
            subprocess.Popen([daemon_bin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    except Exception as e:
        print(f"  ERROR starting daemon: {e}")
        return False

    for _ in range(20):
        time.sleep(0.3)
        if daemon_healthy(): return True
    return False


def store_session_as_turn_pairs(session, date=None, session_idx=0):
    stored = 0
    turns = list(session)
    i = 0
    pair_idx = 0
    while i < len(turns):
        parts = []
        if date: parts.append(f"[{date}]")
        turn = turns[i]
        parts.append(f"{turn['role']}: {turn['content']}")
        if turn["role"] == "user" and i + 1 < len(turns) and turns[i+1]["role"] == "assistant":
            parts.append(f"assistant: {turns[i+1]['content']}")
            i += 2
        else:
            i += 1
        text = "\n".join(parts)
        if len(text.strip()) < 20: continue
        if store_memory(text, f"s{session_idx}-p{pair_idx}"): stored += 1
        pair_idx += 1
    return stored


def main():
    dataset_path = "LongMemEval/data/longmemeval_s_cleaned.json"
    with open(dataset_path, encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}

    test_items = [(qid, ref_map[qid]) for qid in TEST_IDS if qid in ref_map]

    print("=" * 90)
    print("COSINE THRESHOLD SWEEP — Finding the optimal similarity_threshold")
    print("=" * 90)
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Questions: {len(test_items)}")
    print()

    # Collect results per threshold
    sweep_results = []

    for threshold in THRESHOLDS:
        print(f"\n{'-'*70}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'-'*70}")

        total_results = 0
        total_answer_found = 0
        total_questions = 0
        all_sims = []
        per_question = []

        for qid, item in test_items:
            question = item["question"]
            answer = item["answer"]
            sessions = item["haystack_sessions"]
            dates = item.get("haystack_dates", [None] * len(sessions))

            # Restart with this threshold + fresh data
            restart_daemon_with_threshold(threshold)

            # Store turn-pairs
            stored = 0
            for j, (session, date) in enumerate(zip(sessions, dates)):
                stored += store_session_as_turn_pairs(session, date, session_idx=j)

            # Echo
            echo_result = echo_query(question, max_results=20)
            results = echo_result.get("results", [])
            count = echo_result.get("count", 0)

            # Check if answer is in results
            all_text = " ".join(r.get("content", "") for r in results)
            answer_found = answer.lower() in all_text.lower()

            # Collect similarity scores
            sims = [r.get("similarity", 0) for r in results]
            avg_sim = sum(sims) / len(sims) if sims else 0
            all_sims.extend(sims)

            total_results += count
            total_answer_found += 1 if answer_found else 0
            total_questions += 1

            status = "FOUND" if answer_found else "MISS"
            print(f"  {qid} | {count:2d} results | avg_sim={avg_sim:.3f} | {status} | {question[:40]}")

            per_question.append({
                "qid": qid,
                "count": count,
                "answer_found": answer_found,
                "avg_sim": avg_sim,
                "top_sim": sims[0] if sims else 0,
            })

        avg_results = total_results / max(total_questions, 1)
        answer_rate = total_answer_found / max(total_questions, 1)
        overall_avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0

        sweep_results.append({
            "threshold": threshold,
            "avg_results": avg_results,
            "answer_found_rate": answer_rate,
            "overall_avg_sim": overall_avg_sim,
            "total_found": total_answer_found,
            "total_questions": total_questions,
            "per_question": per_question,
        })

    # Final summary table
    print(f"\n{'='*90}")
    print("THRESHOLD SWEEP RESULTS")
    print(f"{'='*90}")
    print(f"{'Threshold':>10} {'Avg Results':>12} {'Answer Found':>13} {'Found Rate':>11} {'Avg Sim':>8}")
    print(f"{'-'*10} {'-'*12} {'-'*13} {'-'*11} {'-'*8}")

    best_threshold = None
    best_rate = -1

    for sr in sweep_results:
        t = sr["threshold"]
        ar = sr["avg_results"]
        af = sr["total_found"]
        tq = sr["total_questions"]
        rate = sr["answer_found_rate"]
        avg_s = sr["overall_avg_sim"]

        marker = ""
        if rate > best_rate:
            best_rate = rate
            best_threshold = t

        print(f"{t:>10.2f} {ar:>12.1f} {af:>8d}/{tq:<4d} {rate:>10.0%} {avg_s:>8.3f}")

    # Find sweet spot: highest answer_found_rate, break ties with fewer results (precision)
    candidates = [sr for sr in sweep_results if sr["answer_found_rate"] == best_rate]
    if len(candidates) > 1:
        # Among ties, pick the one with fewest avg results (higher precision)
        best = min(candidates, key=lambda x: x["avg_results"])
        best_threshold = best["threshold"]

    print(f"\n{'-'*90}")
    print(f"RECOMMENDATION: threshold = {best_threshold}")
    print(f"  Answer found rate: {best_rate:.0%}")
    print(f"  Current default: 0.14")
    if best_threshold != 0.14:
        print(f"  Change: 0.14 → {best_threshold}")
    else:
        print(f"  Current default is already optimal!")

    # Save detailed results
    results_path = "results/threshold_sweep.json"
    os.makedirs("results", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")


if __name__ == "__main__":
    main()
