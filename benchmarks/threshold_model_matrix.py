#!/usr/bin/env python3
"""
ShrimPK Threshold x Model Matrix Benchmark

Tests 4 thresholds x 4 models x 10 questions = 160 LLM calls.
Optimization: retrieval is done once per threshold x question (40 cycles),
then each model gets the same pre-built context (160 Ollama calls, 0 redundant daemon work).

Usage:
  python threshold_model_matrix.py
  python threshold_model_matrix.py --resume   # continue from checkpoint
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests

DAEMON_URL = "http://127.0.0.1:11435"
OLLAMA_URL = "http://127.0.0.1:11434"

THRESHOLDS = [0.08, 0.10, 0.14, 0.22]
MODELS = ["gemma3:1b", "llama3.2:3b", "qwen2.5:7b", "gemma3:12b"]

TEST_IDS = [
    "e47becba", "118b2229", "51a45a95", "6ade9755", "c5e8278d",
    "6f9b354f", "58ef2f1c", "f8c5f88b", "5d3d2817", "7527f7e2",
]

CHECKPOINT_FILE = "results/matrix_checkpoint.json"
RESULTS_FILE = "results/matrix_results.json"


def daemon_healthy():
    try:
        return requests.get(f"{DAEMON_URL}/health", timeout=2).status_code == 200
    except:
        return False


def store_memory(text, source="matrix"):
    try:
        return requests.post(f"{DAEMON_URL}/api/store", json={"text": text, "source": source}, timeout=10).status_code == 200
    except:
        return False


def echo_query(query, max_results=15):
    try:
        r = requests.post(f"{DAEMON_URL}/api/echo", json={"query": query, "max_results": max_results}, timeout=10)
        return r.json() if r.status_code == 200 else {"results": [], "count": 0}
    except:
        return {"results": [], "count": 0}


def restart_daemon(threshold):
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = os.path.join(os.path.dirname(__file__), "..", "target", "release", "shrimpk-daemon.exe" if os.name == "nt" else "shrimpk-daemon")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"], capture_output=True, timeout=5)
    except:
        pass
    time.sleep(0.3)
    if os.path.exists(data_file):
        try: os.remove(data_file)
        except: pass
    env = os.environ.copy()
    env["SHRIMPK_SIMILARITY_THRESHOLD"] = str(threshold)
    try:
        subprocess.Popen([daemon_bin], creationflags=0x08000000, env=env)
    except:
        return False
    for _ in range(20):
        time.sleep(0.3)
        if daemon_healthy(): return True
    return False


def store_turn_pairs(session, date=None, session_idx=0):
    stored = 0
    turns = list(session)
    i = 0
    p = 0
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
        if store_memory(text, f"s{session_idx}-p{p}"): stored += 1
        p += 1
    return stored


def format_context(echo_results):
    parts = []
    for i, r in enumerate(echo_results, 1):
        content = r.get("content", "")
        part = f"=== Memory {i} ===\n{content}"
        if len(part) > 2000:
            cut = part[:2000]
            lp = cut.rfind(". ")
            if lp > 1000: cut = cut[:lp+1]
            part = cut
        parts.append(part)
    total = 0
    kept = []
    for p in parts:
        if total + len(p) > 16000: break
        kept.append(p)
        total += len(p)
    return "\n\n".join(kept) if kept else "No relevant memories found."


def ask_ollama(question, context, model):
    system = ("You are extracting facts from conversation memories. "
              "The answer is contained in the memories below. "
              "Focus on what the USER said -- user statements contain personal facts. "
              "Extract the specific answer. Respond in one short sentence.")
    user = f"{context}\n\nQuestion: {question}\nBased on the memories above, the answer is:"

    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64},
    }, timeout=600)
    elapsed_ms = int((time.time() - t0) * 1000)
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip(), elapsed_ms
    raise Exception(f"Ollama {r.status_code}")


def check_correct(hypothesis, answer):
    h = hypothesis.lower()
    a = answer.lower()
    if a in h:
        return True
    # Check key words
    words = [w.strip() for w in a.split() if len(w.strip()) > 3]
    if words:
        matched = sum(1 for w in words if w in h)
        if matched / len(words) >= 0.5:
            return True
    return False


def is_refusal(hypothesis):
    h = hypothesis.lower()
    return ("don't have" in h or "do not have" in h or "don\u2019t have" in h or
            "cannot answer" in h or "no relevant" in h or "i don't know" in h or
            "not enough information" in h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}
    test_items = [(qid, ref_map[qid]) for qid in TEST_IDS if qid in ref_map]

    # Check all models available
    print("Checking models...")
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        available = [m["name"] for m in tags.get("models", [])]
    except:
        available = []

    for model in MODELS:
        found = model in available or any(model in m for m in available)
        status = "OK" if found else "MISSING"
        print(f"  {model}: {status}")
        if not found:
            print(f"  Pulling {model}...")
            subprocess.run(["ollama", "pull", model], timeout=1800)

    os.makedirs("results", exist_ok=True)

    # Load checkpoint
    completed = {}
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            completed = json.load(f)
        print(f"Resuming: {len(completed)} results loaded")

    print(f"\n{'='*90}")
    print(f"THRESHOLD x MODEL MATRIX BENCHMARK")
    print(f"{'='*90}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Models: {MODELS}")
    print(f"Questions: {len(test_items)}")
    print(f"Total LLM calls: {len(THRESHOLDS) * len(MODELS) * len(test_items)}")

    # Phase 1: Build contexts (threshold x question)
    # Reuse across models
    contexts = {}  # key: (threshold, qid) -> context string

    print(f"\n--- Phase 1: Building retrieval contexts ---")
    for threshold in THRESHOLDS:
        print(f"\n  Threshold: {threshold}")
        for qid, item in test_items:
            key = f"{threshold}_{qid}"
            if key in contexts:
                continue

            restart_daemon(threshold)
            sessions = item["haystack_sessions"]
            dates = item.get("haystack_dates", [None] * len(sessions))

            stored = 0
            for j, (session, date) in enumerate(zip(sessions, dates)):
                stored += store_turn_pairs(session, date, session_idx=j)

            echo_result = echo_query(item["question"], max_results=15)
            results = echo_result.get("results", [])
            context = format_context(results)

            # Also check if answer is in context
            answer_in_ctx = item["answer"].lower() in context.lower()
            contexts[key] = {
                "context": context,
                "echo_count": echo_result.get("count", 0),
                "answer_in_context": answer_in_ctx,
            }
            status = "FOUND" if answer_in_ctx else "miss"
            print(f"    {qid} | {echo_result.get('count',0):2d} results | {status}")

    print(f"\n  Contexts built: {len(contexts)}")

    # Phase 2: Run LLM for each model x threshold x question
    print(f"\n--- Phase 2: Running LLM inference ---")
    all_results = {}

    for model in MODELS:
        print(f"\n  Model: {model}")
        for threshold in THRESHOLDS:
            for qid, item in test_items:
                result_key = f"{model}_{threshold}_{qid}"
                if result_key in completed:
                    all_results[result_key] = completed[result_key]
                    continue

                ctx_key = f"{threshold}_{qid}"
                ctx_data = contexts[ctx_key]
                question = item["question"]
                answer = item["answer"]

                try:
                    hypothesis, response_ms = ask_ollama(question, ctx_data["context"], model)
                except Exception as e:
                    hypothesis = f"ERROR: {e}"
                    response_ms = 0

                correct = check_correct(hypothesis, answer)
                refusal = is_refusal(hypothesis)

                result = {
                    "model": model,
                    "threshold": threshold,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "hypothesis": hypothesis[:200],
                    "correct": correct,
                    "refusal": refusal,
                    "answer_in_context": ctx_data["answer_in_context"],
                    "echo_count": ctx_data["echo_count"],
                    "response_ms": response_ms,
                }
                all_results[result_key] = result

                mark = "Y" if correct else ("R" if refusal else "N")
                print(f"    t={threshold} q={qid[:8]} [{mark}] {response_ms:>5}ms {hypothesis[:50]}")

                # Save checkpoint after each result
                completed[result_key] = result
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(completed, f)

    # Phase 3: Summary
    print(f"\n{'='*90}")
    print(f"THRESHOLD x MODEL MATRIX RESULTS")
    print(f"{'='*90}")

    # Accuracy table
    header = f"{'':>14}"
    for model in MODELS:
        short = model.split(":")[0][:8] + ":" + model.split(":")[1] if ":" in model else model[:12]
        header += f" {short:>14}"
    print(header)
    print("-" * (14 + 15 * len(MODELS)))

    best_overall = None
    best_score = -1

    for threshold in THRESHOLDS:
        row = f"t={threshold:<10}"
        for model in MODELS:
            correct = sum(1 for qid, _ in test_items
                         if all_results.get(f"{model}_{threshold}_{qid}", {}).get("correct", False))
            total = len(test_items)
            row += f" {correct:>6}/{total:<7}"
            score = correct
            if score > best_score:
                best_score = score
                best_overall = (threshold, model)
        print(row)

    # Refusal rate table
    print(f"\nRefusal rates:")
    header = f"{'':>14}"
    for model in MODELS:
        short = model.split(":")[0][:8] + ":" + model.split(":")[1] if ":" in model else model[:12]
        header += f" {short:>14}"
    print(header)
    print("-" * (14 + 15 * len(MODELS)))

    for threshold in THRESHOLDS:
        row = f"t={threshold:<10}"
        for model in MODELS:
            refusals = sum(1 for qid, _ in test_items
                          if all_results.get(f"{model}_{threshold}_{qid}", {}).get("refusal", False))
            total = len(test_items)
            row += f" {refusals:>6}/{total:<7}"
        print(row)

    # Avg response time table
    print(f"\nAvg response time (ms):")
    header = f"{'':>14}"
    for model in MODELS:
        short = model.split(":")[0][:8] + ":" + model.split(":")[1] if ":" in model else model[:12]
        header += f" {short:>14}"
    print(header)
    print("-" * (14 + 15 * len(MODELS)))

    for threshold in THRESHOLDS:
        row = f"t={threshold:<10}"
        for model in MODELS:
            times = [all_results.get(f"{model}_{threshold}_{qid}", {}).get("response_ms", 0)
                     for qid, _ in test_items]
            avg_ms = sum(times) // max(len(times), 1)
            row += f" {avg_ms:>10}ms   "
        print(row)

    # Best per model
    print(f"\nBest threshold per model:")
    for model in MODELS:
        best_t = None
        best_c = -1
        for threshold in THRESHOLDS:
            correct = sum(1 for qid, _ in test_items
                         if all_results.get(f"{model}_{threshold}_{qid}", {}).get("correct", False))
            if correct > best_c:
                best_c = correct
                best_t = threshold
        avg_time = sum(all_results.get(f"{model}_{best_t}_{qid}", {}).get("response_ms", 0)
                       for qid, _ in test_items) // len(test_items)
        print(f"  {model}: threshold={best_t} ({best_c}/{len(test_items)}) avg={avg_time}ms")

    if best_overall:
        print(f"\nOverall best: threshold={best_overall[0]}, model={best_overall[1]} ({best_score}/{len(test_items)})")

    # Save full results
    with open(RESULTS_FILE, "w") as f:
        json.dump(list(all_results.values()), f, indent=2)
    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
