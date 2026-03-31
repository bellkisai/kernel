#!/usr/bin/env python3
"""
ShrimPK vs Competitors — Same questions, same model, different memory systems.

Tests:
  1. ShrimPK Echo Memory (v0.3.0)
  2. Mem0 (local, qdrant in-memory)
  3. No memory baseline (raw LLM, no context)

All use gemma3:1b via Ollama for the reader step.
Same 5 questions, same conversation data, same prompt template.

Usage:
  python competitor_comparison.py
"""

import json
import os
import subprocess
import sys
import time
import requests

OLLAMA = "http://127.0.0.1:11434"
DAEMON = "http://127.0.0.1:11435"
MODEL = "gemma3:1b"

TEST_CASES = [
    ("51a45a95", "Where did I redeem a $5 coupon on coffee creamer?", "Target"),
    ("6ade9755", "Where do I take yoga classes?", "Serenity Yoga"),
    ("c5e8278d", "What was my last name before I changed it?", "Johnson"),
    ("7527f7e2", "How much did I spend on a designer handbag?", "$800"),
    ("6f9b354f", "What color did I repaint my bedroom walls?", "a lighter shade of gray"),
]

SYSTEM_PROMPT = (
    "You are extracting facts from conversation memories. "
    "The answer is contained in the memories below. "
    "Focus on what the USER said -- user statements contain personal facts. "
    "Extract the specific answer. Respond in one short sentence."
)


def ask_ollama(question, context):
    user = f"{context}\n\nQuestion: {question}\nBased on the memories above, the answer is:"
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64},
    }, timeout=300)
    ms = int((time.time() - t0) * 1000)
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip(), ms
    return f"ERROR: {r.status_code}", ms


def ask_ollama_no_context(question):
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer the following personal question. Be concise."},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64},
    }, timeout=300)
    ms = int((time.time() - t0) * 1000)
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip(), ms
    return f"ERROR: {r.status_code}", ms


# ===========================================================================
# ShrimPK
# ===========================================================================

def shrimpk_restart():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"],
                       capture_output=True, timeout=5)
    except:
        pass
    time.sleep(0.5)
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    if os.path.exists(data_file):
        os.remove(data_file)
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = os.path.join(os.path.dirname(__file__), "..", "target", "release", "shrimpk-daemon.exe" if os.name == "nt" else "shrimpk-daemon")
    env = os.environ.copy()
    env["SHRIMPK_SIMILARITY_THRESHOLD"] = "0.10"
    subprocess.Popen([daemon_bin], creationflags=0x08000000, env=env)
    for _ in range(20):
        time.sleep(0.3)
        try:
            if requests.get(f"{DAEMON}/health", timeout=1).status_code == 200:
                return True
        except:
            pass
    return False


def shrimpk_store_turn_pairs(sessions, dates):
    stored = 0
    for j, (session, date) in enumerate(zip(sessions, dates)):
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
            try:
                requests.post(f"{DAEMON}/api/store", json={"text": text, "source": f"s{j}-p{p}"}, timeout=10)
                stored += 1
            except:
                pass
            p += 1
    return stored


def shrimpk_echo(question, max_results=15):
    try:
        r = requests.post(f"{DAEMON}/api/echo", json={"query": question, "max_results": max_results}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"results": [], "count": 0}


def shrimpk_format_context(echo_results):
    parts = []
    for i, r in enumerate(echo_results, 1):
        content = r.get("content", "")
        part = f"=== Memory {i} ===\n{content}"
        if len(part) > 2000:
            part = part[:2000]
        parts.append(part)
    total = 0
    kept = []
    for p in parts:
        if total + len(p) > 16000:
            break
        kept.append(p)
        total += len(p)
    return "\n\n".join(kept) if kept else "No relevant memories found."


# ===========================================================================
# Mem0
# ===========================================================================

def mem0_test(test_items, ref_map):
    """Run Mem0 on the same questions."""
    try:
        from mem0 import Memory
    except ImportError:
        print("  Mem0 not installed, skipping")
        return {}

    results = {}

    for qid, question, expected in TEST_CASES:
        item = ref_map[qid]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))

        # Initialize fresh Mem0 instance per question
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "gemma3:1b",
                    "ollama_base_url": OLLAMA,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "ollama_base_url": OLLAMA,
                },
            },
            "version": "v1.1",
        }

        try:
            m = Memory.from_config(config)
        except Exception as e:
            print(f"  Mem0 init failed: {e}")
            results[qid] = {"hypothesis": f"ERROR: {e}", "correct": False, "ms": 0,
                            "store_ms": 0, "echo_ms": 0, "echo_count": 0}
            continue

        # Store conversations as turn-pairs
        t0 = time.time()
        stored = 0
        for j, (session, date) in enumerate(zip(sessions, dates)):
            turns = list(session)
            for k in range(0, len(turns), 2):
                parts = []
                if date:
                    parts.append(f"[{date}]")
                parts.append(f"{turns[k]['role']}: {turns[k]['content']}")
                if k + 1 < len(turns):
                    parts.append(f"{turns[k+1]['role']}: {turns[k+1]['content']}")
                text = "\n".join(parts)
                if len(text.strip()) < 20:
                    continue
                try:
                    m.add(text, user_id="benchmark")
                    stored += 1
                except Exception:
                    pass
        store_ms = int((time.time() - t0) * 1000)

        # Search memories
        t1 = time.time()
        try:
            search_results = m.search(question, user_id="benchmark", limit=15)
            if isinstance(search_results, dict):
                memories = search_results.get("results", search_results.get("memories", []))
            elif isinstance(search_results, list):
                memories = search_results
            else:
                memories = []
        except Exception as e:
            print(f"  Mem0 search failed: {e}")
            memories = []
        echo_ms = int((time.time() - t1) * 1000)

        # Format context from Mem0 results
        context_parts = []
        for i, mem in enumerate(memories[:15], 1):
            if isinstance(mem, dict):
                content = mem.get("memory", mem.get("text", mem.get("content", str(mem))))
            else:
                content = str(mem)
            context_parts.append(f"=== Memory {i} ===\n{content}")
        context = "\n\n".join(context_parts) if context_parts else "No relevant memories found."

        # Ask Ollama
        try:
            hypothesis, llm_ms = ask_ollama(question, context)
        except Exception as e:
            hypothesis, llm_ms = f"ERROR: {e}", 0

        correct = expected.lower() in hypothesis.lower()
        total_ms = store_ms + echo_ms + llm_ms

        results[qid] = {
            "hypothesis": hypothesis,
            "correct": correct,
            "ms": total_ms,
            "store_ms": store_ms,
            "echo_ms": echo_ms,
            "llm_ms": llm_ms,
            "echo_count": len(memories),
            "stored": stored,
        }

        mark = "Y" if correct else "N"
        print(f"  [{mark}] {total_ms:>7}ms (store:{store_ms} echo:{echo_ms} llm:{llm_ms}) {hypothesis[:50]}")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    with open("LongMemEval/data/longmemeval_s_cleaned.json", encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}

    print("=" * 90)
    print("COMPETITOR COMPARISON -- ShrimPK vs Mem0 vs No Memory")
    print("=" * 90)
    print(f"Model: {MODEL}")
    print(f"Questions: {len(TEST_CASES)}")
    print()

    all_results = {}

    # ---- 1. No Memory Baseline ----
    print("--- NO MEMORY BASELINE ---")
    baseline = {}
    for qid, question, expected in TEST_CASES:
        hypothesis, ms = ask_ollama_no_context(question)
        correct = expected.lower() in hypothesis.lower()
        baseline[qid] = {"hypothesis": hypothesis, "correct": correct, "ms": ms}
        mark = "Y" if correct else "N"
        print(f"  [{mark}] {ms:>7}ms  {hypothesis[:60]}")
    all_results["no_memory"] = baseline
    print()

    # ---- 2. ShrimPK ----
    print("--- SHRIMPK v0.3.0 ---")
    shrimpk_results = {}
    for qid, question, expected in TEST_CASES:
        item = ref_map[qid]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))

        shrimpk_restart()
        t0 = time.time()
        stored = shrimpk_store_turn_pairs(sessions, dates)
        store_ms = int((time.time() - t0) * 1000)

        t1 = time.time()
        echo = shrimpk_echo(question)
        echo_ms = int((time.time() - t1) * 1000)
        context = shrimpk_format_context(echo.get("results", []))
        echo_count = echo.get("count", 0)

        hypothesis, llm_ms = ask_ollama(question, context)
        total_ms = store_ms + echo_ms + llm_ms
        correct = expected.lower() in hypothesis.lower()

        shrimpk_results[qid] = {
            "hypothesis": hypothesis, "correct": correct, "ms": total_ms,
            "store_ms": store_ms, "echo_ms": echo_ms, "llm_ms": llm_ms,
            "echo_count": echo_count, "stored": stored,
        }
        mark = "Y" if correct else "N"
        print(f"  [{mark}] {total_ms:>7}ms (store:{store_ms} echo:{echo_ms} llm:{llm_ms}) echo:{echo_count} {hypothesis[:50]}")
    all_results["shrimpk"] = shrimpk_results

    # Kill ShrimPK daemon before competitor tests
    try:
        subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"], capture_output=True, timeout=5)
    except:
        pass
    time.sleep(1)
    print()

    # ---- 3. Mem0 ----
    print("--- MEM0 ---")
    mem0_results = mem0_test(TEST_CASES, ref_map)
    all_results["mem0"] = mem0_results
    print()

    # ---- Summary ----
    print("=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)

    systems = [("no_memory", "No Memory"), ("shrimpk", "ShrimPK"), ("mem0", "Mem0")]

    header = f"{'Question':<45}"
    for _, label in systems:
        header += f" {label:>12}"
    print(header)
    print("-" * (45 + 13 * len(systems)))

    for qid, question, expected in TEST_CASES:
        row = f"{question[:44]:<45}"
        for key, _ in systems:
            r = all_results.get(key, {}).get(qid, {})
            mark = "Y" if r.get("correct") else "N"
            ms = r.get("ms", 0)
            row += f"  {mark} {ms:>7}ms"
        print(row)

    print()
    print("ACCURACY + TIMING SUMMARY:")
    print(f"{'System':<16} {'Correct':>8} {'Accuracy':>10} {'Avg Time':>10}")
    print("-" * 50)
    for key, label in systems:
        results = all_results.get(key, {})
        correct = sum(1 for r in results.values() if r.get("correct"))
        total = len(TEST_CASES)
        avg_ms = sum(r.get("ms", 0) for r in results.values()) // max(len(results), 1)
        pct = 100 * correct // total if total > 0 else 0
        print(f"{label:<16} {correct:>4}/{total:<3} {pct:>8}%  {avg_ms:>8}ms")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/competitor_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results/competitor_comparison.json")


if __name__ == "__main__":
    main()
