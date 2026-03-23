#!/usr/bin/env python3
"""Cross-model smoke test: all 4 Ollama models with echo memory."""

import json
import os
import subprocess
import sys
import time
import requests

DAEMON = "http://127.0.0.1:11435"
OLLAMA = "http://127.0.0.1:11434"
MODELS = ["gemma3:1b", "qwen2.5:1.5b", "phi4-mini"]

TEST_CASES = [
    ("51a45a95", "Where did I redeem a $5 coupon on coffee creamer?", "Target"),
    ("6ade9755", "Where do I take yoga classes?", "Serenity Yoga"),
    ("c5e8278d", "What was my last name before I changed it?", "Johnson"),
    ("7527f7e2", "How much did I spend on a designer handbag?", "$800"),
    ("6f9b354f", "What color did I repaint my bedroom walls?", "a lighter shade of gray"),
]


def restart_daemon():
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
        daemon_bin = "C:/Users/lior1/bellkis/kernel/target/release/shrimpk-daemon.exe"
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


def store_turn_pairs(session, date=None, idx=0):
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
        try:
            requests.post(f"{DAEMON}/api/store", json={"text": text, "source": f"s{idx}-p{p}"}, timeout=10)
            stored += 1
        except:
            pass
        p += 1
    return stored


def format_context(echo_results):
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


def ask_ollama(question, context, model):
    system = ("You are extracting facts from conversation memories. "
              "The answer is contained in the memories below. "
              "Focus on what the USER said -- user statements contain personal facts. "
              "Extract the specific answer. Respond in one short sentence.")
    user = f"{context}\n\nQuestion: {question}\nBased on the memories above, the answer is:"
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64},
    }, timeout=600)
    ms = int((time.time() - t0) * 1000)
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip(), ms
    return f"ERROR: {r.status_code}", ms


def main():
    with open("LongMemEval/data/longmemeval_s_cleaned.json", encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}

    print("=" * 90)
    print("CROSS-MODEL SMOKE TEST -- All models with echo memory")
    print("=" * 90)
    print(f"Models: {MODELS}")
    print(f"Questions: {len(TEST_CASES)}")
    print()

    results = {}

    for qid, question, expected in TEST_CASES:
        item = ref_map[qid]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))

        restart_daemon()
        stored = 0
        for j, (session, date) in enumerate(zip(sessions, dates)):
            stored += store_turn_pairs(session, date, idx=j)

        echo = requests.post(f"{DAEMON}/api/echo",
                             json={"query": question, "max_results": 15}, timeout=10).json()
        context = format_context(echo.get("results", []))
        echo_count = echo.get("count", 0)

        print(f"Q: {question[:60]}")
        print(f"  Expected: {expected} | Echo: {echo_count} results | Stored: {stored}")

        for model in MODELS:
            try:
                hypothesis, ms = ask_ollama(question, context, model)
            except Exception as e:
                hypothesis, ms = f"ERROR: {e}", 0

            correct = expected.lower() in hypothesis.lower()
            mark = "Y" if correct else "N"
            results[(model, qid)] = {"hypothesis": hypothesis, "correct": correct, "ms": ms}
            print(f"  [{mark}] {model:<14} {ms:>6}ms  {hypothesis[:60]}")
        print()

    # Summary
    print("=" * 90)
    print("ACCURACY PER MODEL")
    print("=" * 90)
    for m in MODELS:
        correct = sum(1 for qid, _, _ in TEST_CASES
                      if results.get((m, qid), {}).get("correct"))
        total = len(TEST_CASES)
        avg_ms = sum(results.get((m, qid), {}).get("ms", 0)
                     for qid, _, _ in TEST_CASES) // total
        print(f"  {m:<16} {correct}/{total} ({100 * correct // total}%)  avg {avg_ms}ms")


if __name__ == "__main__":
    main()
