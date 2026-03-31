#!/usr/bin/env python3
"""
ShrimPK Sanity Test — POC: Does the LLM actually use echo memories?

Compares LLM answers WITH echo memory context vs WITHOUT (raw question only).
If echo memory works, Run B (with context) should give specific correct answers
while Run A (no context) gives generic/wrong/refusal answers.

Usage:
  python sanity_test.py
"""

import json
import os
import subprocess
import sys
import time
import requests

DAEMON_URL = "http://127.0.0.1:11435"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "gemma3:1b"

# Questions where v2 diagnostic confirmed echo FINDS the answer
TEST_CASES = [
    ("51a45a95", "Where did I redeem a $5 coupon on coffee creamer?", "Target"),
    ("6ade9755", "Where do I take yoga classes?", "Serenity Yoga"),
    ("c5e8278d", "What was my last name before I changed it?", "Johnson"),
    ("7527f7e2", "How much did I spend on a designer handbag?", "$800"),
    ("6f9b354f", "What color did I repaint my bedroom walls?", "a lighter shade of gray"),
]


def daemon_healthy():
    try:
        return requests.get(f"{DAEMON_URL}/health", timeout=2).status_code == 200
    except:
        return False


def store_memory(text, source="sanity"):
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


def clear_and_restart():
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
    try:
        subprocess.Popen([daemon_bin], creationflags=0x08000000)
    except:
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
    return "\n\n".join(kept) if kept else ""


def ask_ollama(question, context=None):
    """Ask Ollama with or without context."""
    if context:
        # WITH echo memory
        system = ("You are extracting facts from conversation memories. "
                  "The answer is contained in the memories below. "
                  "Focus on what the USER said — user statements contain personal facts. "
                  "Extract the specific answer. Respond in one short sentence.")
        user = f"{context}\n\nQuestion: {question}\nBased on the memories above, the answer is:"
    else:
        # WITHOUT echo memory — raw question, no context
        system = "Answer the following personal question. Be concise."
        user = question

    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64},
    }, timeout=300)
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip()
    return f"ERROR: {r.status_code}"


def main():
    dataset_path = "LongMemEval/data/longmemeval_s_cleaned.json"
    with open(dataset_path, encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}

    print("=" * 80)
    print("SANITY TEST: Does the LLM use echo memory in its thinking?")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Test: 5 questions, each asked WITH and WITHOUT echo context\n")

    results = []

    for qid, question, expected in TEST_CASES:
        item = ref_map.get(qid)
        if not item:
            print(f"  WARNING: {qid} not found")
            continue

        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))

        print(f"\n{'-'*70}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")

        # Run A: NO memory — raw question
        print(f"\n  [A] Without memory...")
        answer_no_mem = ask_ollama(question, context=None)
        print(f"      -> {answer_no_mem[:120]}")

        # Store memories for this question
        clear_and_restart()
        stored = 0
        for j, (session, date) in enumerate(zip(sessions, dates)):
            stored += store_session_as_turn_pairs(session, date, session_idx=j)

        # Echo
        echo_result = echo_query(question, max_results=15)
        echo_results = echo_result.get("results", [])
        context = format_context(echo_results)

        # Run B: WITH echo memory
        print(f"  [B] With echo memory ({len(echo_results)} results, {len(context)} chars)...")
        answer_with_mem = ask_ollama(question, context=context)
        print(f"      -> {answer_with_mem[:120]}")

        # Verdict
        a_lower = answer_no_mem.lower()
        b_lower = answer_with_mem.lower()
        expected_lower = expected.lower()

        a_correct = expected_lower in a_lower
        b_correct = expected_lower in b_lower
        answers_different = a_lower.strip() != b_lower.strip()

        if b_correct and not a_correct:
            verdict = "PROVEN — memory changed answer from wrong to correct"
        elif b_correct and a_correct:
            verdict = "BOTH CORRECT — model knew already (but may still use memory)"
        elif not b_correct and not a_correct:
            verdict = "BOTH WRONG — memory not helping"
        elif a_correct and not b_correct:
            verdict = "REGRESSION — memory made it worse"
        else:
            verdict = "UNCLEAR"

        if answers_different and not a_correct and not b_correct:
            verdict += " (but answers differ — memory IS influencing output)"

        print(f"\n  VERDICT: {verdict}")

        results.append({
            "qid": qid,
            "question": question,
            "expected": expected,
            "without_memory": answer_no_mem[:100],
            "with_memory": answer_with_mem[:100],
            "a_correct": a_correct,
            "b_correct": b_correct,
            "answers_different": answers_different,
            "verdict": verdict,
        })

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Question':<45} {'No Mem':>8} {'With Mem':>8} {'Different':>10} {'Verdict'}")
    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*10} {'-'*30}")

    proven = 0
    influenced = 0
    for r in results:
        q = r["question"][:44]
        a_mark = "Y" if r["a_correct"] else "N"
        b_mark = "Y" if r["b_correct"] else "N"
        diff = "YES" if r["answers_different"] else "no"
        v = "PROVEN" if "PROVEN" in r["verdict"] else ("INFLUENCED" if r["answers_different"] else "SAME")
        print(f"{q:<45} {a_mark:>8} {b_mark:>8} {diff:>10} {v}")

        if "PROVEN" in r["verdict"]:
            proven += 1
        if r["answers_different"]:
            influenced += 1

    print(f"\nResults: {proven}/{len(results)} PROVEN correct via memory")
    print(f"         {influenced}/{len(results)} answers influenced by memory")
    print(f"         {len(results) - influenced}/{len(results)} answers unchanged (model ignoring memory)")

    if proven > 0:
        print(f"\nY CONCLUSION: Echo memory IS being used by the LLM.")
    elif influenced > 0:
        print(f"\n~ CONCLUSION: Echo memory INFLUENCES the LLM but doesn't always help.")
    else:
        print(f"\nN CONCLUSION: Echo memory is NOT being used — investigate pipeline.")


if __name__ == "__main__":
    main()
