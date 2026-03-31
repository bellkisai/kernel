#!/usr/bin/env python3
"""
ShrimPK Diagnostic Test — traces echo memories through the full pipeline.

For each question:
  1. Store sessions → call echo → format context → call Ollama
  2. Check: does the answer appear in raw echo results?
  3. Check: does the answer survive truncation into the prompt?
  4. Check: does the model extract it correctly?
  5. Verdict: PASS / TRUNCATED / IGNORED / RETRIEVAL_MISS

Usage:
  python diagnostic_test.py --version v1   # test original pipeline
  python diagnostic_test.py --version v2   # test fixed pipeline
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

TEST_IDS = [
    "e47becba", "118b2229", "51a45a95", "6ade9755", "c5e8278d",
    "6f9b354f", "58ef2f1c", "f8c5f88b", "5d3d2817", "7527f7e2",
]


def daemon_healthy():
    try:
        return requests.get(f"{DAEMON_URL}/health", timeout=2).status_code == 200
    except:
        return False


def store_memory(text, source="diag"):
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


def clear_memories():
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = os.path.join(os.path.dirname(__file__), "..", "target", "release", "shrimpk-daemon.exe" if os.name == "nt" else "shrimpk-daemon")
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"], capture_output=True, timeout=5)
        else:
            subprocess.run(["pkill", "-f", "shrimpk-daemon"], capture_output=True, timeout=5)
    except: pass
    time.sleep(0.3)
    if os.path.exists(data_file):
        try: os.remove(data_file)
        except: pass
    try:
        if sys.platform == "win32":
            subprocess.Popen([daemon_bin], creationflags=0x08000000)
        else:
            subprocess.Popen([daemon_bin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: return False
    for _ in range(20):
        time.sleep(0.3)
        if daemon_healthy(): return True
    return False


# ---- V1 pipeline (original) ----

def v1_store(sessions, dates):
    stored = 0
    for j, (session, date) in enumerate(zip(sessions, dates)):
        lines = []
        if date: lines.append(f"[Date: {date}]")
        for turn in session:
            lines.append(f"{turn['role']}: {turn['content']}")
        if store_memory("\n".join(lines), f"s-{j}"): stored += 1
    return stored


def v1_format(echo_results):
    parts = []
    for r in echo_results:
        content = r.get("content", "")
        sim = r.get("similarity", 0)
        part = f"[Similarity: {sim:.0%}]\n{content}"
        if len(part) > 3000:
            part = part[:3000] + "\n[...truncated]"
        parts.append(part)
    total = 0
    kept = []
    for p in parts:
        if total + len(p) > 16000: break
        kept.append(p)
        total += len(p)
    return "\n\n---\n\n".join(kept) if kept else "No relevant memories found."


def v1_prompt(question, context):
    system = ("You are answering questions about past conversations. "
              "Use ONLY the retrieved conversation memories below to answer. "
              "If the information is not in the memories, say you don't have that information. "
              "Be concise and direct. Give short factual answers, ideally one sentence.")
    user = f"Retrieved memories:\n\n{context}\n\nQuestion: {question}\n\nAnswer in one short sentence."
    return system, user


# ---- V2 pipeline (fixed) ----

def v2_store(sessions, dates):
    stored = 0
    for j, (session, date) in enumerate(zip(sessions, dates)):
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
            if len(text.replace(f"[{date}]", "").strip()) < 20: continue
            if store_memory(text, f"s{j}-p{pair_idx}"): stored += 1
            pair_idx += 1
    return stored


def v2_format(echo_results):
    parts = []
    for i, r in enumerate(echo_results, 1):
        content = r.get("content", "")
        part = f"=== Memory {i} ===\n{content}"
        if len(part) > 2000:
            cut = part[:2000]
            lp = cut.rfind(". ")
            if lp > 1000: cut = cut[:lp+1]
            part = cut + "\n[...truncated]"
        parts.append(part)
    total = 0
    kept = []
    for p in parts:
        if total + len(p) > 16000: break
        kept.append(p)
        total += len(p)
    return "\n\n".join(kept) if kept else "No relevant memories found."


def v2_prompt(question, context):
    system = ("You are extracting facts from conversation memories. "
              "The answer is contained in the memories below. "
              "Focus on what the USER said — user statements contain personal facts. "
              "Extract the specific answer. Respond in one short sentence.")
    user = f"{context}\n\nQuestion: {question}\nBased on the memories above, the answer is:"
    return system, user


def ask_ollama(system, user, model="gemma3:1b", temperature=0.1, num_predict=128):
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }, timeout=300)
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip()
    return f"ERROR: {r.status_code}"


def run_diagnostic(dataset_path, version="v2", model="gemma3:1b"):
    with open(dataset_path, encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}

    store_fn = v2_store if version == "v2" else v1_store
    format_fn = v2_format if version == "v2" else v1_format
    prompt_fn = v2_prompt if version == "v2" else v1_prompt
    temp = 0.0 if version == "v2" else 0.1
    np = 64 if version == "v2" else 128

    results = []

    for qid in TEST_IDS:
        item = ref_map.get(qid)
        if not item:
            print(f"  WARNING: {qid} not found")
            continue

        question = item["question"]
        answer = item["answer"]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))

        print(f"\n{'='*70}")
        print(f"QID: {qid} | Q: {question}")
        print(f"EXPECTED: {answer}")

        # Reset + store
        clear_memories()
        stored = store_fn(sessions, dates)
        print(f"  Stored: {stored} memories")

        # Echo
        echo_result = echo_query(question, max_results=15)
        raw_results = echo_result.get("results", [])
        echo_count = echo_result.get("count", 0)

        # Check: answer in raw echo?
        all_raw = " ".join(r.get("content", "") for r in raw_results)
        answer_in_raw = answer.lower() in all_raw.lower()
        print(f"  Echo: {echo_count} results | Answer in raw echo: {answer_in_raw}")

        # Format context
        context = format_fn(raw_results)
        context_len = len(context)

        # Check: answer in formatted context?
        answer_in_context = answer.lower() in context.lower()
        print(f"  Context length: {context_len} chars | Answer in context: {answer_in_context}")

        if answer_in_context:
            pos = context.lower().find(answer.lower())
            pct = 100 * pos / max(context_len, 1)
            print(f"  Answer position: char {pos} ({pct:.0f}% through context)")

        # Ask model
        system, user = prompt_fn(question, context)
        hypothesis = ask_ollama(system, user, model=model, temperature=temp, num_predict=np)
        print(f"  Model says: {hypothesis[:120]}")

        # Verdict
        hyp_lower = hypothesis.lower()
        is_refusal = ("don't have" in hyp_lower or "do not have" in hyp_lower or
                      "don\u2019t have" in hyp_lower or "no relevant" in hyp_lower)
        answer_in_hyp = answer.lower() in hyp_lower

        if not answer_in_raw and echo_count == 0:
            verdict = "RETRIEVAL_MISS"
        elif not answer_in_raw:
            verdict = "WRONG_RESULTS"
        elif not answer_in_context:
            verdict = "TRUNCATED"
        elif is_refusal:
            verdict = "IGNORED"
        elif answer_in_hyp:
            verdict = "PASS"
        else:
            verdict = "PARTIAL"

        print(f"  VERDICT: {verdict}")

        results.append({
            "qid": qid,
            "question": question,
            "answer": answer,
            "echo_count": echo_count,
            "answer_in_raw": answer_in_raw,
            "answer_in_context": answer_in_context,
            "context_len": context_len,
            "hypothesis": hypothesis[:200],
            "is_refusal": is_refusal,
            "answer_in_hyp": answer_in_hyp,
            "verdict": verdict,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({version} | {model})")
    print(f"{'='*70}")
    verdicts = {}
    for r in results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    total = len(results)
    for v in ["PASS", "PARTIAL", "IGNORED", "TRUNCATED", "WRONG_RESULTS", "RETRIEVAL_MISS"]:
        c = verdicts.get(v, 0)
        print(f"  {v:20s}: {c}/{total} ({100*c/total:.0f}%)")

    retrieval_ok = sum(1 for r in results if r["answer_in_raw"])
    context_ok = sum(1 for r in results if r["answer_in_context"])
    refusal_count = sum(1 for r in results if r["is_refusal"])

    print(f"\n  Answer in raw echo:     {retrieval_ok}/{total} ({100*retrieval_ok/total:.0f}%)")
    print(f"  Answer in final context: {context_ok}/{total} ({100*context_ok/total:.0f}%)")
    print(f"  Truncation loss:         {retrieval_ok - context_ok}/{total}")
    print(f"  Refusal rate:            {refusal_count}/{total} ({100*refusal_count/total:.0f}%)")
    print(f"  Avg context length:      {sum(r['context_len'] for r in results)//total} chars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShrimPK Diagnostic Test")
    parser.add_argument("--dataset", default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--model", default="gemma3:1b")
    args = parser.parse_args()

    run_diagnostic(args.dataset, version=args.version, model=args.model)
