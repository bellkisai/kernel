#!/usr/bin/env python3
"""
ShrimPK LongMemEval Benchmark Runner — v2

Fixes applied (from 3-way architect/ML-engineer/QA analysis):
  1. Turn-pair storage: store user+assistant pairs, not full sessions
  2. Extraction-focused prompt: no refusal instruction, positive framing
  3. Clean context formatting: no similarity scores, clear markers
  4. Smart truncation: 2K per item (rarely triggers with turn-pairs)

Usage:
  python run_longmemeval_v2.py --model gemma3:1b --limit 50
  python run_longmemeval_v2.py --model qwen2.5:7b
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


def daemon_healthy():
    try:
        return requests.get(f"{DAEMON_URL}/health", timeout=2).status_code == 200
    except:
        return False


def ollama_healthy(model):
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if model in models or any(model in m for m in models):
                return True
            print(f"  Model '{model}' not found. Available: {models}")
        return False
    except:
        return False


def store_memory(text, source="benchmark"):
    try:
        r = requests.post(
            f"{DAEMON_URL}/api/store",
            json={"text": text, "source": source},
            timeout=10,
        )
        return r.status_code == 200
    except:
        return False


def echo_query(query, max_results=15):
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


def clear_memories():
    """Fast reset: stop daemon, delete data file, restart daemon."""
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = "C:/Users/lior1/bellkis/kernel/target/release/shrimpk-daemon.exe"

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"],
                           capture_output=True, timeout=5)
        else:
            subprocess.run(["pkill", "-f", "shrimpk-daemon"], capture_output=True, timeout=5)
    except:
        pass
    time.sleep(0.3)

    if os.path.exists(data_file):
        try:
            os.remove(data_file)
        except:
            pass

    try:
        if sys.platform == "win32":
            subprocess.Popen([daemon_bin], creationflags=0x08000000)
        else:
            subprocess.Popen([daemon_bin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  WARNING: Could not restart daemon: {e}")
        return False

    for _ in range(20):
        time.sleep(0.3)
        if daemon_healthy():
            return True
    print("  WARNING: Daemon did not restart in time")
    return False


# ---------------------------------------------------------------------------
# FIX 1: Turn-pair storage — store each user+assistant pair as a separate memory
# ---------------------------------------------------------------------------

def store_session_as_turn_pairs(session, date=None, session_idx=0):
    """Split a session into user+assistant turn-pairs and store each separately.
    Returns the number of memories stored."""
    stored = 0
    turns = list(session)
    i = 0
    pair_idx = 0

    while i < len(turns):
        turn = turns[i]
        # Build a turn-pair: user turn + following assistant turn (if any)
        parts = []
        if date:
            parts.append(f"[{date}]")

        parts.append(f"{turn['role']}: {turn['content']}")

        # If this is a user turn, include the next assistant turn
        if turn["role"] == "user" and i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
            parts.append(f"assistant: {turns[i + 1]['content']}")
            i += 2
        # If this is an assistant turn without preceding user, store solo
        elif turn["role"] == "assistant" and (i == 0 or turns[i - 1]["role"] != "user"):
            i += 1
        else:
            i += 1

        text = "\n".join(parts)
        # Skip very short pairs (< 20 chars of actual content)
        if len(text.replace(f"[{date}]", "").strip()) < 20:
            continue

        if store_memory(text, source=f"s{session_idx}-p{pair_idx}"):
            stored += 1
        pair_idx += 1

    return stored


# ---------------------------------------------------------------------------
# FIX 3 + 4: Clean context formatting with smart truncation
# ---------------------------------------------------------------------------

def truncate_context(context_parts, max_total=16000, max_per_item=2000):
    """Truncate with larger per-item budget. Turn-pairs rarely hit this."""
    truncated = []
    total = 0
    for part in context_parts:
        if len(part) > max_per_item:
            # Try to cut at sentence boundary
            cut = part[:max_per_item]
            last_period = cut.rfind(". ")
            if last_period > max_per_item // 2:
                cut = cut[:last_period + 1]
            part = cut + "\n[...truncated]"
        if total + len(part) > max_total:
            break
        truncated.append(part)
        total += len(part)
    return truncated


def format_context(echo_results):
    """Format echo results with clean markers, no similarity scores."""
    raw_parts = []
    for i, r in enumerate(echo_results, 1):
        content = r.get("content", "")
        raw_parts.append(f"=== Memory {i} ===\n{content}")
    parts = truncate_context(raw_parts)
    return "\n\n".join(parts) if parts else "No relevant memories found."


# ---------------------------------------------------------------------------
# FIX 2: Extraction-focused prompt — no refusal, positive framing
# ---------------------------------------------------------------------------

def ask_ollama(question, context, model="gemma3:1b"):
    """Ask Ollama with extraction-focused prompt."""
    system_prompt = (
        "You are extracting facts from conversation memories. "
        "The answer is contained in the memories below. "
        "Focus on what the USER said — user statements contain personal facts. "
        "Extract the specific answer. Respond in one short sentence."
    )

    user_prompt = f"""{context}

Question: {question}
Based on the memories above, the answer is:"""

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64},
        },
        timeout=300,
    )
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip()
    raise Exception(f"Ollama returned {r.status_code}: {r.text[:200]}")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(dataset_path, output_path, model="gemma3:1b", max_results=15,
                  limit=None, resume=False):
    if not daemon_healthy():
        print("ERROR: ShrimPK daemon not running at", DAEMON_URL)
        sys.exit(1)
    print(f"ShrimPK daemon: OK")

    if not ollama_healthy(model):
        print(f"ERROR: Ollama model '{model}' not available at {OLLAMA_URL}")
        sys.exit(1)
    print(f"Ollama model: {model} OK")
    print(f"Version: v2 (turn-pair storage, extraction prompt)")

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} questions")

    if limit:
        data = data[:limit]
        print(f"  Limited to {limit}")

    done_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["question_id"])
        print(f"  Resuming: {len(done_ids)} done")

    if not resume and os.path.exists(output_path):
        os.remove(output_path)

    total = len(data)
    processed = 0
    refusals = 0
    start_time = time.time()
    cat_total = {}
    cat_echo_hits = {}

    for i, item in enumerate(data):
        qid = item["question_id"]
        if qid in done_ids:
            continue

        question = item["question"]
        q_type = item["question_type"]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))

        cat_total[q_type] = cat_total.get(q_type, 0) + 1

        print(f"\n[{i+1}/{total}] {qid} ({q_type})")
        print(f"  Q: {question[:80]}...")

        # Clear and restart
        clear_memories()

        # FIX 1: Store as turn-pairs
        stored = 0
        for j, (session, date) in enumerate(zip(sessions, dates)):
            stored += store_session_as_turn_pairs(session, date, session_idx=j)
        print(f"  Stored {stored} turn-pairs from {len(sessions)} sessions")

        # Query echo
        echo_result = echo_query(question, max_results=max_results)
        results = echo_result.get("results", [])
        count = echo_result.get("count", 0)
        print(f"  Echo: {count} results")

        if count > 0:
            cat_echo_hits[q_type] = cat_echo_hits.get(q_type, 0) + 1

        # FIX 3: Clean context formatting
        context = format_context(results)

        # FIX 2: Ask with extraction prompt
        try:
            hypothesis = ask_ollama(question, context, model=model)
            # Track refusals
            hyp_lower = hypothesis.lower()
            if "don't have" in hyp_lower or "do not have" in hyp_lower or "don\u2019t have" in hyp_lower or "no relevant" in hyp_lower:
                refusals += 1
            print(f"  A: {hypothesis[:100]}...")
        except Exception as e:
            print(f"  LLM ERROR: {e}")
            hypothesis = "I don't have that information."
            refusals += 1

        result = {"question_id": qid, "hypothesis": hypothesis}
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        processed += 1

        if processed % 25 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - len(done_ids) - processed) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {processed} done, {rate:.1f} q/s, ~{remaining/60:.0f}min remaining ---")
            print(f"  --- Refusal rate: {100*refusals/processed:.0f}% ---")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE: {processed} questions in {elapsed:.0f}s ({processed/elapsed:.1f} q/s)")
    print(f"Model: {model} | Version: v2")
    print(f"Refusal rate: {refusals}/{processed} ({100*refusals/max(processed,1):.0f}%)")
    print(f"Output: {output_path}")
    print(f"\nEcho retrieval hit rate by category:")
    for cat in sorted(cat_total.keys()):
        hits = cat_echo_hits.get(cat, 0)
        tot = cat_total[cat]
        print(f"  {cat}: {hits}/{tot} ({100*hits/tot:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShrimPK LongMemEval v2")
    parser.add_argument("--dataset", default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="gemma3:1b")
    parser.add_argument("--max-results", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        safe_model = args.model.replace(":", "-").replace("/", "-")
        args.output = f"results/shrimpk_v2_{safe_model}.jsonl"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        model=args.model,
        max_results=args.max_results,
        limit=args.limit,
        resume=args.resume,
    )
