#!/usr/bin/env python3
"""
ShrimPK LongMemEval Benchmark Runner

Evaluates ShrimPK Echo Memory against the LongMemEval benchmark (ICLR 2025).
Comparison target: Hindsight 91.4% (Gemini-3), Supermemory 85.2% (Gemini-3).

Uses Ollama for the reader LLM (no paid APIs).

Flow:
  1. For each question, ingest all conversation sessions into a fresh ShrimPK instance
  2. Query ShrimPK echo with the question
  3. Feed retrieved memories + question to Ollama for answering
  4. Output JSONL for evaluation with evaluate_qa.py

Usage:
  python run_longmemeval.py --model gemma3:1b --limit 5
  python run_longmemeval.py --model qwen2.5:7b
"""

import argparse
import json
import os
import sys
import time
import requests

DAEMON_URL = "http://127.0.0.1:11435"
OLLAMA_URL = "http://127.0.0.1:11434"


def daemon_healthy():
    try:
        r = requests.get(f"{DAEMON_URL}/health", timeout=2)
        return r.status_code == 200
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


def echo_query(query, max_results=10):
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
    import subprocess
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        # Fallback: try to find it in the kernel build
        daemon_bin = os.path.join(os.path.dirname(__file__), "..", "target", "release", "shrimpk-daemon.exe" if os.name == "nt" else "shrimpk-daemon")

    # Kill daemon
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"],
                           capture_output=True, timeout=5)
        else:
            subprocess.run(["pkill", "-f", "shrimpk-daemon"], capture_output=True, timeout=5)
    except:
        pass
    time.sleep(0.3)

    # Delete data file
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
        except:
            pass

    # Restart daemon
    try:
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen([daemon_bin], creationflags=CREATE_NO_WINDOW)
        else:
            subprocess.Popen([daemon_bin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  WARNING: Could not restart daemon: {e}")
        return False

    # Wait for daemon to be ready
    for _ in range(20):
        time.sleep(0.3)
        if daemon_healthy():
            return True
    print("  WARNING: Daemon did not restart in time")
    return False


def session_to_text(session, date=None):
    lines = []
    if date:
        lines.append(f"[Date: {date}]")
    for turn in session:
        lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(lines)


def truncate_context(context_parts, max_total=16000, max_per_item=3000):
    """Truncate retrieved context to fit in small model context windows."""
    truncated = []
    total = 0
    for part in context_parts:
        if len(part) > max_per_item:
            part = part[:max_per_item] + "\n[...truncated]"
        if total + len(part) > max_total:
            break
        truncated.append(part)
        total += len(part)
    return truncated


READER_SYSTEM_PROMPT = (
    "You are extracting facts from conversation memories. "
    "The answer is contained in the memories below. "
    "Focus on what the USER said — user statements contain personal facts. "
    "Extract the specific answer. Respond in one short sentence."
)

READER_USER_TEMPLATE = (
    "Context:\n"
    "-----\n"
    "{context}\n"
    "-----\n"
    "\n"
    "Given only the context above and not prior knowledge, extract the answer.\n"
    "Question: {question}\n"
    "Answer:"
)


def ask_ollama(question, context, model="gemma3:1b"):
    """Ask Ollama to answer based on retrieved context."""
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": READER_SYSTEM_PROMPT},
                {"role": "user", "content": READER_USER_TEMPLATE.format(
                    context=context, question=question)},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64},
        },
        timeout=300,
    )
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip()
    raise Exception(f"Ollama returned {r.status_code}: {r.text[:200]}")


def run_benchmark(dataset_path, output_path, model="gemma3:1b", max_results=10,
                  limit=None, resume=False):
    # Check services
    if not daemon_healthy():
        print("ERROR: ShrimPK daemon not running at", DAEMON_URL)
        sys.exit(1)
    print(f"ShrimPK daemon: OK")

    if not ollama_healthy(model):
        print(f"ERROR: Ollama model '{model}' not available at {OLLAMA_URL}")
        sys.exit(1)
    print(f"Ollama model: {model} OK")

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} questions")

    if limit:
        data = data[:limit]
        print(f"  Limited to {limit}")

    # Resume support
    done_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["question_id"])
        print(f"  Resuming: {len(done_ids)} done")

    # If not resuming, start fresh output
    if not resume and os.path.exists(output_path):
        os.remove(output_path)

    total = len(data)
    correct_count = 0
    processed = 0
    start_time = time.time()

    # Stats per category
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

        # Clear memories for fresh evaluation per question
        clear_memories()

        # Ingest sessions
        stored = 0
        for j, (session, date) in enumerate(zip(sessions, dates)):
            text = session_to_text(session, date)
            if store_memory(text, source=f"s-{j}"):
                stored += 1
        print(f"  Stored {stored}/{len(sessions)} sessions")

        # Query echo
        echo_result = echo_query(question, max_results=max_results)
        count = echo_result.get("count", 0)
        print(f"  Echo: {count} results")

        if count > 0:
            cat_echo_hits[q_type] = cat_echo_hits.get(q_type, 0) + 1

        # Format context (truncated to fit model context window)
        raw_parts = []
        for r in echo_result.get("results", []):
            sim = r.get("similarity", 0)
            content = r.get("content", "")
            raw_parts.append(f"[Similarity: {sim:.0%}]\n{content}")
        parts = truncate_context(raw_parts)
        context = "\n\n---\n\n".join(parts) if parts else "No relevant memories found."

        # Ask Ollama
        try:
            hypothesis = ask_ollama(question, context, model=model)
            print(f"  A: {hypothesis[:100]}...")
        except Exception as e:
            print(f"  LLM ERROR: {e}")
            hypothesis = "I don't have that information."

        # Write result
        result = {"question_id": qid, "hypothesis": hypothesis}
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        processed += 1

        # Progress stats every 25 questions
        if processed % 25 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (total - len(done_ids) - processed) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {processed} done, {rate:.1f} q/s, ~{remaining/60:.0f}min remaining ---")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE: {processed} questions in {elapsed:.0f}s ({processed/elapsed:.1f} q/s)")
    print(f"Output: {output_path}")
    print(f"Model: {model}")
    print(f"\nEcho retrieval hit rate by category:")
    for cat in sorted(cat_total.keys()):
        hits = cat_echo_hits.get(cat, 0)
        tot = cat_total[cat]
        print(f"  {cat}: {hits}/{tot} ({100*hits/tot:.0f}%)")
    print(f"\nTo evaluate (requires OpenAI API key for LLM-as-judge):")
    print(f"  cd LongMemEval/src/evaluation")
    print(f"  python evaluate_qa.py gpt-4o ../../{output_path} ../../LongMemEval/data/longmemeval_oracle.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShrimPK LongMemEval Benchmark")
    parser.add_argument("--dataset", default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--output", default=None, help="Output JSONL (auto-named if omitted)")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model name")
    parser.add_argument("--max-results", type=int, default=10, help="Max echo results")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    # Auto-name output based on model
    if args.output is None:
        safe_model = args.model.replace(":", "-").replace("/", "-")
        args.output = f"results/shrimpk_{safe_model}.jsonl"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        model=args.model,
        max_results=args.max_results,
        limit=args.limit,
        resume=args.resume,
    )
