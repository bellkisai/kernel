#!/usr/bin/env python3
"""
ShrimPK retrieval stage for LongMemEval-S — split pipeline compatible
with the official run_generation.py and print_retrieval_metrics.py.

This script performs ONLY the retrieval step:
  1. For each question, clear ShrimPK and ingest all haystack sessions
  2. Query ShrimPK echo with the question
  3. Map echo results back to session IDs
  4. Compute recall/NDCG metrics (session-level)
  5. Write augmented dataset JSON (one line per question)

Output is a JSONL file where each line is the original dataset entry
augmented with `retrieval_results.ranked_items` and `.metrics`.
This file can be fed directly to:
  - print_retrieval_metrics.py (retrieval-only evaluation)
  - run_generation.py --retriever_type flat-session (reader generation)

Usage:
  python run_retrieval_shrimpk.py --limit 5          # quick test
  python run_retrieval_shrimpk.py                     # full 500q
  python run_retrieval_shrimpk.py --max-results 20    # more echo results
  python run_retrieval_shrimpk.py --resume             # continue from last

Evaluate retrieval:
  python LongMemEval/src/evaluation/print_retrieval_metrics.py results/<output>.jsonl

Feed into reader:
  python LongMemEval/src/generation/run_generation.py \\
    --in_file results/<output>.jsonl \\
    --retriever_type flat-session --topk_context 5 ...
"""

import argparse
import json
import os
import sys
import time
import subprocess
import numpy as np

DAEMON_URL = "http://127.0.0.1:11435"


# ---------------------------------------------------------------------------
# Daemon interaction
# ---------------------------------------------------------------------------

def daemon_healthy():
    import requests
    try:
        r = requests.get(f"{DAEMON_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def store_memory(text, source="benchmark"):
    import requests
    try:
        r = requests.post(
            f"{DAEMON_URL}/api/store",
            json={"text": text, "source": source},
            timeout=30,
        )
        return r.status_code == 200
    except Exception:
        return False


def echo_query(query, max_results=10):
    import requests
    try:
        r = requests.post(
            f"{DAEMON_URL}/api/echo",
            json={"query": query, "max_results": max_results},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        return {"results": [], "count": 0}
    except Exception:
        return {"results": [], "count": 0}


def clear_memories():
    """Fast reset: kill daemon, delete data, restart."""
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    hebbian_file = os.path.expanduser("~/.shrimpk-kernel/hebbian.json")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = os.path.join(
            os.path.dirname(__file__), "..", "target", "release",
            "shrimpk-daemon.exe" if os.name == "nt" else "shrimpk-daemon"
        )

    # Kill daemon
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "shrimpk-daemon.exe"],
                           capture_output=True, timeout=5)
        else:
            subprocess.run(["pkill", "-f", "shrimpk-daemon"],
                           capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(0.3)

    # Delete data files
    for f in [data_file, hebbian_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    # Restart daemon
    try:
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen([daemon_bin], creationflags=CREATE_NO_WINDOW)
        else:
            subprocess.Popen([daemon_bin],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
# Session ingestion — one memory per session (user turns only)
# ---------------------------------------------------------------------------

def session_to_user_text(session, date=None):
    """Concatenate user turns from a session, with optional date prefix.

    Matches the official run_retrieval.py `process_item_flat_index` at
    session granularity: user turns joined by space.
    """
    user_turns = [turn["content"] for turn in session if turn["role"] == "user"]
    text = " ".join(user_turns)
    if date:
        text = f"[Date: {date}] {text}"
    return text


# ---------------------------------------------------------------------------
# Metrics — local computation matching eval_utils.py
# ---------------------------------------------------------------------------

def dcg(relevances, k):
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))


def ndcg_score(rankings, correct_docs, corpus_ids, k=10):
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_rels = [relevances[idx] for idx in rankings[:k]]
    ideal_rels = sorted(relevances, reverse=True)
    ideal = dcg(ideal_rels, k)
    actual = dcg(sorted_rels, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    recalled = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(doc in recalled for doc in correct_docs))
    recall_all = float(all(doc in recalled for doc in correct_docs))
    ndcg_val = ndcg_score(rankings, correct_docs, corpus_ids, k)
    return recall_any, recall_all, ndcg_val


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_retrieval(dataset_path, output_path, max_results=10, limit=None, resume=False):
    if not daemon_healthy():
        print("ERROR: ShrimPK daemon not running at", DAEMON_URL)
        sys.exit(1)
    print(f"ShrimPK daemon: OK")

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

    if not resume and os.path.exists(output_path):
        os.remove(output_path)

    total = len(data)
    start_time = time.time()
    processed = 0

    # Aggregate metrics for summary
    all_metrics = []

    for i, item in enumerate(data):
        qid = item["question_id"]
        if qid in done_ids:
            continue

        question = item["question"]
        q_type = item["question_type"]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))
        session_ids = item.get("haystack_session_ids", [f"sess_{j}" for j in range(len(sessions))])
        answer_session_ids = item.get("answer_session_ids", [])

        print(f"\n[{i+1}/{total}] {qid} ({q_type})")
        print(f"  Q: {question[:80]}...")

        # Clear and restart daemon
        clear_memories()

        # Ingest sessions — one memory per session, source tag = session_id
        # Build corpus and ID mapping for metric computation
        corpus = []        # text per session (user turns)
        corpus_ids = []    # session IDs
        corpus_timestamps = []

        stored = 0
        for j, (session, date, sid) in enumerate(zip(sessions, dates, session_ids)):
            text = session_to_user_text(session, date=None)  # no date prefix in stored text
            corpus.append(text)
            corpus_ids.append(sid)
            corpus_timestamps.append(date or "")

            # Store in ShrimPK with date prefix (helps temporal retrieval)
            store_text = session_to_user_text(session, date=date)
            if store_memory(store_text, source=sid):
                stored += 1

        print(f"  Stored {stored}/{len(sessions)} sessions")

        # Identify correct documents
        correct_docs = list(set(
            doc_id for doc_id in corpus_ids if "answer" in doc_id
        ))

        # Query echo
        echo_result = echo_query(question, max_results=max_results)
        results = echo_result.get("results", [])
        print(f"  Echo: {len(results)} results")

        # Map echo results to session IDs via the source tag
        # Each echo result has r["source"] = session_id we stored with
        ranked_items = []
        seen_sids = set()

        for r in results:
            source = r.get("source", "")
            # The source tag is the session_id
            if source in seen_sids:
                continue  # deduplicate (one entry per session)
            seen_sids.add(source)

            # Find the corpus index for this session_id
            text = ""
            timestamp = ""
            if source in corpus_ids:
                idx = corpus_ids.index(source)
                text = corpus[idx]
                timestamp = corpus_timestamps[idx]
            else:
                text = r.get("content", "")

            ranked_items.append({
                "corpus_id": source,
                "text": text,
                "timestamp": timestamp,
            })

        # Fill in un-retrieved sessions at the end (so all sessions appear in ranking)
        for sid, text, ts in zip(corpus_ids, corpus, corpus_timestamps):
            if sid not in seen_sids:
                ranked_items.append({
                    "corpus_id": sid,
                    "text": text,
                    "timestamp": ts,
                })

        # Build rankings array (indices into corpus_ids) for metric computation
        rankings = []
        for ri in ranked_items:
            if ri["corpus_id"] in corpus_ids:
                rankings.append(corpus_ids.index(ri["corpus_id"]))

        # Compute metrics
        metrics = {"session": {}, "turn": {}}
        for k in [1, 3, 5, 10, 30, 50]:
            recall_any, recall_all, ndcg_val = evaluate_retrieval(
                rankings, correct_docs, corpus_ids, k=k
            )
            metrics["session"][f"recall_any@{k}"] = recall_any
            metrics["session"][f"recall_all@{k}"] = recall_all
            metrics["session"][f"ndcg_any@{k}"] = ndcg_val

        all_metrics.append(metrics)

        # Print key metrics
        r5 = metrics["session"].get("recall_all@5", 0)
        r10 = metrics["session"].get("recall_all@10", 0)
        n10 = metrics["session"].get("ndcg_any@10", 0)
        print(f"  Metrics: recall_all@5={r5:.2f} recall_all@10={r10:.2f} ndcg@10={n10:.3f}")
        if correct_docs:
            print(f"  Correct docs: {correct_docs}")

        # Build output entry — pass through all original fields + retrieval_results
        out_entry = {
            "question_id": qid,
            "question_type": q_type,
            "question": question,
            "answer": item["answer"],
            "question_date": item.get("question_date", ""),
            "haystack_dates": dates,
            "haystack_sessions": sessions,
            "haystack_session_ids": session_ids,
            "answer_session_ids": answer_session_ids,
            "retrieval_results": {
                "query": question,
                "ranked_items": ranked_items,
                "metrics": metrics,
            },
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out_entry) + "\n")

        processed += 1

        if processed % 25 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - len(done_ids) - processed) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {processed} done, {rate:.1f} q/s, ~{remaining/60:.0f}min remaining ---")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE: {processed} questions in {elapsed:.0f}s ({processed/elapsed:.1f} q/s)" if elapsed > 0 else f"DONE: {processed} questions")
    print(f"Output: {output_path}")

    # Aggregate metrics (excluding abstention questions)
    valid_metrics = [m for m, item in zip(all_metrics, data[:len(all_metrics)])
                     if "_abs" not in item["question_id"]]
    if valid_metrics:
        print(f"\nAggregated session-level metrics ({len(valid_metrics)} questions):")
        for name in ["recall_all@5", "ndcg_any@5", "recall_all@10", "ndcg_any@10"]:
            vals = [m["session"].get(name, 0) for m in valid_metrics]
            print(f"  {name} = {np.mean(vals):.4f}")

    print(f"\nTo evaluate retrieval:")
    print(f"  python LongMemEval/src/evaluation/print_retrieval_metrics.py {output_path}")
    print(f"\nTo run generation (reader):")
    print(f"  python LongMemEval/src/generation/run_generation.py \\")
    print(f"    --in_file {output_path} --retriever_type flat-session \\")
    print(f"    --topk_context 5 --history_format nl --useronly true --cot false \\")
    print(f"    --model_name <model> --model_alias <alias> --openai_key <key> \\")
    print(f"    --out_dir results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShrimPK retrieval for LongMemEval-S")
    parser.add_argument("--dataset",
                        default="LongMemEval/data/longmemeval_s_cleaned.json",
                        help="Path to LME-S cleaned dataset JSON")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (auto-named if omitted)")
    parser.add_argument("--max-results", type=int, default=10,
                        help="Max echo results per query")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    # Auto-name output
    if args.output is None:
        args.output = f"results/shrimpk_retrieval_k{args.max_results}.jsonl"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_retrieval(
        dataset_path=args.dataset,
        output_path=args.output,
        max_results=args.max_results,
        limit=args.limit,
        resume=args.resume,
    )
