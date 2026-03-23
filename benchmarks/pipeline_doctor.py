#!/usr/bin/env python3
"""
ShrimPK Pipeline Doctor -- diagnose WHY specific questions fail retrieval.

For each failing question, tests each pipeline layer:
  Layer 1: ISOLATION -- store ONLY the answer turn-pair, echo. Does 1-on-1 work?
  Layer 2: EMBEDDING -- compute raw cosine similarity between question and answer text
  Layer 3: NOISE -- store answer + increasing noise. At what level does it break?
  Layer 4: QUERY REFORMULATION -- try alternate phrasings of the question
  Layer 5: FULL PIPELINE -- store all sessions, echo with threshold=0.01 (near-zero). Is the answer even in the candidates?

Usage:
  python pipeline_doctor.py
"""

import json
import os
import subprocess
import sys
import time
import requests
import re

DAEMON_URL = "http://127.0.0.1:11435"
OLLAMA_URL = "http://127.0.0.1:11434"

# Questions that NEVER retrieve the answer at ANY threshold
FAILING_QIDS = ["e47becba", "118b2229", "58ef2f1c", "f8c5f88b", "5d3d2817"]


def daemon_healthy():
    try:
        return requests.get(f"{DAEMON_URL}/health", timeout=2).status_code == 200
    except:
        return False


def store_memory(text, source="doctor"):
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


def restart_daemon(threshold=0.01):
    data_file = os.path.expanduser("~/.shrimpk-kernel/echo_store.shrm")
    daemon_bin = os.path.expandvars(r"%LOCALAPPDATA%\ShrimPK\bin\shrimpk-daemon.exe")
    if not os.path.exists(daemon_bin):
        daemon_bin = "C:/Users/lior1/bellkis/kernel/target/release/shrimpk-daemon.exe"
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


def get_answer_turn_pair(item):
    """Find the user+assistant turn-pair containing the answer."""
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates", [None] * len(sessions))
    for i, (session, date) in enumerate(zip(sessions, dates)):
        turns = list(session)
        for j, turn in enumerate(turns):
            if turn.get("has_answer"):
                parts = []
                if date:
                    parts.append(f"[{date}]")
                parts.append(f"{turn['role']}: {turn['content']}")
                # Include following turn if exists
                if j + 1 < len(turns):
                    parts.append(f"{turns[j+1]['role']}: {turns[j+1]['content']}")
                return "\n".join(parts), i, turn["content"]
    return None, -1, None


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


def main():
    with open("LongMemEval/data/longmemeval_s_cleaned.json", encoding="utf-8") as f:
        all_data = json.load(f)
    ref_map = {d["question_id"]: d for d in all_data}

    print("=" * 80)
    print("PIPELINE DOCTOR -- Diagnosing retrieval failures")
    print("=" * 80)
    print(f"Patients: {len(FAILING_QIDS)} failing questions\n")

    for qid in FAILING_QIDS:
        item = ref_map[qid]
        question = item["question"]
        answer = item["answer"]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [None] * len(sessions))

        answer_text, answer_session_idx, answer_content = get_answer_turn_pair(item)
        if not answer_text:
            print(f"  WARNING: No answer turn found for {qid}")
            continue

        print(f"\n{'='*80}")
        print(f"PATIENT: {qid}")
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"  Answer in session {answer_session_idx}")
        print(f"  Answer turn: {answer_content[:120]}...")

        # ---- LAYER 1: ISOLATION TEST ----
        # Store ONLY the answer turn-pair. Does echo find it 1-on-1?
        print(f"\n  [Layer 1] ISOLATION -- answer turn-pair only")
        restart_daemon(threshold=0.01)
        store_memory(answer_text, "answer-only")
        echo_result = echo_query(question, max_results=5)
        count = echo_result.get("count", 0)
        if count > 0:
            top_sim = echo_result["results"][0].get("similarity", 0)
            top_content = echo_result["results"][0].get("content", "")[:80]
            found = answer.lower() in " ".join(r.get("content","") for r in echo_result["results"]).lower()
            print(f"    Results: {count} | Top sim: {top_sim:.3f} | Answer found: {found}")
            print(f"    Top result: {top_content}...")
            if found:
                print(f"    DIAGNOSIS: 1-on-1 WORKS. Problem is noise/competition.")
            else:
                print(f"    DIAGNOSIS: 1-on-1 FAILS. Embedding mismatch between Q and A.")
        else:
            print(f"    Results: 0 | DIAGNOSIS: Echo returns NOTHING even 1-on-1. Severe embedding gap.")

        # ---- LAYER 2: SIMILARITY SCORE ----
        # What is the actual similarity between question and answer text?
        # We can infer this from Layer 1's top_sim
        if count > 0:
            print(f"\n  [Layer 2] SIMILARITY SCORE")
            print(f"    Q->A similarity: {top_sim:.4f}")
            if top_sim < 0.05:
                print(f"    DIAGNOSIS: Extremely low (<0.05). Embeddings see Q and A as unrelated.")
            elif top_sim < 0.10:
                print(f"    DIAGNOSIS: Very low (<0.10). Below default threshold 0.14. Would be filtered.")
            elif top_sim < 0.14:
                print(f"    DIAGNOSIS: Low (<0.14). Below default threshold. Lowering to 0.10 would help.")
            elif top_sim < 0.20:
                print(f"    DIAGNOSIS: Moderate. Above threshold but may rank below noise.")
            else:
                print(f"    DIAGNOSIS: Good (>0.20). Should rank well.")

        # ---- LAYER 3: NOISE TOLERANCE ----
        # Add increasing amounts of noise memories. At what level does the answer drop out?
        print(f"\n  [Layer 3] NOISE TOLERANCE")
        noise_levels = [0, 5, 10, 25, 50, 100, 250]

        for noise_count in noise_levels:
            restart_daemon(threshold=0.01)
            # Store the answer
            store_memory(answer_text, "answer")

            # Store noise from other sessions (skip the answer session)
            noise_stored = 0
            for j, (session, date) in enumerate(zip(sessions, dates)):
                if j == answer_session_idx:
                    continue
                if noise_stored >= noise_count:
                    break
                turns = list(session)
                for k in range(0, len(turns), 2):
                    if noise_stored >= noise_count:
                        break
                    parts = []
                    if date: parts.append(f"[{date}]")
                    parts.append(f"{turns[k]['role']}: {turns[k]['content']}")
                    if k+1 < len(turns):
                        parts.append(f"{turns[k+1]['role']}: {turns[k+1]['content']}")
                    store_memory("\n".join(parts), f"noise-{noise_stored}")
                    noise_stored += 1

            # Echo
            echo_result = echo_query(question, max_results=20)
            all_text = " ".join(r.get("content", "") for r in echo_result.get("results", []))
            found = answer.lower() in all_text.lower()
            result_count = echo_result.get("count", 0)

            # Where does the answer rank?
            rank = -1
            if found:
                for idx, r in enumerate(echo_result.get("results", [])):
                    if answer.lower() in r.get("content", "").lower():
                        rank = idx + 1
                        break

            status = f"FOUND (rank #{rank})" if found else "LOST"
            print(f"    Noise={noise_count:>3}: {result_count:>3} results | {status}")

            if not found and noise_count > 0:
                print(f"    DIAGNOSIS: Answer drops out at {noise_count} noise memories.")
                break

        # ---- LAYER 4: QUERY REFORMULATION ----
        # Try alternate phrasings of the question
        print(f"\n  [Layer 4] QUERY REFORMULATION")
        # Store just the answer memory
        restart_daemon(threshold=0.01)
        store_memory(answer_text, "answer")

        # Try different phrasings
        alt_queries = []
        # Extract key terms from the answer
        if "degree" in question.lower() or "graduate" in question.lower():
            alt_queries = [
                question,
                f"I graduated with a degree",
                f"my degree in college",
                f"Business Administration degree",
                answer_content[:60],
            ]
        elif "commute" in question.lower():
            alt_queries = [
                question,
                f"daily commute time to work",
                f"45 minutes commute",
                f"listening to audiobooks during commute",
                answer_content[:60],
            ]
        elif "volunteer" in question.lower():
            alt_queries = [
                question,
                f"volunteered at animal shelter fundraising",
                f"February 14th fundraising dinner",
                f"Love is in the Air fundraising dinner",
                answer_content[:60],
            ]
        elif "tennis" in question.lower() or "racket" in question.lower():
            alt_queries = [
                question,
                f"bought tennis racket sports store",
                f"new tennis racket from downtown",
                f"sports store downtown tennis",
                answer_content[:60],
            ]
        elif "occupation" in question.lower() or "previous" in question.lower():
            alt_queries = [
                question,
                f"previous job marketing specialist",
                f"worked at a small startup",
                f"marketing specialist startup role",
                answer_content[:60],
            ]
        else:
            alt_queries = [question, answer_content[:60]]

        for alt_q in alt_queries:
            echo_result = echo_query(alt_q, max_results=5)
            count = echo_result.get("count", 0)
            if count > 0:
                top_sim = echo_result["results"][0].get("similarity", 0)
                found = answer.lower() in " ".join(r.get("content","") for r in echo_result["results"]).lower()
                status = f"FOUND (sim={top_sim:.3f})" if found else f"MISS (sim={top_sim:.3f})"
            else:
                status = "NO RESULTS"
            print(f"    \"{alt_q[:50]}...\" -> {status}")

        # ---- LAYER 5: FULL PIPELINE, THRESHOLD=0.01 ----
        # Store ALL sessions as turn-pairs, threshold near zero. Is the answer even a candidate?
        print(f"\n  [Layer 5] FULL PIPELINE (threshold=0.01, all sessions)")
        restart_daemon(threshold=0.01)
        total_stored = 0
        for j, (session, date) in enumerate(zip(sessions, dates)):
            total_stored += store_turn_pairs(session, date, session_idx=j)

        echo_result = echo_query(question, max_results=20)
        all_text = " ".join(r.get("content", "") for r in echo_result.get("results", []))
        found = answer.lower() in all_text.lower()
        result_count = echo_result.get("count", 0)

        print(f"    Stored: {total_stored} turn-pairs")
        print(f"    Echo results: {result_count}")
        print(f"    Answer in top-20: {found}")

        if found:
            for idx, r in enumerate(echo_result.get("results", [])):
                if answer.lower() in r.get("content", "").lower():
                    print(f"    Answer at rank #{idx+1}, sim={r.get('similarity', 0):.4f}")
                    break
        else:
            # Show what DID come back
            print(f"    Top 3 results (what echo returned instead):")
            for idx, r in enumerate(echo_result.get("results", [])[:3]):
                sim = r.get("similarity", 0)
                content = r.get("content", "")[:80]
                print(f"      #{idx+1} sim={sim:.3f}: {content}...")

        # ---- FINAL DIAGNOSIS ----
        print(f"\n  FINAL DIAGNOSIS for {qid}:")
        # Summarize findings

    print(f"\n{'='*80}")
    print("DOCTOR'S SUMMARY")
    print("=" * 80)
    print("Review each patient's layer results above to identify the failing component.")
    print("Common patterns:")
    print("  - Layer 1 FAILS = embedding model doesn't connect Q to A semantically")
    print("  - Layer 1 OK, Layer 3 breaks early = noise drowns out weak signal")
    print("  - Layer 4 finds with alt query = query reformulation could help")
    print("  - Layer 5 not in top-20 = LSH/Bloom pre-filter may be excluding it")


if __name__ == "__main__":
    main()
