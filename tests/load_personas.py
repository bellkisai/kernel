#!/usr/bin/env python3
"""Bulk-load Sam Torres and Alex Chen persona memories into the ShrimPK daemon."""

import json
import time
import urllib.request
import urllib.error
import sys

DAEMON_URL = "http://localhost:11435/api/store"

MEMORIES = [
    # =========================================================================
    # Sam Torres — 41 memories, 6 sessions
    # =========================================================================

    # Session 1 (Day 1): Personal basics
    {"text": "My name is Sam Torres, I'm 29 years old", "source": "sam-session1"},
    {"text": "I live in a one-bedroom apartment in Oakland, California", "source": "sam-session1"},
    {"text": "I work as a frontend developer at Figma on their design tools team", "source": "sam-session1"},
    {"text": "I have a tabby cat named Mochi who is 3 years old", "source": "sam-session1"},
    {"text": "I grew up in Sacramento and got my CS degree from UC Davis in 2019", "source": "sam-session1"},
    {"text": "My parents still live in Sacramento, I visit them once a month", "source": "sam-session1"},

    # Session 2 (Day 3): Work context
    {"text": "My team at Figma works on the real-time collaboration features like multiplayer cursors and live comments", "source": "sam-session2"},
    {"text": "We use TypeScript and React for the frontend with a custom WebGL renderer for the canvas", "source": "sam-session2"},
    {"text": "I'm currently working on improving the performance of multiplayer editing for large files", "source": "sam-session2"},
    {"text": "My manager is Lisa Chen and our team has 6 engineers including me", "source": "sam-session2"},
    {"text": "We do two-week sprints and I'm the on-call rotation lead this month", "source": "sam-session2"},
    {"text": "Our biggest competitor is Penpot and the team watches their releases closely", "source": "sam-session2"},
    {"text": "I sit in the 4th floor open office area next to the coffee station", "source": "sam-session2"},

    # Session 3 (Day 7): Personal preferences
    {"text": "I use VS Code with the Dracula theme and Vim keybindings for all my coding", "source": "sam-session3"},
    {"text": "I'm vegetarian and love cooking Indian food at home, especially daal and paneer tikka masala", "source": "sam-session3"},
    {"text": "I run 5K three times a week, usually in the morning before work around Lake Merritt", "source": "sam-session3"},
    {"text": "I take BART to the Figma office in San Francisco, about a 35-minute commute each way", "source": "sam-session3"},
    {"text": "I'm reading Designing Data-Intensive Applications by Martin Kleppmann, about halfway through", "source": "sam-session3"},
    {"text": "I drink oat milk lattes, usually grab one from Blue Bottle Coffee near the office", "source": "sam-session3"},
    {"text": "I play acoustic guitar in the evenings, mostly folk and indie songs", "source": "sam-session3"},

    # Session 4 (Day 14): Work update
    {"text": "We shipped the new real-time collaboration engine last week and got great user feedback on performance improvements", "source": "sam-session4"},
    {"text": "I've been learning Rust on the side, working through the Rust Book and building a small CLI tool", "source": "sam-session4"},
    {"text": "Lisa asked me to lead the new plugin API project starting next sprint, it's a big opportunity", "source": "sam-session4"},
    {"text": "I got a new 32-inch 4K LG monitor and a standing desk from Autonomous for my home office", "source": "sam-session4"},
    {"text": "Our team is hiring two more engineers and I'm helping conduct technical interviews", "source": "sam-session4"},
    {"text": "I signed up for a half marathon in June, so I'm increasing my running distance gradually", "source": "sam-session4"},

    # Session 5 (Day 30): Preference changes
    {"text": "I switched from VS Code to Neovim with a custom Lua config, took me two weeks to set up but I love it", "source": "sam-session5"},
    {"text": "I had to stop running because of a knee injury from over-training for the half marathon", "source": "sam-session5"},
    {"text": "I started doing yoga three times a week at CorePower instead of running, it's easier on my joints", "source": "sam-session5"},
    {"text": "I've gotten into sourdough baking, made my first decent loaf last weekend after three failed attempts", "source": "sam-session5"},
    {"text": "I finished DDIA and now I'm reading The Pragmatic Programmer, finding it really practical", "source": "sam-session5"},
    {"text": "I started biking to work instead of taking BART, got a used Trek road bike for the commute", "source": "sam-session5"},
    {"text": "I'm learning Japanese on Duolingo, about 30 days into my streak now", "source": "sam-session5"},

    # Session 6 (Day 60): Major life updates
    {"text": "I left Figma last month and joined Vercel as a senior frontend engineer on the Next.js team", "source": "sam-session6"},
    {"text": "I moved from Oakland to San Francisco, got an apartment in the Mission District to be closer to work", "source": "sam-session6"},
    {"text": "Jordan and I started dating three weeks ago, we met at the SF Rust meetup", "source": "sam-session6"},
    {"text": "At Vercel I'm working on Next.js server components and the edge runtime", "source": "sam-session6"},
    {"text": "I adopted a second cat, a black kitten named Pixel, Mochi is adjusting to having a sibling", "source": "sam-session6"},
    {"text": "I passed my Rust skills assessment and I'm now contributing to an open source Rust project on weekends", "source": "sam-session6"},
    {"text": "I switched to making pour-over coffee at home with a Chemex after visiting a specialty roaster", "source": "sam-session6"},
    {"text": "My new commute is a 15-minute bike ride which is way better than the 35-minute BART from Oakland", "source": "sam-session6"},

    # =========================================================================
    # Alex Chen — 25 memories, 5 sessions
    # =========================================================================

    # Session 1: Personal basics
    {"text": "My name is Alex Chen and I'm 32 years old", "source": "alex-session1"},
    {"text": "I was born in Taipei, Taiwan but grew up in Vancouver, Canada", "source": "alex-session1"},
    {"text": "I have a golden retriever named Pixel who is 4 years old", "source": "alex-session1"},
    {"text": "My partner's name is Jordan and we've been together for 6 years", "source": "alex-session1"},
    {"text": "I'm allergic to shellfish and cats", "source": "alex-session1"},

    # Session 2: Work and education
    {"text": "I work as a senior backend engineer at Stripe in San Francisco", "source": "alex-session2"},
    {"text": "I graduated from the University of British Columbia with a CS degree in 2015", "source": "alex-session2"},
    {"text": "Before Stripe I worked at Shopify for 3 years on their payments team", "source": "alex-session2"},
    {"text": "My team at Stripe works on the billing infrastructure service", "source": "alex-session2"},
    {"text": "I'm being considered for a staff engineer promotion next quarter", "source": "alex-session2"},

    # Session 3: Technical preferences
    {"text": "I prefer Rust for systems programming and Go for microservices", "source": "alex-session3"},
    {"text": "My IDE is Neovim with LazyVim config and Catppuccin theme", "source": "alex-session3"},
    {"text": "For databases I use PostgreSQL for OLTP and ClickHouse for analytics", "source": "alex-session3"},
    {"text": "I run NixOS on my personal machines and macOS at work", "source": "alex-session3"},
    {"text": "My dotfiles are managed with chezmoi and stored on GitHub", "source": "alex-session3"},

    # Session 4: Hobbies and lifestyle
    {"text": "I practice Brazilian jiu-jitsu three times a week at a Gracie gym", "source": "alex-session4"},
    {"text": "I'm learning Japanese and currently at JLPT N3 level", "source": "alex-session4"},
    {"text": "I collect mechanical keyboards and my daily driver is a Keychron Q1 with Boba U4T switches", "source": "alex-session4"},
    {"text": "I brew pour-over coffee every morning using a Hario V60 and light roast beans", "source": "alex-session4"},
    {"text": "My favorite cuisine is Thai food, especially pad see ew and massaman curry", "source": "alex-session4"},

    # Session 5: Travel and goals
    {"text": "I visited Tokyo last November and stayed in Shinjuku for two weeks", "source": "alex-session5"},
    {"text": "My next trip is planned for Barcelona in April 2027", "source": "alex-session5"},
    {"text": "My long-term goal is to start a developer tools company focused on observability", "source": "alex-session5"},
    {"text": "I'm saving for a house in the Oakland Hills area", "source": "alex-session5"},
    {"text": "I want to compete in a jiu-jitsu tournament by the end of the year", "source": "alex-session5"},
]

def main():
    total = len(MEMORIES)
    print(f"Loading {total} persona memories into ShrimPK daemon at {DAEMON_URL}")
    print(f"  Sam Torres: 41 memories (6 sessions)")
    print(f"  Alex Chen:  25 memories (5 sessions)")
    print()

    t0 = time.time()
    ok = 0
    errors = 0

    for i, mem in enumerate(MEMORIES, 1):
        payload = json.dumps(mem).encode("utf-8")
        req = urllib.request.Request(
            DAEMON_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            ok += 1
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            errors += 1
            print(f"  ERROR on memory {i}: {e}")

        # Progress every 10, plus first and last
        if i % 10 == 0 or i == 1 or i == total:
            elapsed = time.time() - t0
            print(f"  loaded {i}/{total} memories  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print()
    print(f"Done. {ok} stored, {errors} errors, {elapsed:.1f}s total.")

    if errors > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
