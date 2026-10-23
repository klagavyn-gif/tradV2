#!/usr/bin/env python3
"""
Project Memory Helper for Antigravity IDE.
Displays or updates the active memory ledger (.agents/memory/ACTIVE_CONTEXT.md).
"""

import sys
import os
import argparse
from datetime import datetime

MEMORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "memory", "ACTIVE_CONTEXT.md"))


def read_memory():
    if not os.path.exists(MEMORY_FILE):
        print(f"[ERROR] Memory file not found: {MEMORY_FILE}")
        return None
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Antigravity Project Memory Manager")
    parser.add_argument("--status", action="store_true", help="Print current active working memory")
    args = parser.parse_args()

    content = read_memory()
    if content:
        print("=== tradV2 Active Working Memory ===")
        print(content)
        print("====================================")


if __name__ == "__main__":
    main()
