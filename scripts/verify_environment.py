#!/usr/bin/env python3
"""Verify the runtime environment for the Industrial PDF-to-Graph pipeline."""

from scripts.check_environment import run_all_checks


if __name__ == "__main__":
    ok = run_all_checks()
    raise SystemExit(0 if ok else 2)
