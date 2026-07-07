#!/usr/bin/env python3
"""Perform one bounded HTTP health check using only the standard library."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expect", required=True)
    parser.add_argument("--timeout", type=float, default=4.0)
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:
            body = response.read(256_000).decode("utf-8", errors="replace")
            if response.status != 200:
                raise RuntimeError(f"unexpected HTTP status {response.status}")
            if args.expect not in body:
                raise RuntimeError("expected health marker was not present")
    except (OSError, urllib.error.URLError, RuntimeError) as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
