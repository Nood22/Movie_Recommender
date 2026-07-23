#!/usr/bin/env python3
"""Call the current API from user feedback through final recommendations."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def post(base_url: str, path: str, payload: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--liked", action="append", default=[])
    parser.add_argument("--disliked", action="append", default=[])
    parser.add_argument("--context", default="")
    parser.add_argument("--summary")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.75)
    args = parser.parse_args()

    summary = args.summary
    if not summary:
        summary = post(
            args.base_url,
            "/summarize",
            {"liked": args.liked, "disliked": args.disliked, "context": args.context},
        )["summary"]
    result = post(
        args.base_url,
        "/recommend",
        {
            "summary": summary,
            "context": args.context,
            "liked": args.liked,
            "disliked": args.disliked,
            "top_k": args.top_k,
            "alpha": args.alpha,
        },
    )
    print(json.dumps({"summary": summary, **result}, indent=2))


if __name__ == "__main__":
    main()

