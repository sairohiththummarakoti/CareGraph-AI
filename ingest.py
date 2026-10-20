"""Build the approved synthetic reference library; no report ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from caregraph.retrieval import DEFAULT_INDEX_DIR, IndexSetupError, SentenceTransformerEmbedder, build_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rebuild", action="store_true", help="Explicitly replace the reference index.")
    parser.add_argument("--offline", action="store_true", help="Require the pinned model to be cached already.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable build status.")
    args = parser.parse_args(argv)
    try:
        result = build_index(
            args.index_dir,
            SentenceTransformerEmbedder(offline=args.offline),
            rebuild=args.rebuild,
        )
    except (IndexSetupError, OSError) as exc:
        print(f"Index setup error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        action = "Built" if result["rebuilt"] else "Verified unchanged"
        print(f"{action}: {result['count']} synthetic reference chunks, {result['dimension']} dimensions.")
        print(f"Model: {result['model']['id']} @ {result['model']['revision']}")
        print(f"Index: {result['index_dir']}")
        print("Sources: " + ", ".join(result["sources"]))
        print("Reports are excluded. Runtime search uses this existing index without writing to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
