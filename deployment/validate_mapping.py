"""Validate an allocation profile against the real Phase 5 windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.federated.allocation import DEFAULT_MAPPING, validate_allocation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--profile", type=int, choices=(2, 4, 8, 17), required=True)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    args = parser.parse_args(argv)
    kwargs = {}
    if args.processed_dir is not None:
        kwargs["processed_dir"] = args.processed_dir
    if args.split_file is not None:
        kwargs["split_file"] = args.split_file
    try:
        result = validate_allocation(args.mapping, profile=args.profile, **kwargs)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
