"""Command-line entry point for Phase 6 Steps 32–34."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.training.engine import TrainingConfig
from src.training.runner import DEFAULT_BUILDINGS, ROOT, run_step32, run_step33, run_step34


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the existing FEDEMS DailyHeatingCNN."
    )
    subparsers = parser.add_subparsers(dest="step", required=True)
    for step in ("step32", "step33", "step34"):
        command = subparsers.add_parser(step)
        if step == "step32":
            command.add_argument("--building-id", type=int, default=6479)
        else:
            command.add_argument(
                "--buildings",
                type=int,
                nargs="+",
                default=list(DEFAULT_BUILDINGS),
            )
        command.add_argument("--data-dir", type=Path, default=ROOT / "Dataset" / "processed")
        command.add_argument(
            "--split-file",
            type=Path,
            default=ROOT / "docs" / "data_reports" / "step21_temporal_split.csv",
        )
        command.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase6")
        command.add_argument("--epochs", type=int, default=100)
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--learning-rate", type=float, default=1e-3)
        command.add_argument("--patience", type=int, default=15)
        command.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    if args.step == "step32":
        summary = run_step32(
            args.building_id,
            processed_dir=args.data_dir,
            split_file=args.split_file,
            output_dir=args.output_dir,
            config=config,
        )
        output_path = args.output_dir / f"step32_building_{args.building_id}" / "metrics.json"
    elif args.step == "step33":
        summary = run_step33(
            args.buildings,
            processed_dir=args.data_dir,
            split_file=args.split_file,
            output_dir=args.output_dir,
            config=config,
        )
        output_path = args.output_dir / "step33" / "metrics.json"
    else:
        summary = run_step34(
            args.buildings,
            processed_dir=args.data_dir,
            split_file=args.split_file,
            output_dir=args.output_dir,
            config=config,
        )
        output_path = args.output_dir / "step34" / "metrics.json"
    print(
        f"Completed {args.step}: {', '.join(map(str, summary['building_ids']))} "
        f"(best checkpoint(s) and metrics written under {output_path.parent})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
