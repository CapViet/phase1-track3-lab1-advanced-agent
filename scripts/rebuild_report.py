"""Rebuild report.json / report.md from saved *_runs.jsonl without re-running the LLM.

Useful after changing reporting code, or to apply a cost price to an existing run:
    LLM_PRICE_PER_1M_TOKENS=0.30 python scripts/rebuild_report.py --run-dir outputs/real_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import RunRecord


def load_records(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    return [RunRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/real_run")
    ap.add_argument("--dataset", default="multihop_eval.json", help="Dataset name to record in meta.")
    ap.add_argument("--mode", default=None, help="Mode label for the report (defaults to existing meta if present).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    records = load_records(run_dir / "react_runs.jsonl") + load_records(run_dir / "reflexion_runs.jsonl")
    if not records:
        raise SystemExit(f"No *_runs.jsonl records found in {run_dir}")

    mode = args.mode
    if mode is None:
        import json

        existing = run_dir / "report.json"
        mode = json.loads(existing.read_text(encoding="utf-8"))["meta"]["mode"] if existing.exists() else "rebuilt"

    report = build_report(records, dataset_name=args.dataset, mode=mode)
    json_path, md_path = save_report(report, run_dir)
    print(f"Rebuilt {json_path}\nRebuilt {md_path}")
    print("Cost table:", report.cost)


if __name__ == "__main__":
    main()
