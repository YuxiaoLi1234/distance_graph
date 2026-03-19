#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
from typing import Dict, List


def read_pairs_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    required = ["nodes1", "edges1", "nodes2", "edges2"]
    for key in required:
        if key not in reader.fieldnames:
            raise ValueError(f"Missing required column '{key}' in {path}")
    return rows


def run_one(compute_script: str, python_exec: str, row: Dict[str, str], out_csv: str, default_alpha: float, verbose: bool):
    alpha = row.get("alpha", "").strip()
    alpha = float(alpha) if alpha else default_alpha
    tag = row.get("tag", "").strip()

    cmd = [
        python_exec,
        compute_script,
        "--graph1-files",
        row["nodes1"],
        row["edges1"],
        "--graph2-files",
        row["nodes2"],
        row["edges2"],
        "--alpha",
        str(alpha),
        "--save-csv",
        out_csv,
        "--tag",
        tag,
    ]
    if verbose:
        cmd.append("--verbose")
        print("RUN:", " ".join(cmd))

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch compute fGW for many graph pairs and save CSV.")
    parser.add_argument("--pairs", required=True, help="CSV with columns: nodes1,edges1,nodes2,edges2[,alpha][,tag]")
    parser.add_argument("--out", required=True, help="Output CSV for results")
    parser.add_argument("--python", default="./.venv312/bin/python", help="Python executable used to run compute_fgw.py")
    parser.add_argument("--compute", default="./compute_fgw.py", help="Path to compute_fgw.py")
    parser.add_argument("--alpha", type=float, default=0.5, help="Default alpha when row alpha is empty")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.pairs):
        raise FileNotFoundError(f"pairs file not found: {args.pairs}")
    rows = read_pairs_csv(args.pairs)
    if len(rows) == 0:
        raise ValueError("pairs csv has no rows")

    for i, row in enumerate(rows, start=1):
        if args.verbose:
            print(f"[{i}/{len(rows)}] processing")
        run_one(args.compute, args.python, row, args.out, args.alpha, args.verbose)

    print(f"Done. Results saved to {args.out}")


if __name__ == "__main__":
    main()
