#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DatasetItem:
    name: str
    nx: int
    ny: int
    nz: int
    path: str


def parse_datasets_file(file_path: str) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
            path = parts[4]
            items.append(DatasetItem(name=name, nx=nx, ny=ny, nz=nz, path=path))
    return items


def run_cmd(cmd: List[str], verbose: bool = True):
    if verbose:
        print("$", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def ensure_csv_header(csv_path: str):
    header = [
        "dataset",
        "nx",
        "ny",
        "nz",
        "ori_path",
        "cmp_path",
        "decp_path",
        "cmp_size_bytes",
        "decp_size_bytes",
        "compress_time_s",
        "decompress_time_s",
        "fgw_csv",
        "status",
        "error",
    ]
    need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    if need_header:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_row(csv_path: str, row: List[str]):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def compute_data_range(data_path: str, num_values: int, dtype: str) -> float:
    np_dtype = np.float64 if dtype == "f64" else np.float32
    data = np.fromfile(data_path, dtype=np_dtype)
    if data.size < num_values:
        raise ValueError(
            f"Data size mismatch for {data_path}: got {data.size} values, expected at least {num_values}"
        )
    if data.size > num_values:
        data = data[:num_values]
    if data.size == 0:
        return 0.0
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    return dmax - dmin


def build_compress_cmd(args, item: DatasetItem, cmp_path: str, effective_error_bound: float) -> List[str]:
    if args.codec == "sz3":
        return [
            args.sz3,
            "-d",
            "-i",
            item.path,
            "-z",
            cmp_path,
            "-3",
            str(item.nx),
            str(item.ny),
            str(item.nz),
            "-M",
            "REL",
            str(args.error_bound),
        ]

    # zfp
    cmd = [
        args.zfp,
        "-d" if args.zfp_type == "f64" else "-f",
        "-i",
        item.path,
        "-z",
        cmp_path,
        "-3",
        str(item.nx),
        str(item.ny),
        str(item.nz),
    ]

    if args.zfp_mode == "rate":
        cmd.extend(["-r", str(args.zfp_rate)])
    elif args.zfp_mode == "precision":
        cmd.extend(["-p", str(args.zfp_precision)])
    else:
        # accuracy mode
        cmd.extend(["-a", str(effective_error_bound)])

    return cmd


def build_decompress_cmd(args, item: DatasetItem, cmp_path: str, decp_path: str) -> List[str]:
    if args.codec == "sz3":
        return [
            args.sz3,
            "-d",
            "-z",
            cmp_path,
            "-o",
            decp_path,
            "-3",
            str(item.nx),
            str(item.ny),
            str(item.nz),
        ]

    # zfp
    return [
        args.zfp,
        "-d" if args.zfp_type == "f64" else "-f",
        "-z",
        cmp_path,
        "-o",
        decp_path,
        "-3",
        str(item.nx),
        str(item.ny),
        str(item.nz),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Compress datasets using SZ3 or ZFP, then run ori/decp graph + fGW pipeline."
    )
    parser.add_argument("--datasets", default="./datasets.txt", help="Path to datasets.txt")
    parser.add_argument("--codec", choices=["sz3", "zfp"], default="sz3", help="Compression codec")
    parser.add_argument(
        "--error-bound",
        type=float,
        default=1e-4,
        help="SZ3 REL error bound, or ZFP accuracy tolerance when --zfp-mode=accuracy",
    )
    parser.add_argument("--sz3", default="sz3", help="SZ3 executable path")
    parser.add_argument("--zfp", default="zfp", help="ZFP executable path")
    parser.add_argument(
        "--zfp-mode",
        choices=["accuracy", "rate", "precision"],
        default="accuracy",
        help="ZFP mode: accuracy(-a), rate(-r), or precision(-p)",
    )
    parser.add_argument("--zfp-rate", type=float, default=16.0, help="ZFP fixed rate for --zfp-mode=rate")
    parser.add_argument("--zfp-precision", type=int, default=32, help="ZFP precision for --zfp-mode=precision")
    parser.add_argument("--zfp-type", choices=["f32", "f64"], default="f64", help="ZFP input scalar type")
    parser.set_defaults(zfp_accuracy_is_relative=True)
    parser.add_argument(
        "--zfp-accuracy-is-relative",
        dest="zfp_accuracy_is_relative",
        action="store_true",
        help="Interpret --error-bound as relative and convert to absolute tolerance for ZFP accuracy mode (default).",
    )
    parser.add_argument(
        "--zfp-accuracy-is-absolute",
        dest="zfp_accuracy_is_relative",
        action="store_false",
        help="Interpret --error-bound directly as ZFP absolute tolerance in accuracy mode.",
    )
    parser.add_argument("--builder", default="./construct_extremum_graph_cuda", help="Graph builder executable")
    parser.add_argument("--compute", default="./compute_fgw.py", help="compute_fgw.py path")
    parser.add_argument("--python", default="python", help="Python executable for compute_fgw.py")
    parser.add_argument("--lut", default="./LUT.bin", help="Path to LUT.bin")
    parser.add_argument("--alpha", type=float, default=0.5, help="fGW alpha")
    parser.add_argument("--no-value-feature", action="store_true", help="Pass --no-value-feature to compute_fgw.py")
    parser.add_argument("--out-root", default="/fs/ess/PAS2402/yuxiao/pipeline_rel_1e-4", help="Output root directory")
    parser.add_argument("--summary-csv", default="/fs/ess/PAS2402/yuxiao/pipeline_rel_1e-4/summary.csv", help="Summary CSV path")
    parser.add_argument("--fgw-csv", default="/fs/ess/PAS2402/yuxiao/pipeline_rel_1e-4/fgw_results.csv", help="FGW result CSV path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    items = parse_datasets_file(args.datasets)
    if len(items) == 0:
        raise ValueError(f"No active datasets found in {args.datasets}")

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.fgw_csv) or ".", exist_ok=True)
    ensure_csv_header(args.summary_csv)

    print(f"Found {len(items)} dataset(s).")
    for item in items:
        print(f"\n=== {item.name} ({item.nx}x{item.ny}x{item.nz}) ===")
        dataset_root = os.path.join(args.out_root, item.name)
        os.makedirs(dataset_root, exist_ok=True)

        cmp_ext = "sz3" if args.codec == "sz3" else "zfp"
        cmp_path = os.path.join(dataset_root, f"{item.name}_{args.codec}_{args.error_bound:.0e}.{cmp_ext}")
        decp_path = os.path.join(dataset_root, f"{item.name}_rel_{args.error_bound:.0e}_decp.bin")

        graph_ori = os.path.join(dataset_root, "graph_ori")
        graph_decp = os.path.join(dataset_root, "graph_decp")

        cmp_size = ""
        decp_size = ""
        t_comp = ""
        t_decomp = ""
        effective_error_bound = args.error_bound

        try:
            if args.codec == "zfp" and args.zfp_mode == "accuracy" and args.zfp_accuracy_is_relative:
                num_values = item.nx * item.ny * item.nz
                data_range = compute_data_range(item.path, num_values, args.zfp_type)
                effective_error_bound = args.error_bound * data_range
                if args.verbose:
                    print(
                        f"zfp_accuracy_tol(dataset={item.name}): rel={args.error_bound:.6e}, "
                        f"range={data_range:.6e}, abs={effective_error_bound:.6e}"
                    )

            # 1) compress
            t0 = time.perf_counter()
            run_cmd(build_compress_cmd(args, item, cmp_path, effective_error_bound), verbose=args.verbose)
            t1 = time.perf_counter()
            t_comp = f"{(t1 - t0):.6f}"

            # 2) decompress
            t2 = time.perf_counter()
            run_cmd(build_decompress_cmd(args, item, cmp_path, decp_path), verbose=args.verbose)
            t3 = time.perf_counter()
            t_decomp = f"{(t3 - t2):.6f}"

            if os.path.exists(cmp_path):
                cmp_size = str(os.path.getsize(cmp_path))
            if os.path.exists(decp_path):
                decp_size = str(os.path.getsize(decp_path))

            # 3) build graph for ori
            run_cmd(
                [
                    args.builder,
                    item.path,
                    str(item.nx),
                    str(item.ny),
                    str(item.nz),
                    graph_ori,
                    args.lut,
                ],
                verbose=args.verbose,
            )

            # 4) build graph for decp
            run_cmd(
                [
                    args.builder,
                    decp_path,
                    str(item.nx),
                    str(item.ny),
                    str(item.nz),
                    graph_decp,
                    args.lut,
                ],
                verbose=args.verbose,
            )

            # 5) compute fGW
            fgw_cmd = [
                args.python,
                args.compute,
                "--graph1",
                graph_ori,
                "--graph2",
                graph_decp,
                "--alpha",
                str(args.alpha),
                "--save-csv",
                args.fgw_csv,
                "--tag",
                item.name,
            ]
            if args.no_value_feature:
                fgw_cmd.append("--no-value-feature")
            if args.verbose:
                fgw_cmd.append("--verbose")
            run_cmd(fgw_cmd, verbose=args.verbose)

            append_row(
                args.summary_csv,
                [
                    item.name,
                    str(item.nx),
                    str(item.ny),
                    str(item.nz),
                    item.path,
                    cmp_path,
                    decp_path,
                    cmp_size,
                    decp_size,
                    t_comp,
                    t_decomp,
                    args.fgw_csv,
                    "ok",
                    "",
                ],
            )
        except Exception as e:
            append_row(
                args.summary_csv,
                [
                    item.name,
                    str(item.nx),
                    str(item.ny),
                    str(item.nz),
                    item.path,
                    cmp_path,
                    decp_path,
                    cmp_size,
                    decp_size,
                    t_comp,
                    t_decomp,
                    args.fgw_csv,
                    "failed",
                    str(e),
                ],
            )
            print(f"[FAILED] {item.name}: {e}")

    print("\nAll done.")
    print(f"summary: {args.summary_csv}")
    print(f"fgw results: {args.fgw_csv}")


if __name__ == "__main__":
    main()
