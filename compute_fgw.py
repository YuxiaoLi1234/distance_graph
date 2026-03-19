#!/usr/bin/env python3
import argparse
import os
import struct
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import ot


NODE_RECORD = struct.Struct("<iiiiiid")
EDGE_RECORD = struct.Struct("<iiid")


@dataclass
class GraphData:
    node_ids: np.ndarray
    node_types: np.ndarray
    coords: np.ndarray
    values: np.ndarray
    edges: np.ndarray


def read_nodes(nodes_path: str):
    with open(nodes_path, "rb") as f:
        raw = f.read()
    if len(raw) % NODE_RECORD.size != 0:
        raise ValueError(f"Invalid nodes.bin size: {nodes_path}")
    n = len(raw) // NODE_RECORD.size
    ids = np.empty(n, dtype=np.int32)
    types = np.empty(n, dtype=np.int32)
    coords = np.empty((n, 3), dtype=np.float64)
    values = np.empty(n, dtype=np.float64)

    for i in range(n):
        off = i * NODE_RECORD.size
        rec = NODE_RECORD.unpack_from(raw, off)
        # [id, grid_id, type, x, y, z, value]
        ids[i] = rec[0]
        types[i] = rec[2]
        coords[i, 0] = float(rec[3])
        coords[i, 1] = float(rec[4])
        coords[i, 2] = float(rec[5])
        values[i] = rec[6]

    return ids, types, coords, values


def read_edges(edges_path: str):
    with open(edges_path, "rb") as f:
        raw = f.read()
    if len(raw) % EDGE_RECORD.size != 0:
        raise ValueError(f"Invalid edges.bin size: {edges_path}")
    m = len(raw) // EDGE_RECORD.size
    edges = np.empty((m, 4), dtype=np.float64)
    for i in range(m):
        off = i * EDGE_RECORD.size
        u, v, edge_type, w = EDGE_RECORD.unpack_from(raw, off)
        edges[i, 0] = u
        edges[i, 1] = v
        edges[i, 2] = edge_type
        edges[i, 3] = w
    return edges


def load_graph(graph_dir: str) -> GraphData:
    nodes_path = os.path.join(graph_dir, "nodes.bin")
    edges_path = os.path.join(graph_dir, "edges.bin")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Missing nodes.bin in {graph_dir}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing edges.bin in {graph_dir}")

    ids, types, coords, values = read_nodes(nodes_path)
    edges = read_edges(edges_path)
    return GraphData(ids, types, coords, values, edges)


def load_graph_from_files(nodes_path: str, edges_path: str) -> GraphData:
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    ids, types, coords, values = read_nodes(nodes_path)
    edges = read_edges(edges_path)
    return GraphData(ids, types, coords, values, edges)


def normalize_columns(x: np.ndarray) -> np.ndarray:
    out = x.copy().astype(np.float64)
    for j in range(out.shape[1]):
        col = out[:, j]
        mn = float(np.min(col))
        mx = float(np.max(col))
        if mx > mn:
            out[:, j] = (col - mn) / (mx - mn)
        else:
            out[:, j] = 0.0
    return out


def normalize_matrix_nonnegative(m: np.ndarray) -> np.ndarray:
    out = m.copy().astype(np.float64)
    finite = np.isfinite(out)
    if not np.any(finite):
        return np.zeros_like(out)
    max_val = float(np.max(out[finite]))
    if max_val > 0:
        out[finite] = out[finite] / max_val
    out[~finite] = 1.0
    return out


def build_structural_matrix(graph: GraphData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = graph.node_ids.shape[0]
    id_to_idx: Dict[int, int] = {int(node_id): i for i, node_id in enumerate(graph.node_ids.tolist())}

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    degree = np.zeros(n, dtype=np.float64)

    for e in graph.edges:
        u = int(e[0])
        v = int(e[1])
        w = float(e[3])
        if u not in id_to_idx or v not in id_to_idx:
            continue
        iu = id_to_idx[u]
        iv = id_to_idx[v]
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        vals.extend([w, w])
        degree[iu] += 1.0
        degree[iv] += 1.0

    if len(vals) == 0:
        # no edges: fallback to identity-like distances
        C = np.zeros((n, n), dtype=np.float64)
        p = np.full(n, 1.0 / n, dtype=np.float64)
        feat = normalize_columns(np.hstack([graph.coords, graph.values[:, None]]))
        return C, p, feat

    mat = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    C = shortest_path(mat, directed=False, unweighted=False)

    finite = np.isfinite(C)
    if np.any(finite):
        max_finite = float(np.max(C[finite]))
        fill_value = max(1.0, 2.0 * max_finite)
    else:
        fill_value = 1.0
    C[~finite] = fill_value

    if np.sum(degree) > 0:
        p = degree / np.sum(degree)
    else:
        p = np.full(n, 1.0 / n, dtype=np.float64)

    feat = normalize_columns(np.hstack([graph.coords, graph.values[:, None]]))
    return C, p, feat


def compute_fgw(
    graph1: GraphData,
    graph2: GraphData,
    alpha: float,
    verbose: bool,
    normalize_structure: bool,
):
    C1, p1, feat1 = build_structural_matrix(graph1)
    C2, p2, feat2 = build_structural_matrix(graph2)

    if normalize_structure:
        C1 = normalize_matrix_nonnegative(C1)
        C2 = normalize_matrix_nonnegative(C2)

    M = ot.dist(feat1, feat2, metric="euclidean")

    # POT returns FGW objective value (for square loss setting)
    dist, log = ot.gromov.fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p1,
        p2,
        alpha=alpha,
        loss_fun="square_loss",
        armijo=True,
        log=True,
    )

    if verbose:
        print(f"n1={len(p1)}, n2={len(p2)}")
        print(f"iters={log.get('niter', 'NA')}")

    return float(dist)


def run_graph_builder(
    builder: str,
    scalar_path: str,
    nx: int,
    ny: int,
    nz: int,
    out_dir: str,
    lut_path: str,
):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [builder, scalar_path, str(nx), str(ny), str(nz), out_dir, lut_path]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute fused Gromov-Wasserstein (fGW) between two extremum graphs."
    )

    parser.add_argument("--graph1", type=str, default=None, help="Graph 1 directory containing nodes.bin and edges.bin")
    parser.add_argument("--graph2", type=str, default=None, help="Graph 2 directory containing nodes.bin and edges.bin")
    parser.add_argument("--gragh2", dest="graph2", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--graph1-files",
        nargs=2,
        metavar=("NODES1", "EDGES1"),
        default=None,
        help="Graph 1 explicit files: nodes.bin edges.bin",
    )
    parser.add_argument(
        "--graph2-files",
        nargs=2,
        metavar=("NODES2", "EDGES2"),
        default=None,
        help="Graph 2 explicit files: nodes.bin edges.bin",
    )
    parser.add_argument(
        "--gragh2-files",
        dest="graph2_files",
        nargs=2,
        metavar=("NODES2", "EDGES2"),
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--scalar1", type=str, default=None, help="Scalar field 1 .bin")
    parser.add_argument("--scalar2", type=str, default=None, help="Scalar field 2 .bin")
    parser.add_argument("--nx1", type=int, default=None)
    parser.add_argument("--ny1", type=int, default=None)
    parser.add_argument("--nz1", type=int, default=None)
    parser.add_argument("--nx2", type=int, default=None)
    parser.add_argument("--ny2", type=int, default=None)
    parser.add_argument("--nz2", type=int, default=None)

    parser.add_argument("--builder", type=str, default="./construct_extremum_graph", help="Path to graph builder executable")
    parser.add_argument("--lut", type=str, default="./LUT.bin", help="Path to LUT.bin")
    parser.add_argument("--out1", type=str, default="./graph1")
    parser.add_argument("--out2", type=str, default="./graph2")

    parser.add_argument("--alpha", type=float, default=0.5, help="FGW tradeoff alpha in [0,1]")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-normalize-structure",
        action="store_true",
        help="Disable normalization of structural distance matrices before FGW.",
    )

    args = parser.parse_args()

    if args.graph1_files and args.graph2_files:
        g1 = load_graph_from_files(args.graph1_files[0], args.graph1_files[1])
        g2 = load_graph_from_files(args.graph2_files[0], args.graph2_files[1])
    elif args.graph1 and args.graph2:
        g1 = load_graph(args.graph1)
        g2 = load_graph(args.graph2)
    else:
        required = [
            args.scalar1,
            args.scalar2,
            args.nx1,
            args.ny1,
            args.nz1,
            args.nx2,
            args.ny2,
            args.nz2,
        ]
        if any(v is None for v in required):
            raise ValueError(
                "Provide either --graph1/--graph2, or provide full scalar inputs --scalar1/2 + dims."
            )

        run_graph_builder(args.builder, args.scalar1, args.nx1, args.ny1, args.nz1, args.out1, args.lut)
        run_graph_builder(args.builder, args.scalar2, args.nx2, args.ny2, args.nz2, args.out2, args.lut)

        g1 = load_graph(args.out1)
        g2 = load_graph(args.out2)

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")

    fgw2 = compute_fgw(
        g1,
        g2,
        alpha=args.alpha,
        verbose=args.verbose,
        normalize_structure=(not args.no_normalize_structure),
    )
    fgw = float(np.sqrt(max(0.0, fgw2)))
    print(f"FGW2(alpha={args.alpha}): {fgw2:.12f}")
    print(f"FGW(alpha={args.alpha}): {fgw:.12f}")


if __name__ == "__main__":
    main()
