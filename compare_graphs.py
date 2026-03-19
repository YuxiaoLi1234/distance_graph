#!/usr/bin/env python3
import argparse
import os
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


NODE_RECORD = struct.Struct("<iiiiiid")
EDGE_RECORD = struct.Struct("<iiid")


@dataclass
class GraphData:
    nodes: np.ndarray
    edges: np.ndarray


def read_nodes(nodes_path: str) -> np.ndarray:
    with open(nodes_path, "rb") as f:
        raw = f.read()
    if len(raw) % NODE_RECORD.size != 0:
        raise ValueError(f"Invalid nodes.bin size: {nodes_path}")

    n = len(raw) // NODE_RECORD.size
    arr = np.zeros(
        n,
        dtype=[
            ("id", np.int32),
            ("grid_id", np.int32),
            ("type", np.int32),
            ("x", np.int32),
            ("y", np.int32),
            ("z", np.int32),
            ("value", np.float64),
        ],
    )

    for i in range(n):
        off = i * NODE_RECORD.size
        rec = NODE_RECORD.unpack_from(raw, off)
        arr[i] = rec

    return arr


def read_edges(edges_path: str) -> np.ndarray:
    with open(edges_path, "rb") as f:
        raw = f.read()
    if len(raw) % EDGE_RECORD.size != 0:
        raise ValueError(f"Invalid edges.bin size: {edges_path}")

    m = len(raw) // EDGE_RECORD.size
    arr = np.zeros(
        m,
        dtype=[
            ("u", np.int32),
            ("v", np.int32),
            ("edge_type", np.int32),
            ("weight", np.float64),
        ],
    )

    for i in range(m):
        off = i * EDGE_RECORD.size
        rec = EDGE_RECORD.unpack_from(raw, off)
        arr[i] = rec

    return arr


def load_graph_from_dir(graph_dir: str) -> GraphData:
    nodes_path = os.path.join(graph_dir, "nodes.bin")
    edges_path = os.path.join(graph_dir, "edges.bin")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Missing nodes.bin in {graph_dir}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing edges.bin in {graph_dir}")

    return GraphData(read_nodes(nodes_path), read_edges(edges_path))


def load_graph_from_files(nodes_path: str, edges_path: str) -> GraphData:
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    return GraphData(read_nodes(nodes_path), read_edges(edges_path))


def type_name(t: int) -> str:
    return {0: "min", 1: "max", 2: "saddle1", 3: "saddle2"}.get(int(t), f"unknown({t})")


def summarize_types(nodes: np.ndarray) -> Dict[int, int]:
    c = Counter(nodes["type"].tolist())
    return {k: int(v) for k, v in sorted(c.items())}


def node_index_by_grid(nodes: np.ndarray) -> Dict[int, np.void]:
    out: Dict[int, np.void] = {}
    for n in nodes:
        out[int(n["grid_id"])] = n
    return out


def edge_index(edges: np.ndarray) -> Dict[Tuple[int, int, int], float]:
    out: Dict[Tuple[int, int, int], float] = {}
    for e in edges:
        out[(int(e["u"]), int(e["v"]), int(e["edge_type"]))] = float(e["weight"])
    return out


def print_node_diff(g1: GraphData, g2: GraphData, topk: int, value_tol: float):
    n1 = g1.nodes
    n2 = g2.nodes
    idx1 = node_index_by_grid(n1)
    idx2 = node_index_by_grid(n2)

    grids1 = set(idx1.keys())
    grids2 = set(idx2.keys())

    only1 = sorted(grids1 - grids2)
    only2 = sorted(grids2 - grids1)
    both = sorted(grids1 & grids2)

    print("\n[Node Summary]")
    print(f"- graph1 nodes: {len(n1)}")
    print(f"- graph2 nodes: {len(n2)}")
    print(f"- common grid_id: {len(both)}")
    print(f"- only in graph1: {len(only1)}")
    print(f"- only in graph2: {len(only2)}")

    t1 = summarize_types(n1)
    t2 = summarize_types(n2)
    print("- type count graph1:")
    for k, v in t1.items():
        print(f"    {k} ({type_name(k)}): {v}")
    print("- type count graph2:")
    for k, v in t2.items():
        print(f"    {k} ({type_name(k)}): {v}")

    type_changed: List[Tuple[int, int, int]] = []
    value_changed: List[Tuple[int, float, float, float]] = []

    for gid in both:
        a = idx1[gid]
        b = idx2[gid]
        if int(a["type"]) != int(b["type"]):
            type_changed.append((gid, int(a["type"]), int(b["type"])))
        dv = abs(float(a["value"]) - float(b["value"]))
        if dv > value_tol:
            value_changed.append((gid, float(a["value"]), float(b["value"]), dv))

    print(f"- type changed on common grid_id: {len(type_changed)}")
    print(f"- value changed (>|{value_tol}|) on common grid_id: {len(value_changed)}")

    if only1:
        print(f"\n[Nodes only in graph1] (top {topk})")
        for gid in only1[:topk]:
            a = idx1[gid]
            print(
                f"  grid_id={gid}, type={int(a['type'])}/{type_name(int(a['type']))}, "
                f"xyz=({int(a['x'])},{int(a['y'])},{int(a['z'])}), value={float(a['value']):.6g}"
            )

    if only2:
        print(f"\n[Nodes only in graph2] (top {topk})")
        for gid in only2[:topk]:
            b = idx2[gid]
            print(
                f"  grid_id={gid}, type={int(b['type'])}/{type_name(int(b['type']))}, "
                f"xyz=({int(b['x'])},{int(b['y'])},{int(b['z'])}), value={float(b['value']):.6g}"
            )

    if type_changed:
        print(f"\n[Type changed nodes] (top {topk})")
        for gid, ta, tb in type_changed[:topk]:
            print(f"  grid_id={gid}: graph1={ta}/{type_name(ta)} -> graph2={tb}/{type_name(tb)}")

    if value_changed:
        value_changed.sort(key=lambda x: x[3], reverse=True)
        print(f"\n[Value changed nodes] (top {topk} by abs diff)")
        for gid, va, vb, dv in value_changed[:topk]:
            print(f"  grid_id={gid}: v1={va:.8g}, v2={vb:.8g}, abs_diff={dv:.8g}")


def print_edge_diff(g1: GraphData, g2: GraphData, topk: int, weight_tol: float):
    e1 = edge_index(g1.edges)
    e2 = edge_index(g2.edges)

    keys1 = set(e1.keys())
    keys2 = set(e2.keys())

    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)
    both = sorted(keys1 & keys2)

    print("\n[Edge Summary]")
    print(f"- graph1 edges: {len(g1.edges)}")
    print(f"- graph2 edges: {len(g2.edges)}")
    print(f"- common edges: {len(both)}")
    print(f"- only in graph1: {len(only1)}")
    print(f"- only in graph2: {len(only2)}")

    changed = []
    for k in both:
        w1 = e1[k]
        w2 = e2[k]
        dw = abs(w1 - w2)
        if dw > weight_tol:
            changed.append((k, w1, w2, dw))

    print(f"- weight changed (>|{weight_tol}|) on common edges: {len(changed)}")

    if only1:
        print(f"\n[Edges only in graph1] (top {topk})")
        for (u, v, t) in only1[:topk]:
            print(f"  (u={u}, v={v}, type={t}) w={e1[(u, v, t)]:.8g}")

    if only2:
        print(f"\n[Edges only in graph2] (top {topk})")
        for (u, v, t) in only2[:topk]:
            print(f"  (u={u}, v={v}, type={t}) w={e2[(u, v, t)]:.8g}")

    if changed:
        changed.sort(key=lambda x: x[3], reverse=True)
        print(f"\n[Edge weight changed] (top {topk} by abs diff)")
        for (u, v, t), w1, w2, dw in changed[:topk]:
            print(f"  (u={u}, v={v}, type={t}): w1={w1:.8g}, w2={w2:.8g}, abs_diff={dw:.8g}")


def main():
    parser = argparse.ArgumentParser(description="Compare two extremum graphs and print where they differ.")

    parser.add_argument("--graph1", type=str, default=None, help="Graph1 dir containing nodes.bin and edges.bin")
    parser.add_argument("--graph2", type=str, default=None, help="Graph2 dir containing nodes.bin and edges.bin")
    parser.add_argument("--graph1-files", nargs=2, metavar=("NODES1", "EDGES1"), default=None)
    parser.add_argument("--graph2-files", nargs=2, metavar=("NODES2", "EDGES2"), default=None)
    parser.add_argument("--gragh2-files", dest="graph2_files", nargs=2, metavar=("NODES2", "EDGES2"), default=None, help=argparse.SUPPRESS)

    parser.add_argument("--topk", type=int, default=20, help="Print top-k detailed differences")
    parser.add_argument("--value-tol", type=float, default=1e-12, help="Node value difference tolerance")
    parser.add_argument("--weight-tol", type=float, default=1e-12, help="Edge weight difference tolerance")

    args = parser.parse_args()

    if args.graph1_files and args.graph2_files:
        g1 = load_graph_from_files(args.graph1_files[0], args.graph1_files[1])
        g2 = load_graph_from_files(args.graph2_files[0], args.graph2_files[1])
    elif args.graph1 and args.graph2:
        g1 = load_graph_from_dir(args.graph1)
        g2 = load_graph_from_dir(args.graph2)
    else:
        raise ValueError("Provide either --graph1/--graph2 or --graph1-files/--graph2-files")

    print_node_diff(g1, g2, topk=args.topk, value_tol=args.value_tol)
    print_edge_diff(g1, g2, topk=args.topk, weight_tol=args.weight_tol)


if __name__ == "__main__":
    main()
