# Extremum Graph + fGW 使用说明

本目录包含两步流程：

1. 从标量场 `.bin` 构建 extremum graph（输出 `nodes.bin` 和 `edges.bin`）
2. 计算两张图之间的 Fused Gromov-Wasserstein 距离（fGW）

---

## 1) 文件说明

- `construct_extremum_graph.cpp`：构图程序源码（内部支持分类 + 建图）
- `construct_extremum_graph`：构图可执行文件
- `compute_fgw.py`：fGW 计算脚本
- `LUT.bin`：分类所需查找表

输出图文件格式：

- `nodes.bin`：每条记录为 `[int id, int grid_id, int type, int x, int y, int z, double value]`
- `edges.bin`：每条记录为 `[int u, int v, int edge_type, double weight]`

其中：
- `type` 映射：`0=min, 1=max, 2=saddle1, 3=saddle2`
- `edge_type` 映射：
  - `0`：`saddle1 -> max`
  - `1`：`saddle2 -> min`

---

## 2) 环境准备

### C++ 构图程序

```bash
g++ -O3 -std=c++17 construct_extremum_graph.cpp -o construct_extremum_graph
```

### Python fGW 环境（推荐 3.12 venv）

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
./.venv312/bin/python -m pip install -U pip
./.venv312/bin/python -m pip install numpy scipy POT
```

---

## 3) 从 `.bin` 构建 extremum graph

### 用法 A（推荐，显式指定输出目录和 LUT）

```bash
./construct_extremum_graph <scalar.bin> <nx> <ny> <nz> <out_dir> <LUT.bin>
```

示例：

```bash
./construct_extremum_graph ~/Downloads/at.bin 177 95 48 ./graph_at ./LUT.bin
```

### 用法 B（已有 `vertex_type.bin` 时）

```bash
./construct_extremum_graph <scalar.bin> <vertex_type.bin> <nx> <ny> <nz> <out_dir>
```

输出：
- `<out_dir>/nodes.bin`
- `<out_dir>/edges.bin`

---

## 4) 计算 fGW

`compute_fgw.py` 支持三种输入方式：

### 方式 1：输入两个图目录

```bash
./.venv312/bin/python compute_fgw.py \
  --graph1 ./graph_A \
  --graph2 ./graph_B \
  --alpha 0.5 --verbose
```

### 方式 2：直接输入两对文件

```bash
./.venv312/bin/python compute_fgw.py \
  --graph1-files ./graph_A/nodes.bin ./graph_A/edges.bin \
  --graph2-files ./graph_B/nodes.bin ./graph_B/edges.bin \
  --alpha 0.5 --verbose
```

如果你习惯误写参数，`--gragh2-files` 也兼容。

### 方式 3：从两个 scalar 自动“建图+计算”

```bash
./.venv312/bin/python compute_fgw.py \
  --scalar1 A.bin --nx1 NX1 --ny1 NY1 --nz1 NZ1 \
  --scalar2 B.bin --nx2 NX2 --ny2 NY2 --nz2 NZ2 \
  --builder ./construct_extremum_graph \
  --lut ./LUT.bin \
  --out1 ./graph_A --out2 ./graph_B \
  --alpha 0.5 --verbose
```

### 单次结果保存到 CSV

```bash
./.venv312/bin/python compute_fgw.py \
  --graph1-files ./graph_A/nodes.bin ./graph_A/edges.bin \
  --graph2-files ./graph_B/nodes.bin ./graph_B/edges.bin \
  --alpha 0.5 \
  --save-csv ./fgw_results.csv \
  --tag A_vs_B
```

`fgw_results.csv` 列为：`data1,data2,tag,alpha,fgw2,fgw`。

其中 `data1/data2` 会自动从输入来源推断（图目录名、nodes 文件名或 scalar 文件名）。

### 批量计算并输出 CSV

先准备 `pairs.csv`：

```csv
nodes1,edges1,nodes2,edges2,alpha,tag
./graph_A/nodes.bin,./graph_A/edges.bin,./graph_B/nodes.bin,./graph_B/edges.bin,0.5,A_vs_B
./graph_A/nodes.bin,./graph_A/edges.bin,./graph_C/nodes.bin,./graph_C/edges.bin,0.5,A_vs_C
```

运行：

```bash
./.venv312/bin/python batch_fgw.py --pairs ./pairs.csv --out ./fgw_batch.csv --verbose
```

---

## 5) 输出解释

脚本会输出两种值：

- `FGW2(alpha=...)`：POT 优化目标值（平方形式）
- `FGW(alpha=...)`：`sqrt(FGW2)`，更直观的距离尺度

默认会对图结构距离矩阵做归一化，避免出现非常大的数值。
如需关闭归一化：

```bash
--no-normalize-structure
```

---

## 6) 常见问题

### Q1: `filesystem error: in create_directories: File exists ["./LUT.bin"]`

你把 `LUT.bin` 放到了 `out_dir` 参数位置。请使用：

```bash
./construct_extremum_graph <scalar.bin> <nx> <ny> <nz> <out_dir> <LUT.bin>
```

### Q2: `unrecognized arguments: --gragh2-files ...`

正确参数是 `--graph2-files`。当前脚本也兼容了 `--gragh2-files` 拼写。

### Q3: `Invalid nodes.bin size`

说明输入文件不是本项目定义格式，或文件路径写错。请确认：
- `nodes.bin` 来自本项目构图器
- `edges.bin` 与之配套

---

## 7) 最小可复现实验

```bash
# 1) 生成两张图
./construct_extremum_graph at.bin 177 95 48 graph1 LUT.bin
./construct_extremum_graph at_simplified_1e-05.bin 177 95 48 graph2 LUT.bin

# 2) 计算 fGW
./.venv312/bin/python compute_fgw.py \
  --graph1 graph1 --graph2 graph2 \
  --alpha 0.5 --verbose
```
