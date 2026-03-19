#!/bin/bash
#SBATCH --job-name=fgw-sz3-zfp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=PAS2402
 
set -euo pipefail
module load python/3.12

WORKDIR="/users/PAS2402/yli14/distance_graph"

DATASETS="${WORKDIR}/datasets.txt"
PIPELINE="${WORKDIR}/run_sz3_pipeline.py"
PYTHON_BIN="python"

SZ3_BIN="/users/PAS2402/yli14/bin/sz3"
ZFP_BIN="zfp"
BUILDER_BIN="${WORKDIR}/construct_extremum_graph_cuda"
LUT_BIN="${WORKDIR}/LUT.bin"

ALPHA="0.5"
ERROR_BOUND="1e-4"

OUT_BASE="/fs/ess/PAS2402/yuxiao/pipeline_rel_1e-4"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "[$(date)] Start SZ3 pipeline"
${PYTHON_BIN} "${PIPELINE}" \
  --datasets "${DATASETS}" \
  --codec sz3 \
  --error-bound "${ERROR_BOUND}" \
  --sz3 "${SZ3_BIN}" \
  --builder "${BUILDER_BIN}" \
  --lut "${LUT_BIN}" \
  --alpha "${ALPHA}" \
  --out-root "${OUT_BASE}/sz3" \
  --summary-csv "${OUT_BASE}/summary_sz3.csv" \
  --fgw-csv "${OUT_BASE}/fgw_results_sz3.csv" \
  --no-value-feature \
  --verbose

echo "[$(date)] Start ZFP pipeline"
${PYTHON_BIN} "${PIPELINE}" \
  --datasets "${DATASETS}" \
  --codec zfp \
  --error-bound "${ERROR_BOUND}" \
  --zfp "${ZFP_BIN}" \
  --zfp-mode accuracy \
  --builder "${BUILDER_BIN}" \
  --lut "${LUT_BIN}" \
  --alpha "${ALPHA}" \
  --out-root "${OUT_BASE}/zfp" \
  --summary-csv "${OUT_BASE}/summary_zfp.csv" \
  --fgw-csv "${OUT_BASE}/fgw_results_zfp.csv" \
  --no-value-feature \
  --verbose

echo "[$(date)] All pipelines completed"
