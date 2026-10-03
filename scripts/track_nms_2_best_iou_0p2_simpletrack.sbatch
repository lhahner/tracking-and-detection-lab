#!/bin/bash
#SBATCH --partition=scc-gpu
#SBATCH --job-name=track-nms02-simple
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project/results/nms_iou_sweep_2_best/tracking/logs/simpletrack_%j.out
#SBATCH --error=/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project/results/nms_iou_sweep_2_best/tracking/logs/simpletrack_%j.err

set -euo pipefail

REPOSITORY_ROOT="/mnt/vast-standard/home/lennart.hahner/u28856/tracking-and-detection-lab"
PROJECT_ROOT="/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project"
CONDA_BIN="/mnt/vast-standard/home/lennart.hahner/u28856/miniconda3/bin/conda"
DETECTIONS="${PROJECT_ROOT}/results/nms_iou_sweep_2_best/nms_2_best_iou_0p2_detections.json"
RUN_ROOT="${PROJECT_ROOT}/results/nms_iou_sweep_2_best/tracking/simpletrack"
TRACK_DIR="${RUN_ROOT}/tracks"
EVAL_DIR="${RUN_ROOT}/eval"
TRACK_JSON="${TRACK_DIR}/results_nms_2_best_iou_0p2_detections_tracks.json"

mkdir -p "${TRACK_DIR}" "${EVAL_DIR}"
cd "${REPOSITORY_ROOT}"
export PYTHONPATH="${REPOSITORY_ROOT}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/mnt/vast-standard/home/lennart.hahner/u28856/miniconda3/envs/track-lab/lib:${LD_LIBRARY_PATH:-}"

"${CONDA_BIN}" run --no-capture-output -n track-lab \
    python -u scripts/run_simpletrack_nuscenes.py "${DETECTIONS}" \
    --output-dir "${TRACK_DIR}" \
    --overwrite \
    --write-interval 500

"${CONDA_BIN}" run --no-capture-output -n track-lab \
    python -u scripts/evaluate_nuscenes_tracking.py "${TRACK_JSON}" \
    --dataroot "${PROJECT_ROOT}/datasets/nuscenes" \
    --version v1.0-trainval \
    --eval-set val \
    --output-dir "${EVAL_DIR}" \
    --pad-missing-samples \
    --run-eval
