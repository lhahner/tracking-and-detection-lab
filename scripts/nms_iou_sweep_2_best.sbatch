#!/bin/bash
#SBATCH --partition=scc-gpu
#SBATCH --job-name=nms-2best-iou-sweep
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project/results/nms_iou_sweep_2_best/logs/sweep_%j.out
#SBATCH --error=/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project/results/nms_iou_sweep_2_best/logs/sweep_%j.err

set -euo pipefail

PROJECT_ROOT="/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project"
REPOSITORY_ROOT="/mnt/vast-standard/home/lennart.hahner/u28856/tracking-and-detection-lab"
CONDA_BIN="/mnt/vast-standard/home/lennart.hahner/u28856/miniconda3/bin/conda"
SWEEP_ROOT="${PROJECT_ROOT}/results/nms_iou_sweep_2_best"
NUSCENES_ROOT="${PROJECT_ROOT}/datasets/nuscenes"

mkdir -p "${SWEEP_ROOT}/logs"
cd "${REPOSITORY_ROOT}"
export PYTHONPATH="${REPOSITORY_ROOT}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/mnt/vast-standard/home/lennart.hahner/u28856/miniconda3/envs/track-lab/lib:${LD_LIBRARY_PATH:-}"

"${CONDA_BIN}" run --no-capture-output -n track-lab \
    python scripts/nms_iou_sweep_2_best.py \
    --output-dir "${SWEEP_ROOT}" \
    --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

for threshold in 0p1 0p2 0p3 0p4 0p5 0p6 0p7 0p8 0p9; do
    input_json="${SWEEP_ROOT}/nms_2_best_iou_${threshold}_detections.json"
    result_json="${SWEEP_ROOT}/nms_2_best_iou_${threshold}_results_nusc.json"
    metrics_dir="${SWEEP_ROOT}/nms_2_best_iou_${threshold}_eval"

    "${CONDA_BIN}" run --no-capture-output -n track-lab \
        python scripts/evaluate_nuscenes_from_simpletrack.py "${input_json}" \
        --output-json "${result_json}" \
        --dataroot "${NUSCENES_ROOT}" \
        --version v1.0-trainval \
        --eval-set val \
        --output-dir "${metrics_dir}" \
        --run-eval
done
