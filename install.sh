#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-track-lab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cpu}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-1}"
INSTALL_MMDET3D="${INSTALL_MMDET3D:-0}"
MMDET3D_REPO_URL="${MMDET3D_REPO_URL:-https://github.com/lhahner/mmdetection3d-cpu-only.git}"

usage() {
  cat <<'EOF'
Usage:
  bash install.sh [options]

Options:
  --env NAME              Conda environment name. Default: track-lab
  --python VERSION        Python version. Default: 3.10
  --cuda FLAVOR           One of: cpu, cu121, cu124. Default: cpu
  --without-detectron2    Skip detectron2 installation
  --with-mmdet3d          Install MMDetection3D and its OpenMMLab dependencies
  --help                  Show this help

Examples:
  bash install.sh
  bash install.sh --cuda cu121
  bash install.sh --env track-lab-gpu --cuda cu124
  bash install.sh --with-mmdet3d
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --env" >&2
        usage
        exit 1
      fi
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --python" >&2
        usage
        exit 1
      fi
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --cuda)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --cuda" >&2
        usage
        exit 1
      fi
      CUDA_FLAVOR="$2"
      shift 2
      ;;
    --without-detectron2)
      INSTALL_DETECTRON2=0
      shift
      ;;
    --with-mmdet3d)
      INSTALL_MMDET3D=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

case "${CUDA_FLAVOR}" in
  cpu|cu121|cu124)
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_FLAVOR}"
    ;;
  *)
    echo "Unsupported CUDA flavor: ${CUDA_FLAVOR}. Use one of: cpu, cu121, cu124." >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fx "${ENV_NAME}" >/dev/null 2>&1; then
  echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
echo "Activating '${ENV_NAME}'"
conda activate "${ENV_NAME}"

echo "Installing packaging tools"
# Detectron2 still imports pkg_resources, which is no longer present in setuptools 81+.
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"

echo "Installing helper build/runtime packages"
python -m pip install ninja opencv-python

echo "Installing PyTorch ${PYTORCH_VERSION} and torchvision ${TORCHVISION_VERSION} from ${PYTORCH_INDEX_URL}"
python -m pip install \
  "torch==${PYTORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

echo "Installing NumPy < 2 for motmetrics compatibility"
python -m pip install "numpy<2"

echo "Installing project requirements"
python -m pip install -r "${REPO_ROOT}/requirements.txt"

if [[ "${INSTALL_DETECTRON2}" == "1" ]]; then
  echo "Installing detectron2 from source"
  python -m pip install --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
else
  echo "Skipping detectron2 installation"
fi

if [[ "${INSTALL_MMDET3D}" == "1" ]]; then
  echo "Installing OpenMMLab package manager"
  python -m pip install -U openmim

  echo "Installing MMEngine"
  mim install mmengine

  echo "Installing mmcv-lite for CPU-oriented MMDetection3D usage"
  mim install "mmcv-lite>=2.0.0rc4,<2.2.0"

  echo "Installing MMDetection"
  mim install "mmdet>=3.0.0,<3.4.0"

  if [[ $(lshw -C display | grep vendor) =~ Nvidia && ! $CUDA_FLAVOR =~ cpu ]]; then
  	echo "Installing MMDetection3D from fork without build isolation: ${MMDET3D_REPO_URL}"
  	mim install "mmdet3d>=1.1.0" 
  else
    echo "No Nvidia GPU detected can't install MMDetection3D normally, instead using custom cpu-only"
    if [[ ! -d "./external" ]]; then
      mkdir external
    fi
    if [[ ! -d "./external/mmdetection3d-cpu-only" ]]; then
      git clone "${MMDET3D_REPO_URL}" ./external/mmdetection3d-cpu-only
    fi
  fi
else
  echo "Skipping MMDetection3D installation"
fi

echo "Verifying installed packages"
python - <<'PY'
import importlib.util
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")

if importlib.util.find_spec("detectron2") is not None:
    import detectron2
    print(f"detectron2={getattr(detectron2, '__version__', 'installed')}")
else:
    print("detectron2=not-installed")

if importlib.util.find_spec("mmdet3d") is not None:
    import mmdet3d
    print(f"mmdet3d={getattr(mmdet3d, '__version__', 'installed')}")
else:
    print("mmdet3d=not-installed")

print(f"timm_installed={importlib.util.find_spec('timm') is not None}")
PY

echo
echo "Installation complete."
echo "Note: detectron2 may emit a pkg_resources deprecation warning with setuptools<81."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run the app with: PYTHONPATH=src python src/app.py"
