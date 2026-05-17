#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="${ENV_NAME:-track-lab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.20}"
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
  echo "Conda is required but was not found in PATH." >&2
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

PYTHON_MAJOR_MINOR="$(printf '%s' "${PYTHON_VERSION}" | cut -d. -f1-2)"
case "${PYTHON_MAJOR_MINOR}" in
  3.10|3.11|3.12)
    ;;
  *)
    echo "Unsupported Python version for the pinned dependency set: ${PYTHON_VERSION}" >&2
    echo "Use Python 3.10, 3.11, or 3.12 with torch ${PYTORCH_VERSION}, torchvision ${TORCHVISION_VERSION}, and open3d 0.19.0." >&2
    exit 1
    ;;
esac

echo "Installing packaging tools"
# Detectron2 still imports pkg_resources, which is no longer present in setuptools 81+.
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"

echo "Installing helper build/runtime packages"
python -m pip install ninja "opencv-python==4.10.0.84"

echo "Installing PyTorch ${PYTORCH_VERSION} and torchvision ${TORCHVISION_VERSION} from ${PYTORCH_INDEX_URL}"
conda install "pytorch" "torchvision" -c pytorch
conda install pytorch3d -c pytorch3d

echo "Installing NumPy < 2 for motmetrics compatibility"
python -m pip install "numpy<2"

echo "veryfing glibc version for open3d"
OPEN3D_VIA_CONDA=0
if [[ "$(uname -s)" == "Linux" ]]; then
  GLIBC_VERSION="$(ldd --version 2>/dev/null | awk '{print $NF}' | head -n1)"
  if [[ -n "${GLIBC_VERSION}" ]] && [[ "${GLIBC_VERSION}" != "2.31" ]]; then
    OPEN3D_VIA_CONDA=1
  fi
fi

echo "Installing project requirements"
if [[ "${OPEN3D_VIA_CONDA}" == "1" ]]; then
  echo "Detected glibc ${GLIBC_VERSION}; installing Open3D from conda-forge because pip wheels require glibc >= 2.31"
  TMP_REQUIREMENTS="$(mktemp)"
  grep -v '^open3d==' "${REPO_ROOT}/requirements.txt" > "${TMP_REQUIREMENTS}"
  python -m pip install -r "${TMP_REQUIREMENTS}"
  rm -f "${TMP_REQUIREMENTS}"
  conda install -n "${ENV_NAME}" -c conda-forge open3d -y
else
  python -m pip install -r "${REPO_ROOT}/requirements.txt"
fi

if [[ "${INSTALL_DETECTRON2}" == "1" ]]; then
  echo "Installing Conda C/C++ toolchain for detectron2"
  conda install -n "${ENV_NAME}" -c conda-forge gcc_linux-64 gxx_linux-64 -y

  CONDA_BIN_DIR="$(python - <<'PY'
import sys
from pathlib import Path

print(Path(sys.prefix) / "bin")
PY
)"
  export CC="${CONDA_BIN_DIR}/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_BIN_DIR}/x86_64-conda-linux-gnu-g++"

  echo "Using CC=${CC}"
  echo "Using CXX=${CXX}"
  echo "Installing detectron2 from source"
  python -m pip install --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
else
  echo "Skipping detectron2 installation"
fi

if [[ "${INSTALL_MMDET3D}" == "1" ]]; then
  echo "WARNING - MMDetection3D needs gpu, use gcc 13.2.0 and nvcc 11.8.0"
  echo "Installing OpenMMLab package manager"
  python -m pip install -U openmim

  echo "Installing MMEngine"
  mim install mmengine

  echo "Installing mmcv"
  mim install "mmcv==2.1.0"

  echo "Installing MMDetection"
  mim install 'mmdet>=3.0.0' 

  echo "Installing mmdet3d"
  mim install "mmdet3d>=1.1.0"

# ---- This part covers the CPU verison of mmdet3d which is not working properly ----  
#  if [[ $(lspci | grep -i '.* NVIDIA .*') && ! $CUDA_FLAVOR =~ cpu ]]; then
#  	echo "Installing MMDetection3D from fork without build isolation: ${MMDET3D_REPO_URL}"
#  	mim install "mmdet3d>=1.1.0" 
#  else
#    echo "No Nvidia GPU detected can't install MMDetection3D normally, instead using custom cpu-only"
#    if [[ ! -d "./external" ]]; then
#      mkdir external
#    fi
#    if [[ ! -d "./external/mmdetection3d-cpu-only" ]]; then
#      git clone "${MMDET3D_REPO_URL}" ${REPO_ROOT}/external/mmdetection3d-cpu-only
#    fi
#  fi
else
  echo "Skipping MMDetection3D installation"
fi

echo "Verifying GLIBCXX version"
if [[ $CONDA_DEFAULT_ENV == track-lab ]]; then
	conda install -c conda-forge libstdcxx-ng libgcc-ng
	conda install -c conda-forge gxx_linux-64 gcc_linux-64
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
