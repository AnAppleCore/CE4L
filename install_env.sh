#!/usr/bin/env bash
# Independent env for all benchmarks
# - Python via conda
# - Packages via pip (Tsinghua mirror where applicable)
# - PyTorch 2.7.0 + CUDA 12.8

set -e

ENV_NAME=clego
PY_VER=3.12

# 1) create and activate conda env (adjust conda.sh path if needed)
conda create -n ${ENV_NAME} python=${PY_VER} -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# 2) install PyTorch CUDA wheels (use official index, do NOT use mirror here)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3) install core deps from Tsinghua mirror
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  einops ftfy ipdb numpy \
  omegaconf opencv-python pandas \
  Pillow regex setuptools \
  termcolor timm tqdm \
  transformers decord colorama \
  editdistance imageio tensorboardX \
  matplotlib mmengine scikit-learn

# 4) quick checks
python - <<'PY'
import torch, torchvision
import decord
print('torch:', torch.__version__)
print('torch.cuda.is_available:', torch.cuda.is_available())
print('torchvision:', torchvision.__version__)
print('decord ok')
print('tensorboardX ok')
print('mmengine ok')
print('scikit-learn ok')
print('matplotlib ok')
PY

echo "[Done] association_benchmark env '${ENV_NAME}' is ready."

