#!/bin/bash
# ============================================================
# Install script for LUMI (AMD ROCm environment)
# Run once before opening notebooks in Open OnDemand.
#
# Usage:
#   module load LUMI/24.03 partition/G
#   module load rocm/6.2.2
#   bash scripts/install_deps.sh
# ============================================================

set -euo pipefail

VENV_DIR="${PWD}/.venv"

echo "==> Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing PyTorch with ROCm 6.2 support"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

echo "==> Installing project dependencies"
pip install \
    transformers>=4.50.0 \
    accelerate>=1.0.0 \
    peft>=0.13.0 \
    datasets>=3.0.0 \
    trl>=0.12.0 \
    tokenizers>=0.20.0 \
    sentencepiece \
    jupyterlab>=4.2.0 \
    ipywidgets>=8.1.0 \
    huggingface_hub>=0.25.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    matplotlib>=3.9.0 \
    tqdm>=4.66.0 \
    psutil>=6.0.0 \
    wandb>=0.18.0 \
    tensorboard>=2.18.0

echo "==> Registering Jupyter kernel"
pip install ipykernel
python -m ipykernel install --user --name qwen-lumi --display-name "Python (qwen-lumi ROCm)"

echo ""
echo "Done! Activate with:  source ${VENV_DIR}/bin/activate"
echo "Then launch:          jupyter lab"
