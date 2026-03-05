# LUMI Open OnDemand — Qwen3.5-35B-A3B Fine-Tuning Demo

A Jupyter notebook project for fine-tuning and running inference with
[Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) on
[LUMI](https://www.lumi-supercomputer.eu/) via Open OnDemand.

## Model Overview

| Property | Value |
|---|---|
| Architecture | Gated DeltaNet + sparse MoE |
| Total parameters | 35B |
| Active parameters | 3B (8 routed + 1 shared experts of 256) |
| Context length | 262,144 tokens (native) |
| Thinking mode | Enabled by default (`<think>...</think>`) |
| Modality | Text + Vision (multimodal) |

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 01 | `notebooks/01_environment_test.ipynb` | Verify GPU, ROCm/CUDA, and package setup |
| 02 | `notebooks/02_inference.ipynb` | Load Qwen3.5-35B-A3B and run inference |
| 03 | `notebooks/03_finetune_lora.ipynb` | LoRA fine-tuning with PEFT + TRL |

## Quick Start on LUMI

### 1. Load modules

```bash
module load LUMI/24.03 partition/G
module load rocm/6.2.2
```

### 2. Install dependencies

The install script places the virtual environment in
`/scratch/project_465002745/.venv`.

```bash
bash scripts/install_deps.sh
```

### 3. Set HuggingFace token (model may require access)

```bash
export HF_TOKEN="hf_your_token_here"
# or
huggingface-cli login
```

### 4. Open OnDemand

1. Go to your LUMI Open OnDemand portal
2. Launch **Jupyter Lab** with GPU resources (recommend ≥ 1× MI250X)
3. Select kernel **Python (qwen-lumi ROCm)**
4. Open notebooks in order: 01 → 02 → 03

## Resource Requirements

| Task | Recommended GPUs | VRAM |
|---|---|---|
| Inference (bf16) | 1× MI250X (64 GB) | ~30 GB |
| LoRA fine-tuning (bf16) | 1–2× MI250X | ~40–60 GB |
| Full fine-tuning | 4× MI250X | ≥ 128 GB |

> **Note:** The MoE design activates only 3B parameters per forward pass,
> making inference significantly cheaper than a dense 35B model.

## Project Structure

```
LUMI-openOnDemand-demo/
├── notebooks/
│   ├── 01_environment_test.ipynb
│   ├── 02_inference.ipynb
│   └── 03_finetune_lora.ipynb
├── scripts/
│   └── install_deps.sh
├── data/                  # place your training data here (git-ignored)
├── checkpoints/           # saved LoRA adapters (git-ignored)
├── requirements.txt
├── environment.yml
└── README.md
```

## License

This project is released under the MIT License.
The Qwen3.5 model weights are subject to the
[Qwen License Agreement](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE).
