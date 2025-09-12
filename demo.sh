#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo " hyperbench-llm: One-file Demo Runner  "
echo "========================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] GPU detected:"
  nvidia-smi | sed -n '1,12p'
else
  echo "[warn] nvidia-smi not found. GPU may not be available in this environment."
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Load .env if present for HF_TOK and paths
if [[ -f .env ]]; then
  echo "[info] Loading environment variables from .env"
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '\n' -I {} bash -lc 'echo {}' | xargs)
fi

# Avoid optional torchvision import path issues inside transformers
export TRANSFORMERS_NO_TORCHVISION=1

run_with_conda() {
  echo "[info] Using Conda environment from environment.yml"
  eval "$(conda shell.bash hook)"
  # Try to create; if exists, update instead
  conda env create -f environment.yml || conda env update -f environment.yml --prune
  conda activate qlora
  python --version
  if [[ -z "${HF_TOK:-}" ]] && command -v huggingface-cli >/dev/null 2>&1; then
    echo "[info] No HF_TOK set; attempting device login (press ENTER in browser if prompted)."
    huggingface-cli login --token || true
  fi
  if [[ "${1:-}" == "v2" ]]; then
    python scripts/demo_v2.py
  elif [[ "${1:-}" == "v3" ]]; then
    python scripts/demo_v3.py
  elif [[ "${1:-}" == "v4" ]]; then
    python scripts/demo_v4.py
  elif [[ "${1:-}" == "v5" ]]; then
    python scripts/demo_v5_llamacpp.py --help | cat
    echo "\n[hint] Example:" 
    echo "python scripts/demo_v5_llamacpp.py --ckpt meta-llama/Meta-Llama-3-8B-Instruct --adapter outputs/naive_run/lora_adapter --variants Q8_0 Q6_K"
    exit 0
  else
    python scripts/demo.py
  fi
}

run_with_venv() {
  echo "[info] Conda not found. Falling back to Python venv + pip."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel setuptools
  # Assume torch is present in most GPU images; if not, try to install a CUDA build
  if ! python - <<'PY'
import sys
try:
  import torch  # noqa: F401
  print('torch-found')
except Exception:
  sys.exit(1)
PY
  then
    echo "[info] Installing PyTorch (may take a while)."
    # Try CUDA 12.1 builds first; if fails, fall back to default index (CPU)
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio || \
    python -m pip install torch torchvision torchaudio
  fi
  python -m pip install "transformers==4.52.4" "datasets==3.6.0" "peft==0.15.2" \
                         "bitsandbytes==0.46.0" "sentencepiece==0.2.0" \
                         "accelerate==1.7.0" "matplotlib" "huggingface_hub"
  if [[ -z "${HF_TOK:-}" ]] && command -v huggingface-cli >/dev/null 2>&1; then
    echo "[info] No HF_TOK set; attempting device login (press ENTER in browser if prompted)."
    huggingface-cli login --token || true
  fi
  if [[ "${1:-}" == "v2" ]]; then
    python scripts/demo_v2.py
  elif [[ "${1:-}" == "v3" ]]; then
    python scripts/demo_v3.py
  elif [[ "${1:-}" == "v4" ]]; then
    python scripts/demo_v4.py
  elif [[ "${1:-}" == "v5" ]]; then
    python scripts/demo_v5_llamacpp.py --help | cat
    echo "\n[hint] Example:" 
    echo "python scripts/demo_v5_llamacpp.py --ckpt meta-llama/Meta-Llama-3-8B-Instruct --adapter outputs/naive_run/lora_adapter --variants Q8_0 Q6_K"
    exit 0
  else
    python scripts/demo.py
  fi
}

if command -v conda >/dev/null 2>&1; then
  run_with_conda "$@"
else
  run_with_venv "$@"
fi

if [[ "${1:-}" == "v2" ]]; then
  echo "[done] Demo v2 artifacts: outputs/demo_v2 (metrics_v2.json, demo_v2.png)."
elif [[ "${1:-}" == "v3" ]]; then
  echo "[done] Demo v3 artifacts: outputs/demo_v3 (metrics_v3.json, compression_recreated.png, scaling_recreated.png, live_dashboard.png)."
elif [[ "${1:-}" == "v4" ]]; then
  echo "[done] Demo v4 artifacts: outputs/demo_v4 (metrics_v4.json, compression_real.png)."
elif [[ "${1:-}" == "v5" ]]; then
  echo "[info] v5 prints usage and an example command for llama.cpp workflow. Run the script directly with required flags."
else
  echo "[done] Demo artifacts saved to outputs/demo (metrics.json, demo_plot.png)."
fi


