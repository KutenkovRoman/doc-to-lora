#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="D2L"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.4"

log() {
  printf '\n[install.sh] %s\n' "$1"
}

if [ ! -f pyproject.toml ]; then
  echo "[install.sh] error: run this script from repository root (pyproject.toml not found)." >&2
  exit 1
fi

if ! command -v mlspace >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[install.sh] error: mlspace command is not available in this shell.
Open a Cloud.ru MLSpace Jupyter terminal and run this script there.
MSG
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[install.sh] error: conda is required to use the created MLSpace environment." >&2
  exit 1
fi

log "Ensuring MLSpace environment '$ENV_NAME' exists (python=$PYTHON_VERSION, cuda=$CUDA_VERSION)"
CREATE_LOG=.mlspace_create.log
rm -f "$CREATE_LOG"
if timeout 300 mlspace environments create --env "$ENV_NAME" --python "$PYTHON_VERSION" --cuda "$CUDA_VERSION" >"$CREATE_LOG" 2>&1; then
  log "Environment '$ENV_NAME' created"
else
  if grep -Eqi 'already exists|exist|уже существует' "$CREATE_LOG"; then
    log "Environment '$ENV_NAME' already exists; reusing it"
  else
    echo "[install.sh] error: failed to create environment '$ENV_NAME'" >&2
    sed -n '1,200p' "$CREATE_LOG" >&2 || true
    exit 1
  fi
fi

log "Installing Python packages in environment '$ENV_NAME'"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install uv
conda run -n "$ENV_NAME" python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n "$ENV_NAME" python -m pip install -e .
conda run -n "$ENV_NAME" python -m pip install tokenizers==0.21.0
conda run -n "$ENV_NAME" python -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
conda run -n "$ENV_NAME" python -m pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6

log "Downloading SQuAD dataset"
conda run -n "$ENV_NAME" env HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset rajpurkar/squad --local-dir data/raw_datasets/squad

log "Building compact datasets"
conda run -n "$ENV_NAME" python data/build_drop_compact.py
conda run -n "$ENV_NAME" python data/build_pwc_compact.py
conda run -n "$ENV_NAME" python data/build_ropes_compact.py
conda run -n "$ENV_NAME" python data/build_squad_compact.py

cat <<MSG

[install.sh] Done.
Environment: $ENV_NAME

Optional steps:
  conda run -n $ENV_NAME huggingface-cli login
  conda run -n $ENV_NAME wandb login
MSG
