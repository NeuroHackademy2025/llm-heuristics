#!/bin/bash
# Setup script for downloading LLM models

set -e

echo "LLM-Heuristics Model Setup"
echo "=========================="

# Check if HF_TOKEN is provided
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is required"
    echo ""
    echo "Steps to get your HuggingFace token:"
    echo "1. Create account: https://huggingface.co/join"
    echo "2. Request Llama access: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct"
    echo "3. Create token: https://huggingface.co/settings/tokens"
    echo "4. Wait for Meta's approval (usually takes a few hours)"
    echo ""
    echo "Then run:"
    echo "  export HF_TOKEN='your_token_here'"
    echo "  ./setup_models.sh"
    echo ""
    exit 1
fi

echo " HuggingFace token found"

# Create models directory if it doesn't exist
MODELS_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
echo "Models will be downloaded to: $MODELS_DIR"

# Download models
echo ""
echo "Logging into Hugging Face CLI..."

# Prefer new 'hf' CLI if available; fallback to 'huggingface-cli'
if command -v hf >/dev/null 2>&1; then
  HF_LOGIN_CMD=(hf auth login --token "$HF_TOKEN")
  HF_DOWNLOAD_CMD=hf
else
  HF_LOGIN_CMD=(huggingface-cli login --token "$HF_TOKEN")
  HF_DOWNLOAD_CMD=huggingface-cli
fi

"${HF_LOGIN_CMD[@]}"

# Improve reliability on HPC / slow networks
export HF_HUB_ENABLE_HF_TRANSFER=1   # Use Rust downloader if available
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT=60        # Increase HEAD request timeout from default 10s

echo "Downloading Llama models using Hugging Face CLI..."
echo "Note: This may download up to ~140GB of model data, depending on selections"
echo ""

mkdir -p "$MODELS_DIR/models"

download_with_retry() {
  local repo_id="$1"
  local target_dir="$2"
  local attempts=3
  local delay=10

  for i in $(seq 1 $attempts); do
    echo "Downloading $repo_id (attempt $i/$attempts)..."
    if [ "$HF_DOWNLOAD_CMD" = "hf" ]; then
      # Newer 'hf' CLI has different argument names
      if hf download "$repo_id" \
          --local-dir "$target_dir"; then
        echo "Downloaded $repo_id to $target_dir"
        return 0
      fi
    else
      # Legacy 'huggingface-cli' arguments
      if huggingface-cli download "$repo_id" \
          --repo-type model \
          --local-dir "$target_dir" \
          --local-dir-use-symlinks False \
          --max-workers 4; then
        echo "Downloaded $repo_id to $target_dir"
        return 0
      fi
    fi
    echo "Download failed for $repo_id. Retrying in ${delay}s..."
    sleep "$delay"
    delay=$((delay * 2))
  done

  echo "Failed to download $repo_id after $attempts attempts." >&2
  return 1
}

download_with_retry meta-llama/Meta-Llama-3.1-70B-Instruct \
  "$MODELS_DIR/models/Meta-Llama-3.1-70B-Instruct"

download_with_retry meta-llama/Meta-Llama-3.1-8B-Instruct \
  "$MODELS_DIR/models/Meta-Llama-3.1-8B-Instruct"

echo ""
echo " Models downloaded successfully!"
echo ""
echo "Usage with Apptainer:"
echo "  apptainer run --nv \\"
echo "    -B /path/to/dicom:/data \\"
echo "    -B /path/to/output:/output \\"
echo "    -B $MODELS_DIR:/home/llmuser/models \\"
echo "    llm-heuristics_0.1.0.sif generate /data"
echo ""
echo "Usage with Docker:"
echo "  docker run --gpus all \\"
echo "    -v /path/to/dicom:/data \\"
echo "    -v /path/to/output:/output \\"
echo "    -v $MODELS_DIR:/home/llmuser/models \\"
echo "    llm-heuristics:0.1.0 generate /data"