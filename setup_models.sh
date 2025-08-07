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
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo "Downloading Llama models using Hugging Face CLI..."
echo "Note: This may download up to ~140GB of model data, depending on selections"
echo ""

mkdir -p "$MODELS_DIR/models"

huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
  --repo-type model --local-dir "$MODELS_DIR/models/Meta-Llama-3.1-70B-Instruct" \
  --local-dir-use-symlinks False

huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --repo-type model --local-dir "$MODELS_DIR/models/Meta-Llama-3.1-8B-Instruct" \
  --local-dir-use-symlinks False

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