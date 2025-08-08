FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

# avoid interactive prompts, ensure python output is unbuffered
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    PATH="/home/llmuser/.local/bin:$PATH" \
    HF_HOME=/home/llmuser/models \
    TRANSFORMERS_CACHE=/home/llmuser/models \
    LLM_MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct" \
    LLM_USE_QUANTIZATION="true" \
    LOG_LEVEL="INFO"

# Install OS deps, create user, dirs, and clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      python3-venv \
      python3-pip \
      ca-certificates \
      dcm2niix \
      pigz \
      wget \
      curl \
      git && \
    useradd -m -s /bin/bash llmuser && \
    mkdir -p /home/llmuser/{app,data,output,models} && \
    chown -R llmuser:llmuser /home/llmuser && \
    rm -rf /var/lib/apt/lists/*

USER llmuser
WORKDIR /home/llmuser/app

# Copy only necessary files (avoids .git and other excluded files)
COPY --chown=llmuser:llmuser pyproject.toml README.md /home/llmuser/app/
COPY --chown=llmuser:llmuser llm_heuristics/ /home/llmuser/app/llm_heuristics/

# Ensure no .git artifacts are present (cleanup just in case)
RUN find /home/llmuser/app -name ".git*" -type f -delete 2>/dev/null || true && \
    find /home/llmuser/app -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# Set version for setuptools-scm (since .git folder isn't available in Docker)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    PYTHONPATH="/home/llmuser/.local/lib/python3.12/site-packages:/home/llmuser/app"

# Upgrade pip and install dependencies from pyproject.toml
RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages && \
    pip install --no-cache-dir --break-system-packages \
      torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
      --trusted-host download.pytorch.org && \
    pip install --no-cache-dir --break-system-packages -e '.[test,dev]' \
      --trusted-host pypi.org --trusted-host files.pythonhosted.org && \
    pip list | grep llm-heuristics

# Shell aliases
RUN echo 'alias ll="ls -la"' >> /home/llmuser/.bashrc && \
    echo 'alias la="ls -la"' >> /home/llmuser/.bashrc

EXPOSE 8000
ENTRYPOINT ["llm-heuristics"]
