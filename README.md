# LLM-Heuristics

**LLM-based DICOM header analysis and heuristic file generation for heudiconv**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

LLM-Heuristics is a tool that leverages Large Language Models (LLMs) to automatically analyze DICOM headers and generate heuristic files for [heudiconv](https://heudiconv.readthedocs.io/), streamlining the conversion from DICOM to BIDS format.

## How It Works

1. **DICOM Analysis**: Scans DICOM directory and extracts metadata from headers
2. **BIDS Mapping**: Uses BIDS schema to map scanning sequences automatically using LLM, based on the dicom information using LLM
3. **Heuristic Generation**: LLM generates Python heuristics file

**All processing happens locally on your machine.** No DICOM data, headers, or metadata are ever sent to external services or shared with third parties. The LLM models are downloaded once and run entirely on your local hardware (GPU/CPU).

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Apptainer Installation](#apptainer-installation)
- [Workflow](#workflow)
- [Models](#models)
  - [Model Selection Guidelines](#model-selection-guidelines)
  - [Usage Examples](#usage-examples)
  - [Model Download and Caching](#model-download-and-caching)
  - [Performance Tips](#performance-tips)
- [Configuration](#configuration)
- [Model Setup](#model-setup)
  - [Download Models Locally](#download-models-locally)
  - [Getting HuggingFace Access](#getting-huggingface-access)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Support](#support)


## Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/NeuroHackademy2025/llm-heuristics.git
cd llm-heuristics
pip install -e .[all]

# Install HeuDiConv (required dependency)
# already installed in `docker://tientong/llm-heuristics:0.1.0` 
pip install heudiconv[all]

```

### Apptainer Installation

```bash
# Pull the container
apptainer build llm-heuristics_0-1-0.sif docker://tientong/llm-heuristics:0.1.0

```

## Workflow

The workflow now includes separate grouping and mapping steps for better understanding and control:

1. **Analyze your dataset**:

**Run with multiple cpu (cannot be used with --slurm)**
   ```bash
   llm-heuristics analyze /path/to/dicom/data /path/to/output --n-cpus 16
   ```

**Run using slurm jobs** (cannot be used with --n-cpus)
   ```bash
   # Generate SLURM script for cluster processing
   llm-heuristics analyze /large/dicom/dataset /output/dir --slurm

   # Submit the generated job array
   sbatch /cluster/jobs/heudiconv_extract.slurm
   ```

   This creates in `/path/to/output/`:
   - `.heudiconv/` folder with HeuDiConv output files
   - `aggregated_dicominfo.tsv` file with aggregated dicom metadata across all sequences

2. **Group sequences** (groups similar sequences together):
   ```bash
   llm-heuristics group /path/to/output

   # apptainer
   apptainer run --nv \
    -B /path/to/output:/output \
    -B /path/to/models:/home/llmuser/models \
    llm-heuristics_0-1-0.sif group /output
   ```
   This creates:
   - `aggregated_dicominfo_groups.tsv` - Grouped sequences data
   - `grouping_report.txt` - Grouping summary
   
3. **Map to BIDS** (LLM-powered based):
   ```bash
   llm-heuristics map /path/to/output

   # apptainer
   apptainer run --nv \
    -B /path/to/output:/output \
    -B /path/to/models:/home/llmuser/models \
    llm-heuristics_0-1-0.sif map /output
   ```
   This creates:
   - `aggregated_dicominfo_mapped.tsv` - Groups mapped to BIDS
   - `mapping_report.txt` - BIDS mapping summary
   
4. **Generate a heuristic file** (uses mapped data from step 3):
   ```bash
   llm-heuristics generate /path/to/output -o heuristic.py

   # apptainer
   apptainer run --nv \
    -B /path/to/output:/output \
    -B /path/to/models:/home/llmuser/models \
    llm-heuristics_0-1-0.sif generate /output -o /output/heuristic.py
   ```

5. **Use with heudiconv**:
   ```bash
   heudiconv -d /path/to/dicom/{subject}/*/*.dcm \
       -o /path/to/bids \
       -f heuristic.py \
       -s 01 \
       -c dcm2niix \
       -b
   ```

### Generate Heuristic with Custom Context

The `--context` parameter provides custom guidance to the LLM for generating more appropriate heuristics:

```bash
# Example 1: Prefer raw/original data
llm-heuristics generate /output/dir -o heuristic.py \
    --context "Use raw sequences only, exclude any derived or motion-corrected data"

# Example 2: Use derived/processed data when needed
llm-heuristics generate /output/dir -o heuristic.py \
    --context "For functional data, use motion-corrected (derived) sequences instead of raw"
```

## Models

LLM-Heuristics uses Llama 3.1 models for heuristic generation. The `--model` parameter accepts any **HuggingFace model name** that is compatible with the Llama architecture.

You can use any **Llama 3.1 model** from HuggingFace, including:

- `meta-llama/Meta-Llama-3.1-70B`
- `meta-llama/Meta-Llama-3.1-8B`
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`

### **Model Selection Guidelines**

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **Production** | 70B Instruct | Highest accuracy for medical imaging |
| **Development/Testing** | 8B Instruct | Faster iteration, lower resource usage |
| **Limited Hardware** | 8B Instruct | Lower memory and VRAM requirements |

### **Usage Examples**

```bash
# Use 8B model for lower resource usage
llm-heuristics map /path/to/output \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output heuristic.py

llm-heuristics map /path/to/output \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --no-quantization
```

### **Model Download and Caching**

- **First run**: Models are downloaded from HuggingFace and cached locally
- **Subsequent runs**: Uses cached models for faster startup
- **Cache location**: `~/.cache/huggingface/` (configurable via `HF_HOME`)
- **Access**: Requires HuggingFace account with appropriate model access
- **Quantization**: Enabled by default to reduce memory usage

<details>
<summary><strong>What is Quantization?</strong> (Click to expand)</summary>

Quantization is a technique used to reduce the memory footprint and computational requirements of LLMs. It works by reducing the precision of model weights from high-precision formats to lower-precision formats.

#### **Types of Quantization:**

| Precision | Memory Usage | Quality | Speed |
|-----------|--------------|---------|-------|
| **FP32** (32-bit) | 100% | Best | Slowest |
| **FP16** (16-bit) | 50% | Very Good | Fast |
| **INT8** (8-bit) | 25% | Good | Faster |
| **INT4** (4-bit) | 12.5% | Acceptable | Fastest |

**In This Project:**

- **Default**: 4-bit quantization enabled (`use_quantization=True`)
- **Memory Savings**: ~75% reduction in model size
- **Quality Trade-off**: Small quality reduction for significant memory savings
- **Configurable**: Use `--no-quantization` flag for higher precision

</details>

### **Performance Tips**

- **For development**: Use 8B model with quantization for faster iteration
- **For production**: Use 70B model for best accuracy
- **Memory optimization**: Keep quantization enabled unless you have excess GPU memory

## Configuration

Create a configuration file or use environment variables:

```bash
# Environment variables
export LLM_MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct"  # Default model
export LLM_USE_QUANTIZATION="true"
export LOG_LEVEL="INFO"

# Or create config.json
{
  "model": {
    "name": "meta-llama/Meta-Llama-3.1-70B-Instruct",  # Default model
    "use_quantization": true,
    "temperature": 0.7
  },
  "heuristic": {
    "include_session": true,
    "exclude_motion_corrected": true
  }
}
```

## Model Setup

### Download Models Locally

**Quick Setup:**
```bash
# Set your HuggingFace token
export HF_TOKEN="your_huggingface_token"

# Run setup script (downloads both 70B and 8B models)
./setup_models.sh
```

The setup script will:
- Download models to `~/.cache/huggingface/` by default
- Use `$HF_HOME` if set (e.g., `export HF_HOME=~/llm-models`)
- Cache models for future use

### Getting HuggingFace Access

1. **Create HuggingFace account**: https://huggingface.co/join
2. **Request Llama access**: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
3. **Create access token**: https://huggingface.co/settings/tokens
4. **Wait for approval**: Meta will review your request

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

Note: This software uses Large Language Models that are subject to separate license terms. The Llama models require separate licensing from Meta. See the LICENSE file for complete details on third-party components.

## Acknowledgments

- [HeuDiConv](https://heudiconv.readthedocs.io/) team for the DICOM-BIDS conversion tool
- [Meta](https://ai.meta.com/) for the Llama models
- [Hugging Face](https://huggingface.co/) for the Transformers library

## Support

- **Issues**: [GitHub Issues](https://github.com/NeuroHackademy2025/llm-heuristics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NeuroHackademy2025/llm-heuristics/discussions)
