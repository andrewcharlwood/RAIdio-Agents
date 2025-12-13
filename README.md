# Cloud Deployment Guide

Deploy the 3D Medical Image Analysis Pipeline on cloud GPU instances.

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | 16GB VRAM | NVIDIA A100 40GB+ |
| RAM | 32GB | 64GB |
| Disk | 50GB | 200GB (for all models) |
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| Drivers | NVIDIA 525+ | Latest |

**No system CUDA installation required** - CUDA is bundled via conda.

## Quick Start

```bash
# 1. Install Miniconda (if not present)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc

# 2. Clone the repository
git clone <your-repo-url> medical-imaging
cd medical-imaging

# 3. Run setup (downloads M3D-LaMed by default)
chmod +x scripts/setup_cloud.sh
./scripts/setup_cloud.sh --models=m3d-lamed

# 4. Activate and run
source activate.sh
python scripts/analyse_input_dicoms.py --help
```

## Setup Options

```bash
./scripts/setup_cloud.sh [OPTIONS]

Options:
  --models=MODELS    Models to download (comma-separated)
  --skip-models      Setup environment only, skip model downloads
  --with-experts     Download VILA-M3 expert checkpoints (VISTA3D, etc.)
  --test             Run tests after setup
  --env-name=NAME    Custom conda environment name (default: medical-imaging)
  --help             Show help
```

### Model Selection

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `m3d-lamed` | 16GB | ~8GB | General VQA, fast inference |
| `vila-m3-8b` | 18GB | ~16GB | VQA + expert segmentation |
| `vila-m3-13b` | 30GB | ~26GB | Higher quality, slower |
| `med3dvlm` | 33GB | ~16GB | Alternative VQA model |
| `radfm` | 100GB* | ~24GB | Large-scale training |

*RadFM requires ~150GB during extraction

### Examples

```bash
# Minimal setup (M3D-LaMed only, fits 16GB GPU)
./scripts/setup_cloud.sh --models=m3d-lamed

# Recommended setup (fits A100 40GB)
./scripts/setup_cloud.sh --models=m3d-lamed,vila-m3-8b

# Full setup with expert models
./scripts/setup_cloud.sh --models=m3d-lamed,vila-m3-8b --with-experts

# Environment only (for custom model management)
./scripts/setup_cloud.sh --skip-models

# All models (requires 200GB+ disk, 80GB+ VRAM for some)
./scripts/setup_cloud.sh --models=all
```

## Cloud Provider Quick Start

### AWS (EC2)

```bash
# Recommended instance: p4d.24xlarge (A100) or g5.xlarge (A10G, budget)
# AMI: Deep Learning AMI (Ubuntu) - has conda pre-installed

# Connect and run
ssh -i key.pem ubuntu@<instance-ip>
git clone <repo> && cd medical-imaging
./scripts/setup_cloud.sh --models=m3d-lamed
```

### Google Cloud (GCE)

```bash
# Recommended: a2-highgpu-1g (A100) or n1-standard-8 + T4
# Image: Deep Learning VM

gcloud compute ssh <instance-name>
git clone <repo> && cd medical-imaging
./scripts/setup_cloud.sh --models=m3d-lamed
```

### Lambda Labs / RunPod / Vast.ai

```bash
# These typically have conda pre-installed
git clone <repo> && cd medical-imaging
./scripts/setup_cloud.sh --models=m3d-lamed
```

## Environment Management

### Activation

```bash
# Option 1: Use convenience script
source activate.sh

# Option 2: Direct conda activation
conda activate medical-imaging
```

### Update Environment

```bash
# After modifying environment.yml
conda env update -f environment.yml --prune
```

### Create Reproducible Lockfile

```bash
# Install conda-lock
conda install -c conda-forge conda-lock

# Generate lockfile
conda-lock -f environment.yml -p linux-64

# Deploy from lockfile (exact versions)
conda-lock install -n medical-imaging conda-lock.yml
```

## Running Analysis

### Basic Usage

```bash
# Activate environment
source activate.sh

# Place DICOM files
cp -r /path/to/dicoms Input/DICOM/

# List available series
python scripts/analyse_input_dicoms.py --list

# Analyse all series
python scripts/analyse_input_dicoms.py --all

# Analyse specific series
python scripts/analyse_input_dicoms.py --series "SER0001"
```

### Using Convenience Scripts

```bash
# Single model analysis
./run_analysis.sh m3d-lamed

# Multi-model comparison
./run_comparison.sh m3d-lamed,vila-m3-8b
```

### With OpenRouter Summarisation

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key

# Analyse and summarise
python scripts/analyse_input_dicoms.py --all --summarise

# Summarise existing reports only
python scripts/analyse_input_dicoms.py --summarise-only
```

## Directory Structure

```
medical-imaging/
├── Input/DICOM/           # Place DICOM folders here
├── Output/                # Generated reports
│   └── YYYYMMDD/
│       └── CT/
│           ├── series_name.md
│           └── _SUMMARY.md
├── models/                # Downloaded model weights
├── external/              # VILA-M3 framework (cloned during setup)
├── src/                   # Source code
├── scripts/               # CLI scripts
├── environment.yml        # Conda environment definition
└── activate.sh            # Environment activation script
```

## Troubleshooting

### "Conda not found"

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc
```

### "CUDA out of memory"

- Use a smaller model (`m3d-lamed` needs only ~8GB)
- Close other GPU processes: `nvidia-smi` to check, `kill <PID>` to stop
- Reduce batch size in analysis script

### "No GPU detected"

```bash
# Check NVIDIA drivers
nvidia-smi

# If not installed (Ubuntu)
sudo apt-get update
sudo apt-get install nvidia-driver-535
sudo reboot
```

### Slow model downloads

```bash
# Use HuggingFace CLI with resume support
pip install huggingface_hub
huggingface-cli download BAAI/M3D-LaMed-Phi-3-4B --local-dir models/M3D-LaMed-Phi-3-4B
```

### Environment conflicts

```bash
# Remove and recreate environment
conda deactivate
conda env remove -n medical-imaging
./scripts/setup_cloud.sh --skip-models
```

## Updating

```bash
# Pull latest code
git pull

# Update conda environment
conda env update -f environment.yml --prune

# Re-download models if needed
python scripts/download_model.py --model m3d-lamed --model-dir models/
```

## Security Notes

- **Never commit patient data** - DICOM files are excluded via `.gitignore`
- **API keys** - Use environment variables, never hardcode
- **Model outputs** - This is a research tool; all results require radiologist verification

## Support

- Issues: [GitHub Issues](<your-repo>/issues)
- Model documentation: [M3D-LaMed Paper](https://arxiv.org/abs/2404.00578)
