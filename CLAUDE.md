# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D Medical Image Analysis Pipeline for CT and MRI scans using multi-model VLM architecture. Supports M3D-LaMed, VILA-M3, Med3DVLM, RadFM, and CT-CLIP models for visual question answering and pathology detection.

## Common Commands

### Environment Setup
```bash
# Full setup with model downloads (requires NVIDIA GPU)
./scripts/setup_cloud.sh --models=m3d-lamed,vila-m3-8b

# Environment only (skip model downloads)
./scripts/setup_cloud.sh --skip-models

# Activate environment
source activate.sh
# OR
conda activate medical-imaging
```

### Running Analysis
```bash
# List available DICOM series
python scripts/analyse_input_dicoms.py --list

# Analyse all series with default model (m3d-lamed)
python scripts/analyse_input_dicoms.py --all

# Single model analysis
./run_analysis.sh m3d-lamed

# Multi-model comparison
./run_comparison.sh m3d-lamed,vila-m3-8b

# Analysis with summarisation (requires OPENROUTER_API_KEY)
python scripts/analyse_input_dicoms.py --all --summarise

# Available analysis modes: comprehensive, chained, quick, pathology, simple
python scripts/analyse_input_dicoms.py --all --mode comprehensive --region head
```

### Model Management
```bash
# List available models
python scripts/analyse_input_dicoms.py --list-models

# Download specific model
python scripts/download_model.py --model m3d-lamed --model-dir models/

# Verify downloads
python scripts/download_model.py --verify --model-dir models/
```

## Architecture

### Core Components

- **`src/preprocessing.py`**: DICOM preprocessing pipeline (`DICOMPreprocessor`). Handles loading DICOM series, resampling to isotropic spacing, resizing to model input dimensions (32x256x256), and modality-specific normalization (CT windowing vs MRI percentile normalization). Automatically handles mixed-dimension series by grouping and processing separately.

- **`src/inference.py`**: Model inference wrapper. `M3DLaMedInference` provides backwards-compatible API wrapping the new multi-model system. Use `get_model(name)` or `load_model(name)` for the unified model interface.

- **`src/prompts.py`**: Prompt templates based on M3D-LaMed paper. Contains structured prompts for plane identification (98.8% accuracy), CT phase detection (79.8%), organ identification (74.8%), abnormality detection (66.7%), and location identification (58.9%).

- **`src/openrouter.py`**: OpenRouter API integration for post-processing report summarization via external LLMs (default: deepseek/deepseek-v3.2).

- **`src/utils.py`**: DICOM metadata extraction, body region detection, report formatting (markdown), and segmentation mask utilities (NIfTI output).

### Model System

Models are accessed via `src.models`:
```python
from src.models import get_model, list_models, get_model_info

model = get_model("m3d-lamed")  # or "vila-m3-8b", "med3dvlm", "radfm", "ct-clip"
model.load_model()
response = model.generate_response(image, question, modality)
model.unload_model()  # Free GPU memory before loading next model
```

Available models:
- **m3d-lamed**: Default VQA model (~8GB VRAM), input shape (32, 256, 256)
- **vila-m3-8b/13b**: VQA with expert segmentation support (~16-26GB VRAM)
- **med3dvlm**: Alternative VQA model (~16GB VRAM)
- **radfm**: Large-scale VQA (~24GB VRAM, 100GB download)
- **ct-clip**: Classifier model for pathology detection

### Data Flow

1. DICOM files placed in `Input/DICOM/`
2. `analyse_input_dicoms.py` discovers series, filters duplicates (keeps thinnest slices)
3. `preprocessing.py` converts to normalized tensors (1, D, H, W)
4. Model inference generates analysis results
5. Reports saved to `Output/{date}/{modality}/` as markdown
6. Optional: OpenRouter summarization creates `_SUMMARY.md`

## Key Technical Details

- **Input tensor shape**: (1, 32, 256, 256) - single channel, 32 depth slices, 256x256 spatial
- **CT preprocessing**: Soft tissue window (WW=350, WL=40), normalized to [0,1]
- **MRI preprocessing**: 1-99 percentile clipping, z-score normalization, scaled to [0,1]
- **Mixed dimension handling**: Series with mixed slice dimensions (e.g., scout images) are automatically split and processed as separate volumes
- **Conda environment**: Uses bundled CUDA via `pytorch-cuda=12.1`, no system CUDA required

## Directory Structure

```
Input/DICOM/         # Place DICOM folders here
Output/              # Generated markdown reports
models/              # Downloaded model weights
external/vila-m3/    # VILA-M3 framework (cloned during setup)
src/                 # Core library code
scripts/             # CLI tools
```
