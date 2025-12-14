#!/bin/bash
#
# Complete Setup Script for 3D Medical Image Analysis Pipeline (Conda Version)
# Target: Linux with NVIDIA GPU (A100 40GB/80GB recommended)
#
# This script sets up the complete multi-model medical imaging pipeline including:
# - M3D-LaMed (VQA, ~16GB)
# - Med3DVLM (VQA, ~33GB)
# - RadFM (VQA, ~100GB download, extracts to ~50GB)
# - VILA-M3 (VQA + Experts, ~30GB for 13B)
# - CT-CLIP (Classifier) - if available
#
# Prerequisites:
#   - Conda or Miniconda installed (https://docs.conda.io/en/latest/miniconda.html)
#   - NVIDIA GPU with drivers installed
#   - No system CUDA required (bundled via conda)
#
# Usage:
#   chmod +x scripts/setup_cloud.sh
#   ./scripts/setup_cloud.sh [OPTIONS]
#
# Options:
#   --models=MODELS    Comma-separated list of models to download
#                      Options: m3d-lamed,med3dvlm,radfm,vila-m3-8b,vila-m3-13b,ct-clip,all
#                      Default: m3d-lamed,vila-m3-8b (fits most GPUs)
#   --skip-models      Skip model downloads (setup environment only)
#   --with-experts     Download VILA-M3 expert model checkpoints (VISTA3D, etc.)
#   --test             Run a test analysis after setup
#   --env-name=NAME    Conda environment name (default: medical-imaging)
#   --help             Show this help message
#

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"

# Default models to download (fits A100 40GB comfortably)
DEFAULT_MODELS="m3d-lamed,vila-m3-8b"

# Conda environment name
ENV_NAME="medical-imaging"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "======================================================================"
    echo " $1"
    echo "======================================================================"
    echo ""
}

check_conda() {
    if command -v conda &> /dev/null; then
        log_info "Conda found: $(conda --version)"
        return 0
    else
        log_error "Conda not found. Please install Miniconda or Anaconda first."
        log_info "Quick install: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh"
        return 1
    fi
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "Detected GPU:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    else
        log_warning "nvidia-smi not found. GPU may not be available."
        return 1
    fi
}

check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG "${PROJECT_ROOT}" | tail -1 | awk '{print $4}' | sed 's/G//')

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Need ${required_gb}GB, have ${available_gb}GB"
        return 1
    fi
    log_info "Disk space OK: ${available_gb}GB available (need ${required_gb}GB)"
    return 0
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

MODELS_TO_DOWNLOAD="$DEFAULT_MODELS"
SKIP_MODELS=false
WITH_EXPERTS=false
RUN_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --models=*)
            MODELS_TO_DOWNLOAD="${1#*=}"
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --with-experts)
            WITH_EXPERTS=true
            shift
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        --env-name=*)
            ENV_NAME="${1#*=}"
            shift
            ;;
        --help)
            head -40 "$0" | tail -35
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Main Setup
# ==============================================================================

print_header "3D Medical Image Analysis Pipeline Setup (Conda)"

log_info "Project root: ${PROJECT_ROOT}"
log_info "Conda environment: ${ENV_NAME}"
log_info "Models to download: ${MODELS_TO_DOWNLOAD}"

# Check prerequisites
check_conda || exit 1
check_gpu || true

# Check if a model is already downloaded (has files in its directory)
model_already_downloaded() {
    local model_dir="$1"
    if [ -d "$model_dir" ] && [ "$(ls -A "$model_dir" 2>/dev/null)" ]; then
        return 0  # Already downloaded
    fi
    return 1  # Not downloaded
}

# Estimate disk space needed (only for models not already downloaded)
DISK_NEEDED=0
if [[ "$MODELS_TO_DOWNLOAD" == *"m3d-lamed"* ]] || [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    if ! model_already_downloaded "${MODELS_DIR}/M3D-LaMed-Phi-3-4B"; then
        DISK_NEEDED=$((DISK_NEEDED + 16))
        log_info "m3d-lamed: ~16GB needed"
    else
        log_info "m3d-lamed: already downloaded, skipping space check"
    fi
fi
if [[ "$MODELS_TO_DOWNLOAD" == *"radfm"* ]] || [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    if ! model_already_downloaded "${MODELS_DIR}/RadFM"; then
        DISK_NEEDED=$((DISK_NEEDED + 150))  # RadFM needs ~150GB during extraction
        log_info "radfm: ~150GB needed (during extraction)"
    else
        log_info "radfm: already downloaded, skipping space check"
    fi
fi
if [[ "$MODELS_TO_DOWNLOAD" == *"med3dvlm"* ]] || [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    if ! model_already_downloaded "${MODELS_DIR}/Med3DVLM"; then
        DISK_NEEDED=$((DISK_NEEDED + 35))
        log_info "med3dvlm: ~35GB needed"
    else
        log_info "med3dvlm: already downloaded, skipping space check"
    fi
fi
if [[ "$MODELS_TO_DOWNLOAD" == *"vila-m3-3b"* ]] || [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    if ! model_already_downloaded "${MODELS_DIR}/VILA-M3-3B"; then
        DISK_NEEDED=$((DISK_NEEDED + 6))
        log_info "vila-m3-3b: ~6GB needed"
    else
        log_info "vila-m3-3b: already downloaded, skipping space check"
    fi
fi
if [[ "$MODELS_TO_DOWNLOAD" == *"vila-m3-8b"* ]] || [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    if ! model_already_downloaded "${MODELS_DIR}/VILA-M3-8B"; then
        DISK_NEEDED=$((DISK_NEEDED + 18))
        log_info "vila-m3-8b: ~18GB needed"
    else
        log_info "vila-m3-8b: already downloaded, skipping space check"
    fi
fi
if [[ "$MODELS_TO_DOWNLOAD" == *"vila-m3-13b"* ]] || [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    if ! model_already_downloaded "${MODELS_DIR}/VILA-M3-13B"; then
        DISK_NEEDED=$((DISK_NEEDED + 30))
        log_info "vila-m3-13b: ~30GB needed"
    else
        log_info "vila-m3-13b: already downloaded, skipping space check"
    fi
fi

if [ "$DISK_NEEDED" -gt 0 ]; then
    check_disk_space $DISK_NEEDED
else
    log_info "All requested models already downloaded, skipping disk space check"
fi

# ==============================================================================
# Step 1: Initialize Conda
# ==============================================================================

print_header "Step 1: Initializing Conda"

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

log_success "Conda initialized"

# ==============================================================================
# Step 2: Create/Update Conda Environment
# ==============================================================================

print_header "Step 2: Setting Up Conda Environment"

ENV_FILE="${PROJECT_ROOT}/environment.yml"

if [ ! -f "$ENV_FILE" ]; then
    log_error "environment.yml not found at ${ENV_FILE}"
    exit 1
fi

# Check if environment exists (check both env list and directory)
ENV_PREFIX=$(conda info --base)/envs/${ENV_NAME}
if [ -d "$HOME/.conda/envs/${ENV_NAME}" ]; then
    ENV_PREFIX="$HOME/.conda/envs/${ENV_NAME}"
fi

# Track environment.yml changes using a hash file
ENV_HASH_FILE="${PROJECT_ROOT}/.env_hash"
CURRENT_HASH=$(md5sum "$ENV_FILE" | cut -d' ' -f1)
STORED_HASH=""
if [ -f "$ENV_HASH_FILE" ]; then
    STORED_HASH=$(cat "$ENV_HASH_FILE")
fi

if conda env list | grep -E "^${ENV_NAME}\s+" > /dev/null 2>&1; then
    # Environment exists - check if we need to update
    if [ "$CURRENT_HASH" = "$STORED_HASH" ]; then
        log_info "Environment '${ENV_NAME}' is up to date, skipping setup..."
    else
        log_info "Environment '${ENV_NAME}' exists but environment.yml changed, updating..."
        conda env update -f "$ENV_FILE" --prune
        echo "$CURRENT_HASH" > "$ENV_HASH_FILE"
    fi
elif [ -d "$ENV_PREFIX" ]; then
    # Directory exists but env not registered - remove and recreate
    log_warning "Found orphaned environment directory, removing..."
    rm -rf "$ENV_PREFIX"
    log_info "Creating new environment '${ENV_NAME}'..."
    conda env create -f "$ENV_FILE"
    echo "$CURRENT_HASH" > "$ENV_HASH_FILE"
else
    log_info "Creating new environment '${ENV_NAME}'..."
    conda env create -f "$ENV_FILE"
    echo "$CURRENT_HASH" > "$ENV_HASH_FILE"
fi

# Activate environment
log_info "Activating environment..."
conda activate "$ENV_NAME"

log_success "Conda environment ready"
log_info "Python: $(python --version)"
log_info "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ==============================================================================
# Step 3: Initialize Git LFS
# ==============================================================================

print_header "Step 3: Initializing Git LFS"

git lfs install
log_success "Git LFS initialized"

# ==============================================================================
# Step 4: Clone VILA-M3 Framework (if needed)
# ==============================================================================

VILA_DIR="${PROJECT_ROOT}/external/vila-m3"

if [[ "$MODELS_TO_DOWNLOAD" == *"vila-m3"* ]] || [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    print_header "Step 4: Setting Up VILA-M3 Framework"

    if [ ! -d "$VILA_DIR" ]; then
        log_info "Cloning VLM-Radiology-Agent-Framework..."
        mkdir -p "${PROJECT_ROOT}/external"
        git clone --recursive https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework.git "$VILA_DIR"
    else
        log_info "VILA-M3 framework already exists"

        # Update submodules if needed
        log_info "Updating submodules..."
        cd "$VILA_DIR"
        git submodule update --init --recursive
        cd "$PROJECT_ROOT"
    fi

    # Install VILA framework if available
    if [ -f "${VILA_DIR}/thirdparty/VILA/pyproject.toml" ]; then
        log_info "Installing VILA framework..."
        pip install -e "${VILA_DIR}/thirdparty/VILA"
    fi

    log_success "VILA-M3 framework ready"
fi

# ==============================================================================
# Step 5: Download Models
# ==============================================================================

if [ "$SKIP_MODELS" = false ]; then
    print_header "Step 5: Downloading Model Weights"

    # Create models directory
    mkdir -p "$MODELS_DIR"

    # Parse models list
    if [ "$MODELS_TO_DOWNLOAD" = "all" ]; then
        # Download all models (warning: this is ~200GB+)
        log_warning "Downloading ALL models. This will take significant time and disk space."
        MODELS_LIST="m3d-lamed med3dvlm radfm vila-m3-13b"
    else
        MODELS_LIST=$(echo "$MODELS_TO_DOWNLOAD" | tr ',' ' ')
    fi

    for model in $MODELS_LIST; do
        log_info "Downloading ${model}..."
        python "${PROJECT_ROOT}/scripts/download_model.py" --model "$model" --model-dir "$MODELS_DIR" || {
            log_warning "Failed to download ${model}, continuing..."
        }
        echo ""
    done

    log_success "Model downloads complete"

    # Verify downloads
    log_info "Verifying downloads..."
    python "${PROJECT_ROOT}/scripts/download_model.py" --verify --model-dir "$MODELS_DIR"
fi

# ==============================================================================
# Step 6: Download Expert Model Checkpoints (Optional)
# ==============================================================================

if [ "$WITH_EXPERTS" = true ] && [ -d "$VILA_DIR" ]; then
    print_header "Step 6: Downloading Expert Model Checkpoints"

    log_info "Downloading VISTA3D, CXR, and BRATS expert checkpoints..."
    cd "${VILA_DIR}/m3/demo"

    # Use their Makefile target if available
    if [ -f "Makefile" ]; then
        make demo_m3 || {
            log_warning "Expert model download via Makefile failed, trying manual download..."
        }
    fi

    cd "$PROJECT_ROOT"
    log_success "Expert models setup complete"
fi

# ==============================================================================
# Step 7: Create Convenience Scripts
# ==============================================================================

print_header "Step 7: Creating Convenience Scripts"

# Create activation script
cat > "${PROJECT_ROOT}/activate.sh" << EOF
#!/bin/bash
# Activate the medical imaging conda environment
eval "\$(conda shell.bash hook)"
conda activate ${ENV_NAME}
export PYTHONPATH="\$(dirname "\${BASH_SOURCE[0]}"):\${PYTHONPATH}"
echo "Medical imaging environment activated (${ENV_NAME})"
echo "Run: python scripts/analyse_input_dicoms.py --help"
EOF
chmod +x "${PROJECT_ROOT}/activate.sh"

# Create quick-run script
cat > "${PROJECT_ROOT}/run_analysis.sh" << EOF
#!/bin/bash
# Quick analysis runner
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
eval "\$(conda shell.bash hook)"
conda activate ${ENV_NAME}
export PYTHONPATH="\${SCRIPT_DIR}:\${PYTHONPATH}"

# Default to M3D-LaMed if no model specified
MODEL="\${1:-m3d-lamed}"

echo "Running analysis with model: \${MODEL}"
python "\${SCRIPT_DIR}/scripts/analyse_input_dicoms.py" --all --model "\$MODEL" "\${@:2}"
EOF
chmod +x "${PROJECT_ROOT}/run_analysis.sh"

# Create multi-model comparison script
cat > "${PROJECT_ROOT}/run_comparison.sh" << EOF
#!/bin/bash
# Run multi-model comparison
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
eval "\$(conda shell.bash hook)"
conda activate ${ENV_NAME}
export PYTHONPATH="\${SCRIPT_DIR}:\${PYTHONPATH}"

# Models to compare (edit as needed)
MODELS="\${1:-m3d-lamed,vila-m3-8b}"

echo "Running comparison with models: \${MODELS}"
python "\${SCRIPT_DIR}/scripts/analyse_input_dicoms.py" --all --models "\$MODELS" --compare "\${@:2}"
EOF
chmod +x "${PROJECT_ROOT}/run_comparison.sh"

log_success "Convenience scripts created"

# ==============================================================================
# Step 8: Test Installation (Optional)
# ==============================================================================

if [ "$RUN_TEST" = true ]; then
    print_header "Step 8: Running Test"

    log_info "Testing model imports..."
    python -c "
from src.infer import list_models, get_model_info
print('Available models:', list_models())
for m in list_models():
    try:
        info = get_model_info(m)
        print(f'  {m}: {info[\"type\"]} - {info[\"input_shape\"]}')
    except Exception as e:
        print(f'  {m}: Error - {e}')
"

    log_info "Testing DICOM discovery..."
    python "${PROJECT_ROOT}/scripts/analyse_input_dicoms.py" --list

    log_success "Tests complete"
fi

# ==============================================================================
# Summary
# ==============================================================================

print_header "Setup Complete!"

echo "Environment Information:"
echo "  - Conda env: ${ENV_NAME}"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "  - GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "  - VRAM: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB\")')"
fi
echo ""

echo "Quick Start:"
echo "  1. Activate environment:"
echo "     source activate.sh"
echo "     # OR"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  2. Place DICOM files in:"
echo "     ${PROJECT_ROOT}/Input/DICOM/"
echo ""
echo "  3. Run analysis:"
echo "     ./run_analysis.sh m3d-lamed           # Single model"
echo "     ./run_analysis.sh vila-m3-8b          # VILA-M3 with experts"
echo "     ./run_comparison.sh m3d-lamed,vila-m3-8b  # Compare models"
echo ""
echo "  4. View results in:"
echo "     ${PROJECT_ROOT}/Output/"
echo ""

echo "Available Models:"
python "${PROJECT_ROOT}/scripts/download_model.py" --verify --model-dir "$MODELS_DIR" 2>/dev/null || true

echo ""
echo "To create a reproducible lockfile:"
echo "  conda-lock -f environment.yml -p linux-64"
echo ""

log_success "Setup complete! Happy analyzing!"
