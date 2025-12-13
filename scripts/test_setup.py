"""
Setup Verification Script

Tests all components of the M3D-LaMed installation.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_rocm_pytorch():
    """Test ROCm/PyTorch setup."""
    print("=" * 50)
    print("Test 1: ROCm/PyTorch")
    print("=" * 50)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")

        if not torch.cuda.is_available():
            print("FAIL: torch.cuda.is_available() returned False")
            return False

        print(f"CUDA available: True")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        # Get VRAM info
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {total_mem:.1f} GB")

        # Run small tensor operation
        print("Running GPU tensor test...")
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = torch.matmul(x, y)
        result = z.sum().item()
        print(f"GPU tensor operation successful (checksum: {result:.2f})")

        print("PASS: ROCm/PyTorch working correctly")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_dependencies():
    """Test required dependencies."""
    print()
    print("=" * 50)
    print("Test 2: Dependencies")
    print("=" * 50)

    dependencies = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("huggingface_hub", "huggingface_hub"),
        ("SimpleITK", "SimpleITK"),
        ("pydicom", "pydicom"),
        ("nibabel", "nibabel"),
        ("monai", "monai"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for name, module in dependencies:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {name}: {version}")
        except ImportError as e:
            print(f"  {name}: MISSING ({e})")
            all_ok = False

    if all_ok:
        print("PASS: All dependencies installed")
    else:
        print("FAIL: Some dependencies missing")

    return all_ok


def test_model_files():
    """Test model files are present."""
    print()
    print("=" * 50)
    print("Test 3: Model Files")
    print("=" * 50)

    model_dir = project_root / "models" / "M3D-LaMed-Phi-3-4B"
    vit_path = project_root / "models" / "pretrained_ViT.bin"

    results = {
        "model_dir": model_dir.exists(),
        "config": (model_dir / "config.json").exists() if model_dir.exists() else False,
        "vit": vit_path.exists(),
    }

    print(f"Model directory exists: {results['model_dir']}")
    print(f"config.json present: {results['config']}")
    print(f"pretrained_ViT.bin present: {results['vit']}")

    if model_dir.exists():
        # Calculate model size
        total_size = sum(
            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
        )
        size_gb = total_size / (1024**3)
        print(f"Model size on disk: {size_gb:.2f} GB")

    if results["model_dir"] and results["config"]:
        print("PASS: Model files present")
        return True
    else:
        print("FAIL: Model files missing (run download_model.py)")
        return False


def test_preprocessing():
    """Test preprocessing module."""
    print()
    print("=" * 50)
    print("Test 4: Preprocessing Module")
    print("=" * 50)

    try:
        import numpy as np
        from src.preprocessing import DICOMPreprocessor

        # Instantiate preprocessor
        preprocessor = DICOMPreprocessor(
            target_spacing=(1.0, 1.0, 1.0),
            target_size=(64, 64, 64),  # Small for testing
            modality="CT"
        )
        print("DICOMPreprocessor instantiated: OK")

        # Test CT windowing on dummy volume
        dummy_ct = np.random.uniform(-1000, 1000, (64, 64, 64)).astype(np.float32)
        windowed = preprocessor.apply_ct_windowing(dummy_ct)

        assert windowed.shape == (3, 64, 64, 64), f"Wrong shape: {windowed.shape}"
        assert windowed.dtype == np.float32, f"Wrong dtype: {windowed.dtype}"
        assert windowed.min() >= 0, f"Values below 0: {windowed.min()}"
        assert windowed.max() <= 1, f"Values above 1: {windowed.max()}"

        print(f"CT windowing output shape: {windowed.shape}")
        print(f"CT windowing output range: [{windowed.min():.3f}, {windowed.max():.3f}]")

        # Test MRI normalisation
        preprocessor.modality = "MRI"
        dummy_mri = np.random.uniform(0, 1000, (64, 64, 64)).astype(np.float32)
        normalised = preprocessor.normalise_mri(dummy_mri)

        assert normalised.shape == (1, 64, 64, 64), f"Wrong shape: {normalised.shape}"
        print(f"MRI normalisation output shape: {normalised.shape}")

        print("PASS: Preprocessing module working")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_load(skip: bool = False):
    """Test model loading (slow, optional)."""
    print()
    print("=" * 50)
    print("Test 5: Model Load (Optional)")
    print("=" * 50)

    if skip:
        print("SKIPPED (use --full to enable)")
        return None

    try:
        import torch
        from src.inference import M3DLaMedInference

        print("Initialising inference wrapper...")
        inference = M3DLaMedInference()

        print("Loading model (this may take 1-2 minutes)...")
        inference.load_model()

        # Check memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"GPU memory allocated: {allocated:.2f} GB")
            print(f"GPU memory reserved: {reserved:.2f} GB")

        # Unload
        inference.unload_model()

        print("PASS: Model loads successfully")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test M3D-LaMed setup")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full tests including model loading (slow)"
    )
    args = parser.parse_args()

    print()
    print("M3D-LaMed Setup Verification")
    print("=" * 50)
    print()

    results = {
        "ROCm/PyTorch": test_rocm_pytorch(),
        "Dependencies": test_dependencies(),
        "Model Files": test_model_files(),
        "Preprocessing": test_preprocessing(),
        "Model Load": test_model_load(skip=not args.full),
    }

    # Summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_pass = True
    for name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {name}: {status}")

    print()
    if all_pass:
        print("All tests passed! Setup is complete.")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
