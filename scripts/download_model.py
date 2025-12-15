"""
Download Model Weights

Downloads model weights for supported 3D medical image models from Hugging Face.
Supports: M3D-LaMed, Med3DVLM, RadFM, CT-CLIP
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download


# Default paths
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"

# Model configurations
MODEL_CONFIGS = {
    "m3d-lamed": {
        "repo_id": "Areeb-02/M3D-LaMed-Phi-3-4B-finetuned",
        "local_dir": "M3D-LaMed-Phi-3-4B",
        "ignore_patterns": ["model.bin"],  # Skip legacy format, use SafeTensors
        "description": "M3D-LaMed-Phi-3-4B (VQA, ~16GB)",
        "extra_files": {
            "vit": {
                "repo_id": "GoodBaiBai88/M3D-CLIP",
                "filename": "pretrained_ViT.bin",
                "description": "Vision encoder weights",
            }
        }
    },
    "med3dvlm": {
        "repo_id": "MagicXin/Med3DVLM-Qwen-2.5-7B",
        "local_dir": "Med3DVLM",
        "ignore_patterns": [],
        "description": "Med3DVLM-Qwen-2.5-7B (VQA, ~33GB)",
    },
    "radfm": {
        "repo_id": "chaoyi-wu/RadFM",
        "local_dir": "RadFM",
        "ignore_patterns": ["RadFM.zip", "RadFM.z0*"],  # Skip split archive, use single zip only
        "description": "RadFM (VQA, LLaMA base, ~100GB as zip archives)",
        "archives": [
            {"name": "pytorch_model.zip", "is_split": False},  # Single archive
        ],
        "extract_required": True,
        "language_files_source": "external/RadFM/Quick_demo/Language_files",
        "note_language_files": "Language_files (LLaMA tokenizer) copied from external repo, not HuggingFace",
    },
    "ct-clip": {
        "repo_id": "ibrahimethemhamamci/CT-CLIP",  # May need to be updated
        "local_dir": "CT-CLIP",
        "ignore_patterns": [],
        "description": "CT-CLIP (Classifier, 18 pathologies)",
        "note": "Check https://github.com/ibrahimethemhamamci/CT-CLIP for model availability"
    },
    "vila-m3-3b": {
        "repo_id": "MONAI/Llama3-VILA-M3-3B",
        "local_dir": "VILA-M3-3B",
        "ignore_patterns": [],
        "description": "VILA-M3-3B (VQA + Experts, MONAI, ~6GB)",
        "conv_mode": "vicuna_v1",
        "note": "Requires external/vila-m3 framework. Use conv_mode=vicuna_v1",
    },
    "vila-m3-8b": {
        "repo_id": "MONAI/Llama3-VILA-M3-8B",
        "local_dir": "VILA-M3-8B",
        "ignore_patterns": [],
        "description": "VILA-M3-8B (VQA + Experts, MONAI, ~18GB VRAM)",
        "conv_mode": "llama_3",
        "note": "Requires external/vila-m3 framework. Use conv_mode=llama_3",
    },
    "vila-m3-13b": {
        "repo_id": "MONAI/Llama3-VILA-M3-13B",
        "local_dir": "VILA-M3-13B",
        "ignore_patterns": [],
        "description": "VILA-M3-13B (VQA + Experts, MONAI, ~30GB VRAM)",
        "conv_mode": "vicuna_v1",
        "note": "Requires external/vila-m3 framework. Use conv_mode=vicuna_v1",
    },
}


def extract_radfm_archives(target_dir: Path, cleanup: bool = True) -> bool:
    """
    Extract RadFM zip archive on Linux/macOS.

    RadFM model weights are downloaded as pytorch_model.zip (single archive).

    NOTE: ZIP extraction is inherently single-threaded regardless of tool used.
    This is a limitation of the ZIP format - data must be read sequentially.
    The -mmt flag in 7z only helps with compression, not extraction.
    Split archives (RadFM.z01-z04 + RadFM.zip) are also sequential, not parallel.

    Args:
        target_dir: Directory containing the downloaded archive
        cleanup: Remove zip file after successful extraction

    Returns:
        True if extraction successful
    """
    system = platform.system().lower()

    if system == "windows":
        print("  Automatic extraction not supported on Windows.")
        print("  Please extract manually using 7-Zip or similar:")
        print(f"    Extract pytorch_model.zip")
        return False

    # Check for extraction tools
    use_7z = shutil.which("7z") is not None
    has_unzip = shutil.which("unzip") is not None

    if use_7z:
        print("  Using 7z for extraction...")
        print("  Note: ZIP extraction is single-threaded (format limitation)")
    elif has_unzip:
        print("  Using unzip for extraction...")
    else:
        print("  Error: No extraction tool found. Install with:")
        print("    Ubuntu/Debian: sudo apt-get install p7zip-full")
        print("                   sudo apt-get install unzip")
        print("    macOS: brew install p7zip")
        return False

    success = True
    archives = [
        ("pytorch_model.zip", False),  # Single archive
    ]

    for archive_name, is_split in archives:
        archive_path = target_dir / archive_name

        if not archive_path.exists():
            print(f"  Warning: {archive_name} not found, skipping...")
            continue

        print(f"  Extracting {archive_name}...")

        try:
            if use_7z:
                # 7z extraction (single-threaded for ZIP format)
                # -o specifies output directory (no space after -o)
                # -y answers yes to all prompts (overwrite)
                # Note: -mmt=on removed - it only helps compression, not extraction
                result = subprocess.run(
                    ["7z", "x", "-y", f"-o{target_dir}", str(archive_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(target_dir),
                )
            else:
                # Fall back to unzip
                result = subprocess.run(
                    ["unzip", "-o", "-q", str(archive_path), "-d", str(target_dir)],
                    capture_output=True,
                    text=True,
                    cwd=str(target_dir),
                )

            if result.returncode != 0:
                print(f"    Error extracting {archive_name}:")
                print(f"    {result.stderr}")
                success = False
                continue

            print(f"    Extracted {archive_name} successfully")

            # Cleanup zip files if requested
            if cleanup:
                archive_path.unlink()
                if is_split:
                    # Remove split parts
                    for part in target_dir.glob(f"{archive_name.replace('.zip', '')}.z*"):
                        part.unlink()
                print(f"    Cleaned up archive files")

        except Exception as e:
            print(f"    Error: {e}")
            success = False

    return success


def download_radfm_language_files(target_dir: Path) -> bool:
    """
    Download RadFM Language_files from HuggingFace.

    RadFM requires LLaMA tokenizer files. This function downloads them
    from the RadFM HuggingFace repo, or falls back to base LLaMA tokenizer.

    Args:
        target_dir: RadFM model directory (e.g., models/RadFM)

    Returns:
        True if download successful
    """
    target_lang_dir = target_dir / "Language_files"
    target_lang_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]

    print(f"  Downloading Language_files for RadFM tokenizer...")

    # Method 1: Try to download from RadFM HuggingFace repo
    try:
        print(f"    Attempting download from chaoyi-wu/RadFM...")
        downloaded = 0
        for filename in required_files:
            try:
                hf_hub_download(
                    repo_id="chaoyi-wu/RadFM",
                    filename=f"Language_files/{filename}",
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                )
                print(f"    Downloaded {filename}")
                downloaded += 1
            except Exception:
                pass  # File might not exist in repo

        if downloaded == len(required_files):
            print(f"  Language_files downloaded successfully from RadFM repo")
            return True
        elif downloaded > 0:
            print(f"    Partial download ({downloaded}/{len(required_files)} files)")
    except Exception as e:
        print(f"    RadFM repo download failed: {e}")

    # Method 2: Try base LLaMA-7B tokenizer (smaller, same tokenizer format)
    try:
        print(f"    Falling back to base LLaMA tokenizer...")
        snapshot_download(
            repo_id="huggyllama/llama-7b",
            local_dir=str(target_lang_dir),
            allow_patterns=["tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"],
            local_dir_use_symlinks=False,
        )

        # Create minimal config.json if not present
        config_path = target_lang_dir / "config.json"
        if not config_path.exists():
            import json
            config = {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "vocab_size": 32000,
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"    Created minimal config.json")

        # Verify files exist
        existing = [f for f in required_files if (target_lang_dir / f).exists()]
        if len(existing) >= 3:  # tokenizer.model is essential
            print(f"  Language_files downloaded from LLaMA base ({len(existing)}/{len(required_files)} files)")
            return True

    except Exception as e:
        print(f"    LLaMA tokenizer download failed: {e}")

    # Method 3: Copy from external repo if available (original method)
    source_dir = Path(__file__).parent.parent / "external" / "RadFM" / "Quick_demo" / "Language_files"
    if source_dir.exists():
        print(f"    Copying from external repo...")
        try:
            copied = 0
            for filename in required_files:
                source_file = source_dir / filename
                if source_file.exists():
                    shutil.copy2(source_file, target_lang_dir / filename)
                    copied += 1
            if copied > 0:
                print(f"  Language_files copied from external repo ({copied}/{len(required_files)} files)")
                return copied == len(required_files)
        except Exception as e:
            print(f"    Copy from external failed: {e}")

    print(f"  ERROR: Failed to obtain Language_files from any source")
    print(f"  Manual fix: Download LLaMA tokenizer files to {target_lang_dir}")
    return False


def download_model(
    model_name: str,
    model_dir: Path,
    resume: bool = True,
    auto_extract: bool = True,
    cleanup_archives: bool = True,
) -> bool:
    """
    Download a specific model.

    Args:
        model_name: Name of the model to download
        model_dir: Directory to save model
        resume: Enable resume download
        auto_extract: Automatically extract archives (RadFM)
        cleanup_archives: Remove archive files after extraction

    Returns:
        True if successful
    """
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return False

    config = MODEL_CONFIGS[model_name]
    target_dir = model_dir / config["local_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name}...")
    print(f"  Repository: {config['repo_id']}")
    print(f"  Target: {target_dir}")
    if config.get("note"):
        print(f"  Note: {config['note']}")
    print()

    try:
        snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(target_dir),
            ignore_patterns=config.get("ignore_patterns", []),
        )

        # Verify download
        config_path = target_dir / "config.json"
        if config_path.exists():
            print(f"  Download complete! config.json found")
        else:
            # Some models may not have config.json
            if list(target_dir.iterdir()):
                print(f"  Download complete! Files present in {target_dir}")
            else:
                print("  Warning: Download completed but directory is empty!")
                return False

        # Download extra files if any
        if "extra_files" in config:
            for name, extra in config["extra_files"].items():
                print(f"\n  Downloading extra: {extra['description']}...")
                hf_hub_download(
                    repo_id=extra["repo_id"],
                    filename=extra["filename"],
                    local_dir=str(model_dir),
                )
                print(f"    Downloaded {extra['filename']}")

        # Copy Language_files for RadFM (before extraction)
        if model_name == "radfm" and config.get("language_files_source"):
            print()
            copy_success = download_radfm_language_files(target_dir)
            if not copy_success:
                print("  Warning: Language_files copy failed. RadFM may not work correctly.")

        # Handle extraction for RadFM
        if config.get("extract_required"):
            print()
            if auto_extract:
                print("  Attempting automatic extraction...")
                if model_name == "radfm":
                    extract_success = extract_radfm_archives(target_dir, cleanup=cleanup_archives)
                    if not extract_success:
                        print()
                        print("  " + "=" * 50)
                        print("  MANUAL EXTRACTION REQUIRED")
                        print("  " + "=" * 50)
                        print(f"  Navigate to: {target_dir}")
                        print("  Run: 7z x pytorch_model.zip")
                        print("   Or: unzip pytorch_model.zip")
                        print("  Note: ZIP extraction is single-threaded (format limitation)")
                        print()
            else:
                print("  " + "=" * 50)
                print("  EXTRACTION REQUIRED (--no-extract specified)")
                print("  " + "=" * 50)
                print(f"  Navigate to: {target_dir}")
                print("  Run: 7z x pytorch_model.zip")
                print("   Or: unzip pytorch_model.zip")
                print("  Note: ZIP extraction is single-threaded (format limitation)")
                print()

        return True

    except Exception as e:
        print(f"  Error downloading: {e}")
        if "401" in str(e) or "404" in str(e):
            print(f"\n  The model may not be publicly available on Hugging Face.")
            print(f"  Check the model's repository for download instructions:")
            if config.get("note"):
                print(f"    {config['note']}")
        return False


def verify_model(model_name: str, model_dir: Path) -> dict:
    """
    Verify a model download.

    Args:
        model_name: Name of the model
        model_dir: Model directory

    Returns:
        Dictionary with verification results
    """
    if model_name not in MODEL_CONFIGS:
        return {"exists": False, "error": f"Unknown model: {model_name}"}

    config = MODEL_CONFIGS[model_name]
    target_dir = model_dir / config["local_dir"]

    results = {
        "exists": target_dir.exists(),
        "has_config": (target_dir / "config.json").exists() if target_dir.exists() else False,
        "size_gb": 0,
        "file_count": 0,
    }

    if target_dir.exists():
        files = list(target_dir.rglob("*"))
        results["file_count"] = len([f for f in files if f.is_file()])
        results["size_gb"] = sum(
            f.stat().st_size for f in files if f.is_file()
        ) / (1024**3)

    return results


def list_models():
    """Print information about available models."""
    print("Available Models:")
    print("-" * 60)
    for name, config in MODEL_CONFIGS.items():
        print(f"\n  {name}:")
        print(f"    {config['description']}")
        print(f"    HuggingFace: {config['repo_id']}")
        if config.get("note"):
            print(f"    Note: {config['note']}")
        if config.get("note_language_files"):
            print(f"    Note: {config['note_language_files']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights for 3D medical image analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="m3d-lamed",
        help=f"Model to download (default: m3d-lamed). Options: {', '.join(MODEL_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Directory to save models (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing downloads"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable download resume (start fresh)"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip automatic archive extraction (RadFM)"
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep archive files after extraction (default: delete)"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract archives (skip download, for RadFM)"
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3D Medical Model Downloader")
    print("=" * 60)
    print()

    # List mode
    if args.list:
        list_models()
        return 0

    # Verify mode
    if args.verify:
        print("Verifying model downloads...")
        print()
        for name in MODEL_CONFIGS:
            results = verify_model(name, model_dir)
            status = "OK" if results["exists"] and (results["has_config"] or results["file_count"] > 0) else "MISSING"
            print(f"  {name}: {status}")
            if results["exists"]:
                print(f"    Files: {results['file_count']}, Size: {results['size_gb']:.2f} GB")
        return 0

    # Extract-only mode (for RadFM when archives already downloaded)
    if args.extract_only:
        model_name = args.model.lower()
        if model_name != "radfm":
            print(f"Error: --extract-only only applies to radfm model")
            return 1

        config = MODEL_CONFIGS[model_name]
        target_dir = model_dir / config["local_dir"]

        if not target_dir.exists():
            print(f"Error: {target_dir} does not exist. Run download first.")
            return 1

        print(f"Extracting RadFM archives in {target_dir}...")
        cleanup = not args.keep_archives
        if extract_radfm_archives(target_dir, cleanup=cleanup):
            print("Extraction complete!")
            return 0
        else:
            print("Extraction failed or requires manual intervention.")
            return 1

    # Determine models to download
    if args.all:
        models_to_download = list(MODEL_CONFIGS.keys())
    else:
        models_to_download = [args.model.lower()]

    # Validate model names
    for name in models_to_download:
        if name not in MODEL_CONFIGS:
            print(f"Error: Unknown model '{name}'")
            list_models()
            return 1

    # Download models
    resume = not args.no_resume
    auto_extract = not args.no_extract
    cleanup_archives = not args.keep_archives
    success_count = 0
    fail_count = 0

    for model_name in models_to_download:
        print("-" * 60)
        if download_model(model_name, model_dir, resume, auto_extract, cleanup_archives):
            success_count += 1
        else:
            fail_count += 1
        print()

    # Summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print()

    if fail_count == 0:
        print("All downloads completed successfully!")
        return 0
    else:
        print("Some downloads failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
