"""
Analyse Input DICOM Studies

Script to analyse all DICOM studies in the Input/DICOM directory.
Supports multiple models: M3D-LaMed, Med3DVLM, RadFM, CT-CLIP
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Default input directory
INPUT_DICOM_DIR = project_root / "Input" / "DICOM"

# Available models
AVAILABLE_MODELS = [
    "m3d-lamed",
    "med3dvlm",
    "radfm",
    "ct-clip",
    "vila-m3",      # Default 8B variant
    "vila-m3-3b",
    "vila-m3-8b",
    "vila-m3-13b",
]
DEFAULT_MODEL = "m3d-lamed"


def find_dicom_series_recursive(root_dir: Path) -> list:
    """
    Find all directories containing DICOM files.

    Returns list of (series_path, file_count) tuples.
    """
    import pydicom

    series_list = []
    checked_dirs = set()

    # Find all directories first, then check one file per directory
    for dirpath in root_dir.rglob("*"):
        if not dirpath.is_dir():
            continue
        if dirpath in checked_dirs:
            continue

        # Get files in this directory (not subdirs)
        files = [f for f in dirpath.iterdir() if f.is_file()]
        if not files:
            continue

        checked_dirs.add(dirpath)

        # Only check first file to validate it's DICOM
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
            if hasattr(ds, 'SOPClassUID') or hasattr(ds, 'Modality'):
                series_list.append((str(dirpath), len(files)))
                print(f"  Found: {dirpath.relative_to(root_dir)} ({len(files)} files)")
        except Exception:
            continue

    return sorted(series_list, key=lambda x: x[0])


def get_series_info(series_path: str) -> dict:
    """Get metadata from a DICOM series."""
    import pydicom

    info = {
        "path": series_path,
        "modality": "Unknown",
        "series_description": "Unknown",
        "slice_count": 0,
        "slice_thickness": None,
        "series_base": None,  # For grouping duplicates
    }

    path = Path(series_path)
    files = [f for f in path.iterdir() if f.is_file()]
    info["slice_count"] = len(files)

    if files:
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
            if hasattr(ds, 'Modality'):
                info["modality"] = ds.Modality
            if hasattr(ds, 'SeriesDescription'):
                info["series_description"] = ds.SeriesDescription
            if hasattr(ds, 'SliceThickness'):
                info["slice_thickness"] = float(ds.SliceThickness)

            # Extract base description for grouping (remove slice thickness number)
            # Pattern: "Head 1.0 Hr40" -> base = "Head Hr40"
            desc = info["series_description"]
            if desc and desc != "Unknown":
                import re
                # Remove floating point numbers (slice thickness) from description
                base = re.sub(r'\s+\d+\.?\d*\s+', ' ', desc).strip()
                # Normalise whitespace
                base = re.sub(r'\s+', ' ', base)
                info["series_base"] = base

        except Exception:
            pass

    return info


def filter_non_scan_series(series_info_list: list) -> list:
    """
    Filter out series that are not actual scans (e.g., Patient Protocol).

    Args:
        series_info_list: List of series info dictionaries

    Returns:
        Filtered list with non-scan series removed
    """
    # Patterns to exclude (case-insensitive)
    exclude_patterns = [
        "patient protocol",
        "protocol",
        "scout",
        "localizer",
        "topogram",
        "surview",
    ]

    filtered = []
    skipped = []

    for info in series_info_list:
        desc = info.get("series_description", "").lower()

        # Check if description matches any exclude pattern
        should_exclude = any(pattern in desc for pattern in exclude_patterns)

        if should_exclude:
            skipped.append(info)
        else:
            filtered.append(info)

    if skipped:
        print(f"\n  Skipping {len(skipped)} non-scan series:")
        for info in skipped:
            print(f"    - '{info['series_description']}'")

    return filtered


def filter_duplicate_series(series_info_list: list) -> list:
    """
    Filter duplicate series to keep only the most detailed version.

    Groups series by their base description (excluding slice thickness) and
    keeps only the one with the smallest slice thickness (most detail).

    Args:
        series_info_list: List of series info dictionaries

    Returns:
        Filtered list with duplicates removed
    """
    from collections import defaultdict

    # Group by (modality, series_base)
    groups = defaultdict(list)
    ungrouped = []

    for info in series_info_list:
        base = info.get("series_base")
        modality = info.get("modality", "Unknown")

        if base and info.get("slice_thickness") is not None:
            key = (modality, base)
            groups[key].append(info)
        else:
            # Can't group - keep it
            ungrouped.append(info)

    # Select best from each group (smallest slice thickness)
    selected = []
    skipped = []

    for key, group in groups.items():
        if len(group) == 1:
            selected.append(group[0])
        else:
            # Sort by slice thickness (smallest first)
            sorted_group = sorted(group, key=lambda x: x["slice_thickness"])
            best = sorted_group[0]
            selected.append(best)

            # Track what we skipped
            for info in sorted_group[1:]:
                skipped.append((info, best))

    # Report skipped duplicates
    if skipped:
        print(f"\n  Filtering {len(skipped)} duplicate series (keeping thinnest slices):")
        for info, kept in skipped:
            print(f"    - Skipping '{info['series_description']}' ({info['slice_thickness']}mm)")
            print(f"      Keeping  '{kept['series_description']}' ({kept['slice_thickness']}mm)")

    return selected + ungrouped


def main():
    parser = argparse.ArgumentParser(
        description="Analyse DICOM studies from Input/DICOM directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available DICOM series without analysing"
    )
    parser.add_argument(
        "--series",
        type=str,
        help="Specific series path to analyse (e.g., S0001/SER0002)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyse all DICOM series found"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["CT", "MRI", "auto"],
        default="auto",
        help="Image modality (default: auto-detect)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["comprehensive", "chained", "quick", "pathology", "simple"],
        default="comprehensive",
        help="Analysis mode: comprehensive (full analysis with pathology screening), "
             "chained (step-by-step query chain), quick (shortened chain), "
             "pathology (pathology screening only), simple (original open-ended analysis)"
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["head", "neck", "chest", "thorax", "abdomen", "pelvis", "auto"],
        default="auto",
        help="Body region for pathology screening (default: auto-detect from DICOM metadata)"
    )
    parser.add_argument(
        "--no-pathology-screen",
        action="store_true",
        help="Skip pathology screening in comprehensive mode"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (instead of full analysis)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "Output"),
        help="Directory to save reports"
    )
    parser.add_argument(
        "--skip-download-check",
        action="store_true",
        help="Skip checking if model is downloaded"
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Include all series even if they appear to be duplicates at different slice thicknesses"
    )
    parser.add_argument(
        "--summarise",
        action="store_true",
        help="After analysis, summarise all reports for each scan using OpenRouter LLM"
    )
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help="Skip analysis and only summarise existing reports"
    )
    parser.add_argument(
        "--summarise-model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="OpenRouter model ID for summarisation (default: anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--list-scans",
        action="store_true",
        help="List available scans in output directory that can be summarised"
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Generate segmentation masks for identified abnormalities (saves as NIfTI)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for analysis (default: {DEFAULT_MODEL}). Options: {', '.join(AVAILABLE_MODELS)}"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to run, or 'all' for all available models. "
             "When multiple models are specified, each produces a separate report and a "
             "comparison report is generated. Example: --models m3d-lamed,med3dvlm"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison report when using multiple models"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and their capabilities"
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("3D Medical Image Analysis (Multi-Model)")
    print("=" * 70)
    print()

    # Handle --list-models
    if args.list_models:
        from src.infer import list_models, get_model_info, list_vqa_models, list_classifier_models

        print("Available Models:")
        print("-" * 50)

        vqa_models = list_vqa_models()
        classifier_models = list_classifier_models()

        if vqa_models:
            print("\nVQA Models (text responses):")
            for name in vqa_models:
                try:
                    info = get_model_info(name)
                    shape = info.get("input_shape", "?")
                    channels = info.get("channels", "?")
                    print(f"  - {name}: {shape} x {channels}ch")
                except Exception:
                    print(f"  - {name}")

        if classifier_models:
            print("\nClassifier Models (pathology detection):")
            for name in classifier_models:
                try:
                    info = get_model_info(name)
                    shape = info.get("input_shape", "?")
                    print(f"  - {name}: {shape}")
                except Exception:
                    print(f"  - {name}")

        print()
        print("Usage:")
        print("  --model m3d-lamed          # Single model")
        print("  --models m3d-lamed,radfm   # Multiple models")
        print("  --models all               # All available models")
        return 0

    # Determine which models to use
    models_to_use = []
    if args.models:
        if args.models.lower() == "all":
            models_to_use = AVAILABLE_MODELS.copy()
        else:
            models_to_use = [m.strip().lower() for m in args.models.split(",")]
            # Validate model names
            for m in models_to_use:
                if m not in AVAILABLE_MODELS:
                    print(f"Error: Unknown model '{m}'")
                    print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
                    return 1
    else:
        models_to_use = [args.model.lower()]

    print(f"Models to use: {', '.join(models_to_use)}")
    print()

    # Handle --list-scans (list available output scans for summarisation)
    if args.list_scans:
        from src.openrouter import list_available_scans
        output_dir = Path(args.output_dir)
        scans = list_available_scans(str(output_dir))

        if not scans:
            print("No scans found in output directory.")
            return 0

        print(f"Available scans in {output_dir}:\n")
        for scan in scans:
            date_str = scan["date"]
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            print(f"  [{date_formatted}] {scan['modality']} - {scan['report_count']} reports")
            print(f"    Path: {scan['path']}")
        return 0

    # Handle --summarise-only (skip analysis, just summarise existing reports)
    if args.summarise_only:
        from src.openrouter import list_available_scans, summarise_scan, get_api_key

        if not get_api_key():
            print("ERROR: OPENROUTER_API_KEY environment variable not set.")
            print("Set it with: set OPENROUTER_API_KEY=your-key-here")
            return 1

        output_dir = Path(args.output_dir)
        scans = list_available_scans(str(output_dir))

        if not scans:
            print("No scans found to summarise.")
            return 0

        print(f"Summarising {len(scans)} scan(s) using {args.summarise_model}...\n")

        for scan in scans:
            date_str = scan["date"]
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            print(f"Processing: {date_formatted} {scan['modality']} ({scan['report_count']} reports)")

            result = summarise_scan(
                output_dir=str(output_dir),
                date=scan["date"],
                modality=scan["modality"],
                model=args.summarise_model,
            )

            if result["success"]:
                print(f"  Success: {result['summary_path']}\n")
            else:
                print(f"  Error: {result['error']}\n")

        print("Summarisation complete.")
        return 0

    # Check input directory
    if not INPUT_DICOM_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DICOM_DIR}")
        return 1

    # Find all DICOM series
    print("Scanning for DICOM series...")
    series_list = find_dicom_series_recursive(INPUT_DICOM_DIR)

    if not series_list:
        print("No DICOM series found in Input/DICOM directory.")
        return 1

    print(f"Found {len(series_list)} DICOM series:\n")

    # Get info for each series
    series_info = []
    for series_path, file_count in series_list:
        info = get_series_info(series_path)
        series_info.append(info)
        rel_path = Path(series_path).relative_to(INPUT_DICOM_DIR)
        print(f"  [{len(series_info)}] {rel_path}")
        thickness_str = f"{info['slice_thickness']}mm" if info['slice_thickness'] else "?"
        print(f"      Modality: {info['modality']}, Slices: {info['slice_count']}, Thickness: {thickness_str}")
        print(f"      Desc: {info['series_description']}")

    print()

    # List mode - just show series and exit
    if args.list:
        return 0

    # Filter out non-scan series (Patient Protocol, scouts, etc.)
    series_info = filter_non_scan_series(series_info)

    # Filter duplicate series (same scan at different slice thicknesses)
    if not args.include_duplicates:
        original_count = len(series_info)
        series_info = filter_duplicate_series(series_info)
        if len(series_info) < original_count:
            print()

    if series_info:
        print(f"\n{len(series_info)} series to analyse after filtering")
        print()

    # Determine which series to analyse
    series_to_analyse = []

    if args.series:
        # Find matching series
        target = args.series.replace("\\", "/")
        for info in series_info:
            if target in info["path"].replace("\\", "/"):
                series_to_analyse.append(info)
        if not series_to_analyse:
            print(f"Error: No series matching '{args.series}' found.")
            return 1
    elif args.all:
        series_to_analyse = series_info
    else:
        # Interactive selection
        print("Select series to analyse:")
        print("  Enter number (1-{}) for single series".format(len(series_info)))
        print("  Enter 'all' to analyse all series")
        print("  Enter 'q' to quit")
        print()

        choice = input("Selection: ").strip().lower()

        if choice == 'q':
            return 0
        elif choice == 'all':
            series_to_analyse = series_info
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(series_info):
                    series_to_analyse = [series_info[idx]]
                else:
                    print("Invalid selection.")
                    return 1
            except ValueError:
                print("Invalid selection.")
                return 1

    print(f"\nWill analyse {len(series_to_analyse)} series.")

    # Note about segmentation mode
    if args.segment:
        supported_modes = ["simple", "comprehensive"]
        if args.mode not in supported_modes:
            print()
            print(f"NOTE: --segment is currently only supported with modes: {', '.join(supported_modes)}")
            print("      Segmentation will be skipped for other modes.")
        else:
            print("Segmentation enabled: will generate NIfTI masks for findings")

    # Check model exists (for each model to use)
    if not args.skip_download_check:
        from src.infer import get_model_info
        missing_models = []
        for model_name in models_to_use:
            try:
                info = get_model_info(model_name)
                model_path = Path(info["default_path"])
                # Check for config.json or similar
                config_exists = (model_path / "config.json").exists() or model_path.exists()
                if not config_exists and model_name == "m3d-lamed":
                    # M3D-LaMed must have config
                    missing_models.append(model_name)
            except Exception:
                pass  # Model may not be downloaded yet, warn later

        if missing_models:
            print()
            print(f"WARNING: Model(s) may not be downloaded: {', '.join(missing_models)}")
            print("Run: python scripts/download_model.py --model <name>")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    from src.infer import get_model
    from src.preprocessing import preprocess_for_m3d
    from src.utils import (
        get_dicom_metadata,
        detect_body_region,
        format_analysis_report,
        format_comprehensive_report,
        format_chained_report,
        format_pathology_report,
        format_classifier_report,
        save_segmentation_nifti,
        get_segmentation_stats,
    )

    # Process each series with each model
    all_results = []
    comparison_data = {}  # For generating comparison reports

    for i, info in enumerate(series_to_analyse):
        print()
        print("=" * 70)
        rel_path = Path(info["path"]).relative_to(INPUT_DICOM_DIR)
        print(f"Analysing [{i+1}/{len(series_to_analyse)}]: {rel_path}")
        print("=" * 70)

        # Determine modality
        if args.modality == "auto":
            modality = info["modality"] if info["modality"] in ["CT", "MR", "MRI"] else "CT"
            if modality == "MR":
                modality = "MRI"
        else:
            modality = args.modality

        print(f"Modality: {modality}")

        # Get metadata
        metadata = get_dicom_metadata(info["path"])

        # Determine body region (auto-detect or use specified)
        if args.region == "auto":
            region = detect_body_region(metadata)
            print(f"Body region (auto-detected): {region}")
        else:
            region = args.region
            print(f"Body region (specified): {region}")

        # Track results for this series across models (for comparison)
        series_model_results = {}

        # Process with each model
        for model_name in models_to_use:
            print()
            print(f"----- Model: {model_name} -----")

            try:
                # Load model
                print(f"Loading {model_name}...")
                start_time = time.time()
                model = get_model(model_name)
                model.load_model()
                load_time = time.time() - start_time
                print(f"  Loaded in {load_time:.1f}s")

                # Check if model supports the requested mode
                is_classifier = model.model_type == "classifier"

                if args.question:
                    # Single question mode
                    if is_classifier:
                        print(f"  Classifier model - running classification instead of Q&A")
                        results = model.classify(image=info["path"], modality=modality)
                        print(f"  Classification results: {results}")
                        all_results.append({
                            "series": str(rel_path),
                            "model": model_name,
                            "results": results,
                        })
                    else:
                        print(f"  Question: {args.question}")

                        start_time = time.time()
                        response = model.generate_response(
                            image=info["path"],
                            question=args.question,
                            modality=modality,
                        )
                        inference_time = time.time() - start_time

                        print("  Response:")
                        print("  " + "-" * 40)
                        print(f"  {response}")
                        print("  " + "-" * 40)
                        print(f"  Generated in {inference_time:.1f}s")

                        all_results.append({
                            "series": str(rel_path),
                            "model": model_name,
                            "question": args.question,
                            "response": response,
                        })
                        series_model_results[model_name] = {"response": response}
                else:
                    # Full analysis mode
                    # Preprocess to check for multiple volumes (mixed dimensions)
                    print("  Preprocessing DICOM series...")
                    preprocessed = preprocess_for_m3d(info["path"], modality=modality)

                    # Determine volumes to process
                    if isinstance(preprocessed, list):
                        volumes_to_process = preprocessed
                        print(f"    Detected {len(volumes_to_process)} dimension groups")
                    else:
                        volumes_to_process = [(None, preprocessed)]

                    # Process each volume
                    for vol_label, vol_data in volumes_to_process:
                        if vol_label:
                            print(f"\n    Processing sub-volume: {vol_label}")

                        # Handle classifier vs VQA models
                        if is_classifier:
                            print(f"    Running classification...")
                            start_time = time.time()
                            results = model.analyse_scan(
                                image=vol_data,
                                modality=modality,
                            )
                            inference_time = time.time() - start_time
                            report = format_classifier_report(results, metadata, model_name)
                            summary_key = "overall_findings"
                        elif args.mode == "comprehensive" and hasattr(model, 'comprehensive_analysis'):
                            print(f"    Running {args.mode} analysis...")
                            start_time = time.time()
                            results = model.comprehensive_analysis(
                                image=vol_data,
                                modality=modality,
                                region=region,
                                include_pathology_screen=not args.no_pathology_screen,
                                segment=args.segment,
                            )
                            inference_time = time.time() - start_time
                            report = format_comprehensive_report(results, metadata)
                            summary_key = "report"
                        elif args.mode == "chained" and hasattr(model, 'analyse_chained'):
                            print(f"    Running chained analysis...")
                            start_time = time.time()
                            results = model.analyse_chained(
                                image=vol_data,
                                modality=modality,
                                quick=False,
                            )
                            inference_time = time.time() - start_time
                            report = format_chained_report(results, metadata)
                            summary_key = "summary"
                        elif args.mode == "quick" and hasattr(model, 'analyse_chained'):
                            print(f"    Running quick analysis...")
                            start_time = time.time()
                            results = model.analyse_chained(
                                image=vol_data,
                                modality=modality,
                                quick=True,
                            )
                            inference_time = time.time() - start_time
                            report = format_chained_report(results, metadata)
                            summary_key = "summary"
                        elif args.mode == "pathology" and hasattr(model, 'screen_pathology'):
                            print(f"    Running pathology screening...")
                            start_time = time.time()
                            results = model.screen_pathology(
                                image=vol_data,
                                region=region,
                                modality=modality,
                            )
                            inference_time = time.time() - start_time
                            report = format_pathology_report(results, metadata)
                            summary_key = None
                        else:
                            # Default: simple analysis (works for all VQA models)
                            print(f"    Running simple analysis...")
                            start_time = time.time()
                            results = model.analyse_scan(
                                image=vol_data,
                                modality=modality,
                                segment=args.segment if hasattr(model, 'generate_with_segmentation') else False,
                            )
                            inference_time = time.time() - start_time
                            report = format_analysis_report(results, metadata)
                            summary_key = "overall_findings"

                        # Build output path: Output/{date}/{modality}/{model}/{series}.md
                        study_date = metadata.get("StudyDate", "Unknown")
                        date_str = study_date.replace("-", "") if study_date != "Unknown" else "00000000"
                        series_desc = metadata.get("SeriesDescription", "Unknown")
                        safe_series = "".join(c if c.isalnum() or c in " -_" else "_" for c in series_desc).strip()

                        # Include model name in path if using multiple models
                        if len(models_to_use) > 1:
                            series_output_dir = output_dir / date_str / modality / model_name
                        else:
                            series_output_dir = output_dir / date_str / modality
                        series_output_dir.mkdir(parents=True, exist_ok=True)

                        if vol_label:
                            report_filename = f"{safe_series} ({vol_label}).md"
                        else:
                            report_filename = f"{safe_series}.md"

                        report_path = series_output_dir / report_filename
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(report)

                        # Save segmentation masks if generated
                        if args.segment and results.get("segmentations"):
                            seg_count = len(results["segmentations"])
                            print(f"    Generated {seg_count} segmentation mask(s)")

                            for seg_name, seg_mask in results["segmentations"].items():
                                seg_filename = f"{safe_series}_{seg_name}.nii.gz"
                                if vol_label:
                                    seg_filename = f"{safe_series} ({vol_label})_{seg_name}.nii.gz"
                                seg_path = series_output_dir / seg_filename

                                saved_path = save_segmentation_nifti(
                                    segmentation=seg_mask,
                                    output_path=str(seg_path),
                                    reference_dicom_dir=info["path"],
                                )
                                print(f"      Saved: {seg_path.name}")

                                stats = get_segmentation_stats(seg_mask)
                                if stats["voxel_count"] > 0:
                                    print(f"        Volume: {stats['volume_ml']:.2f} ml ({stats['voxel_count']} voxels)")

                        print(f"    Analysis complete in {inference_time:.1f}s")
                        print(f"    Report saved to: {report_path}")

                        # Store for comparison
                        series_model_results[model_name] = {
                            "results": results,
                            "report_path": str(report_path),
                            "summary_key": summary_key,
                        }

                        all_results.append({
                            "series": str(rel_path),
                            "model": model_name,
                            "volume_label": vol_label,
                            "report_path": str(report_path),
                            "results": results,
                            "mode": args.mode,
                        })

                # Unload model to free memory before loading next
                model.unload_model()

            except Exception as e:
                print(f"  ERROR with {model_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "series": str(rel_path),
                    "model": model_name,
                    "error": str(e),
                })
                # Try to unload even on error
                try:
                    model.unload_model()
                except:
                    pass

        # Store comparison data for this series
        if len(series_model_results) > 1:
            comparison_data[str(rel_path)] = series_model_results

    # Summary
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"Processed: {len(series_to_analyse)} series")
    print(f"Output directory: {output_dir}")

    # Print any errors
    errors = [r for r in all_results if "error" in r]
    if errors:
        print()
        print("Errors encountered:")
        for r in errors:
            print(f"  - {r['series']}: {r['error']}")

    # Post-processing: Summarise reports if requested
    if args.summarise and all_results:
        from src.openrouter import summarise_scan, get_api_key

        print()
        print("-" * 70)
        print("POST-PROCESSING: Generating Summaries")
        print("-" * 70)

        if not get_api_key():
            print()
            print("WARNING: OPENROUTER_API_KEY not set. Skipping summarisation.")
            print("Set it with: set OPENROUTER_API_KEY=your-key-here")
        else:
            # Collect unique date/modality combinations from results
            scans_to_summarise = set()
            for result in all_results:
                if "error" not in result and "report_path" in result:
                    report_path = Path(result["report_path"])
                    # Extract date and modality from path: Output/{date}/{modality}/{file}.md
                    try:
                        modality = report_path.parent.name
                        date = report_path.parent.parent.name
                        if date.isdigit() and len(date) == 8:
                            scans_to_summarise.add((date, modality))
                    except Exception:
                        pass

            if scans_to_summarise:
                print(f"\nSummarising {len(scans_to_summarise)} scan(s) using {args.summarise_model}...")

                for date, modality in sorted(scans_to_summarise):
                    date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
                    print(f"\n  Summarising: {date_formatted} {modality}")

                    result = summarise_scan(
                        output_dir=str(output_dir),
                        date=date,
                        modality=modality,
                        model=args.summarise_model,
                    )

                    if result["success"]:
                        print(f"    Saved: {result['summary_path']}")
                    else:
                        print(f"    Error: {result['error']}")
            else:
                print("\nNo valid reports to summarise.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
