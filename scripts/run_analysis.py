"""
CLI Entry Point for M3D-LaMed Analysis

Command-line interface for running medical image analysis.
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Analyse medical images using M3D-LaMed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pathology screening on CT
  python run_analysis.py "D:\\DICOM\\patient001\\CT_HEAD" --modality CT --full-analysis

  # Single question
  python run_analysis.py "D:\\DICOM\\patient001\\CT_HEAD" --question "Is there evidence of mastoiditis?"

  # Save report to file
  python run_analysis.py "D:\\DICOM\\patient001\\CT_HEAD" --full-analysis --output report.md
"""
    )

    parser.add_argument(
        "dicom_path",
        type=str,
        help="Path to DICOM directory"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["CT", "MRI"],
        default="CT",
        help="Image modality (default: CT)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask about the image"
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Run full pathology screening"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Override default model path"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Validate DICOM path
    dicom_path = Path(args.dicom_path)
    if not dicom_path.exists():
        print(f"Error: DICOM path does not exist: {dicom_path}")
        return 1

    # Must specify either question or full-analysis
    if not args.question and not args.full_analysis:
        print("Error: Must specify either --question or --full-analysis")
        return 1

    # Import modules
    from src.inference import M3DLaMedInference
    from src.utils import get_dicom_metadata, format_analysis_report

    # Print DICOM metadata
    print()
    print("=" * 60)
    print("M3D-LaMed Medical Image Analysis")
    print("=" * 60)
    print()

    print("Loading DICOM metadata...")
    metadata = get_dicom_metadata(str(dicom_path))

    print()
    print("Study Information:")
    print(f"  Patient ID: {metadata['PatientID']}")
    print(f"  Study Date: {metadata['StudyDate']}")
    print(f"  Modality: {metadata['Modality']}")
    print(f"  Series: {metadata['SeriesDescription']}")
    print(f"  Slices: {metadata['NumberOfSlices']}")
    if metadata['Rows'] and metadata['Columns']:
        print(f"  Dimensions: {metadata['Rows']} x {metadata['Columns']}")
    print()

    # Initialise model
    print("Initialising model...")
    start_time = time.time()

    model_kwargs = {}
    if args.model_path:
        model_kwargs["model_path"] = args.model_path

    inference = M3DLaMedInference(**model_kwargs)
    inference.load_model()

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    print()

    # Run analysis
    if args.question:
        # Single question mode
        print(f"Question: {args.question}")
        print()
        print("Generating response...")
        print("-" * 40)

        start_time = time.time()
        response = inference.generate_response(
            str(dicom_path),
            args.question,
            modality=args.modality,
        )
        inference_time = time.time() - start_time

        print(response)
        print("-" * 40)
        print(f"Response generated in {inference_time:.1f}s")

        # Save if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# M3D-LaMed Analysis\n\n")
                f.write(f"**Question**: {args.question}\n\n")
                f.write(f"**Response**:\n\n{response}\n")
            print(f"Response saved to: {output_path}")

    else:
        # Full analysis mode
        print("Running full pathology screening...")
        print("-" * 40)

        start_time = time.time()
        results = inference.analyse_for_pathology(
            str(dicom_path),
            modality=args.modality,
        )
        inference_time = time.time() - start_time

        # Format report
        report = format_analysis_report(results, metadata)

        print()
        print(report)
        print("-" * 40)
        print(f"Analysis completed in {inference_time:.1f}s")

        # Save if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

    # Cleanup
    inference.unload_model()

    return 0


if __name__ == "__main__":
    sys.exit(main())
