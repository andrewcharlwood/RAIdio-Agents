"""
Example Usage Patterns for M3D-LaMed

Demonstrates various ways to use the M3D-LaMed inference API.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Example 1: Single Question
# =============================================================================
def example_single_question():
    """Ask a single question about a CT scan."""
    from src.inference import M3DLaMedInference

    # Initialise model
    model = M3DLaMedInference()

    # Path to your DICOM directory
    dicom_path = r"D:\DICOM\patient001\CT_HEAD"

    # Ask a question
    response = model.generate_response(
        image=dicom_path,
        question="Describe the mastoid air cells. Are there any signs of opacification?",
        modality="CT",
    )

    print("Response:", response)

    # Clean up
    model.unload_model()


# =============================================================================
# Example 2: Full Pathology Screening
# =============================================================================
def example_full_screening():
    """Run comprehensive pathology screening."""
    from src.inference import M3DLaMedInference
    from src.utils import format_analysis_report, get_dicom_metadata

    # Initialise model
    model = M3DLaMedInference()

    # Path to your DICOM directory
    dicom_path = r"D:\DICOM\patient001\CT_HEAD"

    # Get metadata for the report
    metadata = get_dicom_metadata(dicom_path)

    # Run full analysis with default pathologies
    results = model.analyse_for_pathology(
        image=dicom_path,
        modality="CT",
    )

    # Format as report
    report = format_analysis_report(results, metadata)
    print(report)

    # Save to file
    with open("analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    model.unload_model()


# =============================================================================
# Example 3: Custom Pathology List
# =============================================================================
def example_custom_pathologies():
    """Screen for a custom set of pathologies."""
    from src.inference import M3DLaMedInference

    model = M3DLaMedInference()

    dicom_path = r"D:\DICOM\patient001\CT_HEAD"

    # Define custom pathologies to screen for
    custom_pathologies = [
        "temporal bone fracture",
        "cholesteatoma",
        "labyrinthitis",
        "facial nerve involvement",
    ]

    results = model.analyse_for_pathology(
        image=dicom_path,
        modality="CT",
        pathologies=custom_pathologies,
    )

    # Print each assessment
    for pathology, assessment in results["pathology_assessment"].items():
        print(f"\n=== {pathology.upper()} ===")
        print(assessment)

    model.unload_model()


# =============================================================================
# Example 4: Processing Multiple Studies
# =============================================================================
def example_batch_processing():
    """Process multiple DICOM studies."""
    from src.inference import M3DLaMedInference
    from src.utils import find_dicom_series

    model = M3DLaMedInference()

    # Find all DICOM series in a directory
    root_dir = r"D:\DICOM\batch_studies"
    dicom_dirs = find_dicom_series(root_dir)

    print(f"Found {len(dicom_dirs)} DICOM series")

    results = []
    for dicom_dir in dicom_dirs:
        print(f"\nProcessing: {dicom_dir}")

        # Quick screening question
        response = model.generate_response(
            image=dicom_dir,
            question="Summarise any significant findings in this scan in 2-3 sentences.",
            modality="CT",
        )

        results.append({
            "path": dicom_dir,
            "summary": response,
        })

    # Print summaries
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)

    for r in results:
        print(f"\n{r['path']}:")
        print(f"  {r['summary']}")

    model.unload_model()


# =============================================================================
# Example 5: Pre-processed Input
# =============================================================================
def example_preprocessed_input():
    """Use pre-processed numpy array as input."""
    import numpy as np
    from src.inference import M3DLaMedInference
    from src.preprocessing import preprocess_for_m3d

    # Pre-process the DICOM
    dicom_path = r"D:\DICOM\patient001\CT_HEAD"
    preprocessed = preprocess_for_m3d(dicom_path, modality="CT")

    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed dtype: {preprocessed.dtype}")

    # Now use with model
    model = M3DLaMedInference()

    # Can reuse preprocessed array for multiple questions
    questions = [
        "Are the mastoid air cells normally aerated?",
        "Is there any abnormal enhancement pattern?",
        "Describe the appearance of the sigmoid sinus.",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        response = model.generate_response(
            image=preprocessed,  # Pass numpy array directly
            question=q,
            modality="CT",
        )
        print(f"A: {response}")

    model.unload_model()


# =============================================================================
# Example 6: Custom System Prompt
# =============================================================================
def example_custom_system_prompt():
    """Use a custom system prompt for specialised analysis."""
    from src.inference import M3DLaMedInference

    model = M3DLaMedInference()

    dicom_path = r"D:\DICOM\patient001\CT_HEAD"

    # Specialised system prompt for infection assessment
    infection_prompt = """You are an expert neuroradiology AI assistant specialising in
    infectious diseases of the head and neck. Your primary focus is identifying:
    1. Mastoiditis and its complications
    2. Cerebral venous sinus thrombosis (CVST)
    3. Intracranial spread of infection

    When analysing images, always comment on:
    - Mastoid air cell opacification
    - Dural enhancement
    - Venous sinus patency
    - Any signs of intracranial extension

    Be specific about anatomical locations and provide confidence levels for findings."""

    response = model.generate_response(
        image=dicom_path,
        question="Analyse this scan for signs of complicated mastoiditis.",
        modality="CT",
        system_prompt=infection_prompt,
    )

    print(response)
    model.unload_model()


# =============================================================================
# Example 7: Memory-Efficient Processing
# =============================================================================
def example_memory_efficient():
    """Process large datasets with memory management."""
    import torch
    from src.inference import M3DLaMedInference
    from src.utils import find_dicom_series

    # Find studies
    dicom_dirs = find_dicom_series(r"D:\DICOM\large_batch")

    # Process in chunks, reloading model periodically to clear memory
    chunk_size = 5

    for i in range(0, len(dicom_dirs), chunk_size):
        chunk = dicom_dirs[i:i + chunk_size]
        print(f"\nProcessing chunk {i // chunk_size + 1}...")

        # Load model
        model = M3DLaMedInference()

        for dicom_dir in chunk:
            response = model.generate_response(
                image=dicom_dir,
                question="Any acute findings?",
                modality="CT",
            )
            print(f"{Path(dicom_dir).name}: {response[:100]}...")

        # Unload and clear cache
        model.unload_model()
        torch.cuda.empty_cache()

        print(f"GPU memory cleared after chunk {i // chunk_size + 1}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("M3D-LaMed Example Usage")
    print("=" * 60)
    print()
    print("Available examples:")
    print("  1. example_single_question()")
    print("  2. example_full_screening()")
    print("  3. example_custom_pathologies()")
    print("  4. example_batch_processing()")
    print("  5. example_preprocessed_input()")
    print("  6. example_custom_system_prompt()")
    print("  7. example_memory_efficient()")
    print()
    print("To run an example, uncomment the function call below or")
    print("import this module and call the function directly.")
    print()
    print("Note: Update the DICOM paths in each example to match your data.")

    # Uncomment to run an example:
    # example_single_question()
    # example_full_screening()
    # example_custom_pathologies()
