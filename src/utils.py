"""
Utility Functions for M3D-LaMed Project

Helper functions for DICOM handling, metadata extraction, and report formatting.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk


# =============================================================================
# BODY REGION DETECTION
# =============================================================================

# Keywords for body region detection from SeriesDescription
BODY_REGION_KEYWORDS = {
    "head": [
        "head", "brain", "cranial", "skull", "cerebr", "neuro", "orbit",
        "sella", "temporal", "pituitary", "iac", "cta head", "ct head",
        "mri head", "mri brain", "venogram", "mastoid", "sinus", "facial",
    ],
    "neck": [
        "neck", "cervical", "c-spine", "cspine", "thyroid", "larynx",
        "pharynx", "carotid", "soft tissue neck",
    ],
    "chest": [
        "chest", "thorax", "thoracic", "lung", "pulmonary", "cardiac",
        "heart", "mediastin", "hrct", "ctpa", "pe study", "t-spine",
    ],
    "abdomen": [
        "abdo", "abdomen", "abdominal", "liver", "kidney", "renal",
        "pancrea", "spleen", "gallbladder", "bowel", "colon", "gastric",
        "hepat", "biliary", "urogram", "pelvis", "pelvic", "bladder",
        "prostate", "uterus", "ovary", "l-spine", "lspine", "lumbar",
    ],
}


def detect_body_region(metadata: dict) -> str:
    """
    Detect body region from DICOM metadata.

    Uses BodyPartExamined tag if available, otherwise infers from
    SeriesDescription using keyword matching.

    Args:
        metadata: Dictionary from get_dicom_metadata() or raw DICOM attributes

    Returns:
        Body region: "head", "neck", "chest", or "abdomen"
    """
    # Try BodyPartExamined first (most reliable)
    body_part = metadata.get("BodyPartExamined", "").lower()
    if body_part:
        if any(kw in body_part for kw in ["head", "brain", "skull"]):
            return "head"
        elif any(kw in body_part for kw in ["neck", "cervical"]):
            return "neck"
        elif any(kw in body_part for kw in ["chest", "thorax", "lung"]):
            return "chest"
        elif any(kw in body_part for kw in ["abdomen", "pelvis", "liver", "kidney"]):
            return "abdomen"

    # Fall back to SeriesDescription keyword matching
    series_desc = metadata.get("SeriesDescription", "").lower()
    study_desc = metadata.get("StudyDescription", "").lower()
    combined_desc = f"{series_desc} {study_desc}"

    # Check each region's keywords
    region_scores = {region: 0 for region in BODY_REGION_KEYWORDS}

    for region, keywords in BODY_REGION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_desc:
                region_scores[region] += 1

    # Return region with highest score, default to "head" if no matches
    best_region = max(region_scores, key=region_scores.get)
    if region_scores[best_region] > 0:
        return best_region

    # Default fallback
    return "head"


def detect_body_region_from_path(dicom_dir: str) -> str:
    """
    Detect body region by reading DICOM metadata from a directory.

    Args:
        dicom_dir: Path to directory containing DICOM files

    Returns:
        Body region: "head", "neck", "chest", or "abdomen"
    """
    dicom_path = Path(dicom_dir)
    if not dicom_path.exists():
        return "head"

    # Find a DICOM file to read
    for f in dicom_path.iterdir():
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)

                # Build metadata dict from DICOM
                metadata = {}
                if hasattr(ds, 'BodyPartExamined'):
                    metadata["BodyPartExamined"] = str(ds.BodyPartExamined)
                if hasattr(ds, 'SeriesDescription'):
                    metadata["SeriesDescription"] = str(ds.SeriesDescription)
                if hasattr(ds, 'StudyDescription'):
                    metadata["StudyDescription"] = str(ds.StudyDescription)

                return detect_body_region(metadata)

            except Exception:
                continue

    return "head"


# =============================================================================
# CLOSED-ENDED RESPONSE PARSING
# =============================================================================

def parse_closed_response(raw_response: str, choices: list = None) -> dict:
    """
    Parse and clean a closed-ended question response from the model.

    The model often outputs garbage like:
    - "A. Yes, definite mastoiditis B. Possible/subtle findings C. No evidence..."
    - "A. Yes, definite mastoiditis   food."
    - Contradictory statements

    This function extracts the actual answer and provides a clean display string.

    Args:
        raw_response: Raw model output
        choices: List of choices that were offered (e.g., ["A. Yes", "B. No"])

    Returns:
        Dictionary with:
        - answer_letter: Extracted letter (A, B, C, D) or None
        - answer_text: Clean text for the selected answer
        - confidence: "high", "medium", "low", or "uncertain"
        - display_text: Clean string for report display
    """
    result = {
        "answer_letter": None,
        "answer_text": "",
        "confidence": "uncertain",
        "display_text": raw_response.strip(),
    }

    if not raw_response:
        return result

    response = raw_response.strip()
    response_upper = response.upper()

    # Check for contradictory responses (contains both positive and negative)
    has_positive = any(p in response_upper for p in ["A. YES", "YES,", "DEFINITE", "PRESENT"])
    has_negative = any(n in response_upper for n in ["C. NO", "NO EVIDENCE", "NOT PRESENT", "NO HAEMORRHAGE", "NO MASS"])

    if has_positive and has_negative:
        # Contradictory - mark as uncertain
        result["confidence"] = "uncertain"
        result["display_text"] = "Inconclusive (contradictory response)"
        return result

    # Try to extract answer letter
    # Look for patterns like "A.", "A ", "A:" at start or "A. Yes" pattern
    answer_letter = None

    # Check if response starts with a letter choice
    for letter in ["A", "B", "C", "D"]:
        if response_upper.startswith(f"{letter}.") or response_upper.startswith(f"{letter} "):
            answer_letter = letter
            break

    # If no letter at start, look for first letter pattern in response
    if not answer_letter:
        import re
        # Match patterns like "A. Yes" or "A:" but not mid-word
        match = re.search(r'\b([ABCD])\.\s*[Yy]es|\b([ABCD])\.\s*[Nn]o|\b([ABCD])\.\s', response)
        if match:
            answer_letter = match.group(1) or match.group(2) or match.group(3)

    result["answer_letter"] = answer_letter

    # If we have choices, extract the text for the selected answer
    if answer_letter and choices:
        for choice in choices:
            if choice.strip().upper().startswith(f"{answer_letter}."):
                # Extract just the choice text (remove the letter prefix)
                result["answer_text"] = choice.split(".", 1)[1].strip() if "." in choice else choice
                break

    # Determine confidence based on answer
    if answer_letter == "A":
        result["confidence"] = "high"  # Typically "Yes, definite"
    elif answer_letter == "B":
        result["confidence"] = "medium"  # Typically "Possible/uncertain"
    elif answer_letter == "C":
        result["confidence"] = "high"  # Typically "No evidence" - confident negative
    elif answer_letter == "D":
        result["confidence"] = "low"  # "Cannot assess"

    # Build clean display text
    if result["answer_text"]:
        result["display_text"] = result["answer_text"]
    elif answer_letter:
        # Try to extract meaningful text after the letter
        # Remove multiple choice options that got echoed
        clean = response
        # Remove patterns like "B. Possible... C. No evidence... D. Cannot"
        for letter in ["B", "C", "D"]:
            idx = clean.upper().find(f" {letter}.")
            if idx > 0 and answer_letter != letter:
                clean = clean[:idx]

        # Remove the answer letter prefix if present
        if clean.upper().startswith(f"{answer_letter}."):
            clean = clean[2:].strip()

        # Truncate at common garbage patterns
        for pattern in ["   ", "\n\n", " food", " doubled", " marginal"]:
            if pattern in clean.lower():
                idx = clean.lower().find(pattern)
                clean = clean[:idx]

        result["display_text"] = clean.strip() if clean.strip() else f"Answer: {answer_letter}"
    else:
        # No letter found - truncate at first garbage
        clean = response[:100] if len(response) > 100 else response
        result["display_text"] = clean

    return result


def find_dicom_series(directory: str) -> List[str]:
    """
    Recursively search directory for DICOM series.

    Args:
        directory: Root directory to search

    Returns:
        List of directories containing valid DICOM files
    """
    dicom_dirs = []
    root_path = Path(directory)

    if not root_path.exists():
        return dicom_dirs

    # Track directories we've found DICOM files in
    checked_dirs = set()

    for root, dirs, files in os.walk(root_path):
        root_dir = Path(root)

        # Skip if we've already found DICOMs in a parent directory
        if any(root_dir.is_relative_to(d) for d in checked_dirs if d != root_dir):
            continue

        # Check each file
        for filename in files:
            filepath = root_dir / filename

            # Quick check: skip common non-DICOM extensions
            if filepath.suffix.lower() in ('.txt', '.xml', '.json', '.md', '.py', '.exe'):
                continue

            try:
                # Try to read as DICOM
                ds = pydicom.dcmread(str(filepath), stop_before_pixels=True, force=True)

                # Check for essential DICOM attributes
                if hasattr(ds, 'SOPClassUID') or hasattr(ds, 'Modality'):
                    dicom_dirs.append(str(root_dir))
                    checked_dirs.add(root_dir)
                    break  # Found DICOM in this dir, move to next

            except Exception:
                continue

    return sorted(set(dicom_dirs))


def get_dicom_metadata(dicom_dir: str) -> Dict:
    """
    Extract key metadata from a DICOM series.

    Args:
        dicom_dir: Path to directory containing DICOM files

    Returns:
        Dictionary with metadata fields
    """
    metadata = {
        "PatientID": "Unknown",
        "StudyDate": "Unknown",
        "Modality": "Unknown",
        "SeriesDescription": "Unknown",
        "StudyDescription": "Unknown",
        "BodyPartExamined": "",
        "SliceThickness": None,
        "PixelSpacing": None,
        "Rows": None,
        "Columns": None,
        "NumberOfSlices": 0,
    }

    dicom_path = Path(dicom_dir)
    if not dicom_path.exists():
        return metadata

    # Find DICOM files
    dicom_files = []
    for f in dicom_path.iterdir():
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if hasattr(ds, 'SOPClassUID') or hasattr(ds, 'Modality'):
                    dicom_files.append(f)
            except Exception:
                continue

    if not dicom_files:
        return metadata

    metadata["NumberOfSlices"] = len(dicom_files)

    # Read first file for metadata
    try:
        ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)

        # Extract fields with anonymisation for patient ID
        if hasattr(ds, 'PatientID') and ds.PatientID:
            pid = str(ds.PatientID)
            # Anonymise: show only last 4 characters
            metadata["PatientID"] = f"***{pid[-4:]}" if len(pid) > 4 else pid

        if hasattr(ds, 'StudyDate') and ds.StudyDate:
            # Format date nicely
            try:
                date = datetime.strptime(ds.StudyDate, "%Y%m%d")
                metadata["StudyDate"] = date.strftime("%Y-%m-%d")
            except ValueError:
                metadata["StudyDate"] = ds.StudyDate

        if hasattr(ds, 'Modality'):
            metadata["Modality"] = ds.Modality

        if hasattr(ds, 'SeriesDescription'):
            metadata["SeriesDescription"] = ds.SeriesDescription

        if hasattr(ds, 'StudyDescription'):
            metadata["StudyDescription"] = ds.StudyDescription

        if hasattr(ds, 'BodyPartExamined'):
            metadata["BodyPartExamined"] = ds.BodyPartExamined

        if hasattr(ds, 'SliceThickness'):
            metadata["SliceThickness"] = float(ds.SliceThickness)

        if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
            metadata["PixelSpacing"] = [float(x) for x in ds.PixelSpacing]

        if hasattr(ds, 'Rows'):
            metadata["Rows"] = int(ds.Rows)

        if hasattr(ds, 'Columns'):
            metadata["Columns"] = int(ds.Columns)

    except Exception as e:
        metadata["_error"] = str(e)

    return metadata


def format_analysis_report(results: Dict, metadata: Optional[Dict] = None) -> str:
    """
    Format analysis results as a readable markdown report.

    Args:
        results: Output from M3DLaMedInference.analyse_for_pathology()
        metadata: Optional DICOM metadata to include

    Returns:
        Formatted markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# M3D-LaMed Analysis Report",
        "",
        f"**Generated**: {timestamp}",
        "",
    ]

    # Add metadata section if provided
    if metadata:
        lines.extend([
            "## Study Information",
            "",
            f"- **Patient ID**: {metadata.get('PatientID', 'Unknown')}",
            f"- **Study Date**: {metadata.get('StudyDate', 'Unknown')}",
            f"- **Modality**: {metadata.get('Modality', 'Unknown')}",
            f"- **Series**: {metadata.get('SeriesDescription', 'Unknown')}",
            f"- **Dimensions**: {metadata.get('Rows', '?')} x {metadata.get('Columns', '?')} x {metadata.get('NumberOfSlices', '?')}",
            "",
        ])

    # Overall findings
    lines.extend([
        "## Overall Findings",
        "",
        results.get("overall_findings", "No findings available."),
        "",
    ])

    # Analysis sections
    if "analysis" in results and results["analysis"]:
        lines.extend([
            "## Detailed Analysis",
            "",
        ])

        # Map internal names to display titles
        title_map = {
            "abnormalities": "Abnormalities & Pathological Findings",
            "key_findings": "Key Clinical Findings",
            "differential": "Differential Diagnoses",
        }

        for name, assessment in results["analysis"].items():
            title = title_map.get(name, name.replace("_", " ").title())
            lines.extend([
                f"### {title}",
                "",
                assessment,
                "",
            ])

    # Recommendations
    if "recommendations" in results and results["recommendations"]:
        lines.extend([
            "## Recommendations",
            "",
            results["recommendations"],
            "",
        ])

    # Disclaimer
    lines.extend([
        "---",
        "",
        "**Disclaimer**: This analysis was generated by an AI model (M3D-LaMed) ",
        "for research purposes only. It should NOT be used as the sole basis for ",
        "clinical decision-making. Always verify findings with qualified radiologists ",
        "and correlate with clinical presentation.",
        "",
    ])

    return "\n".join(lines)


def format_comprehensive_report(results: Dict, metadata: Optional[Dict] = None) -> str:
    """
    Format comprehensive analysis results as a readable markdown report.

    Args:
        results: Output from M3DLaMedInference.comprehensive_analysis()
        metadata: Optional DICOM metadata to include

    Returns:
        Formatted markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# M3D-LaMed Comprehensive Analysis Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Analysis Mode**: Comprehensive",
        "",
    ]

    # Add metadata section if provided
    if metadata:
        lines.extend([
            "## Study Information",
            "",
            f"- **Patient ID**: {metadata.get('PatientID', 'Unknown')}",
            f"- **Study Date**: {metadata.get('StudyDate', 'Unknown')}",
            f"- **Modality**: {metadata.get('Modality', 'Unknown')}",
            f"- **Series**: {metadata.get('SeriesDescription', 'Unknown')}",
            f"- **Dimensions**: {metadata.get('Rows', '?')} x {metadata.get('Columns', '?')} x {metadata.get('NumberOfSlices', '?')}",
            f"- **Body Region**: {results.get('region', 'Unknown')}",
            "",
        ])

    # Image parameters (high-accuracy findings)
    if "image_parameters" in results and results["image_parameters"]:
        lines.extend([
            "## Image Parameters",
            "",
        ])
        for param, value in results["image_parameters"].items():
            lines.extend([
                f"**{param.title()}**: {value}",
                "",
            ])

    # Findings analysis
    if "findings" in results and results["findings"]:
        lines.extend([
            "## Detailed Findings",
            "",
        ])

        title_map = {
            "abnormalities": "Abnormalities & Pathological Findings",
            "key_findings": "Key Clinical Findings",
            "differential": "Differential Diagnoses",
        }

        for name, assessment in results["findings"].items():
            title = title_map.get(name, name.replace("_", " ").title())
            lines.extend([
                f"### {title}",
                "",
                assessment,
                "",
            ])

    # Pathology screening results
    if "pathology_screening" in results and results["pathology_screening"]:
        screening = results["pathology_screening"]
        lines.extend([
            "## Pathology Screening",
            "",
            f"**Region**: {screening.get('region', 'Unknown').title()}",
            "",
        ])

        # Positive findings (priority)
        if screening.get("positive_findings"):
            lines.extend([
                "### Positive Findings",
                "",
            ])
            for finding in screening["positive_findings"]:
                screen_data = screening["screenings"].get(finding, {})
                # Parse and clean the response
                parsed = parse_closed_response(
                    screen_data.get('raw_response', ''),
                    screen_data.get('choices', [])
                )
                lines.append(f"- **{finding.replace('_', ' ').title()}**: {parsed['display_text']}")
            lines.append("")

        # Uncertain findings
        if screening.get("uncertain_findings"):
            lines.extend([
                "### Uncertain Findings (Requires Follow-up)",
                "",
            ])
            for finding in screening["uncertain_findings"]:
                screen_data = screening["screenings"].get(finding, {})
                # Parse and clean the response
                parsed = parse_closed_response(
                    screen_data.get('raw_response', ''),
                    screen_data.get('choices', [])
                )
                lines.append(f"- **{finding.replace('_', ' ').title()}**: {parsed['display_text']}")
            lines.append("")

        # Negative findings summary
        negative = [
            name for name in screening.get("screenings", {}).keys()
            if name not in screening.get("positive_findings", [])
            and name not in screening.get("uncertain_findings", [])
        ]
        if negative:
            lines.extend([
                "### Negative Findings",
                "",
                "No evidence of: " + ", ".join(f.replace("_", " ") for f in negative),
                "",
            ])

    # Generated report
    if "report" in results and results["report"]:
        lines.extend([
            "## AI-Generated Report",
            "",
            results["report"],
            "",
        ])

    # Recommendations
    if "recommendations" in results and results["recommendations"]:
        lines.extend([
            "## Recommendations",
            "",
            results["recommendations"],
            "",
        ])

    # Disclaimer
    lines.extend([
        "---",
        "",
        "**Disclaimer**: This analysis was generated by an AI model (M3D-LaMed) ",
        "for research purposes only. It should NOT be used as the sole basis for ",
        "clinical decision-making. Always verify findings with qualified radiologists ",
        "and correlate with clinical presentation.",
        "",
    ])

    return "\n".join(lines)


def format_chained_report(results: Dict, metadata: Optional[Dict] = None) -> str:
    """
    Format chained analysis results as a readable markdown report.

    Args:
        results: Output from M3DLaMedInference.analyse_chained()
        metadata: Optional DICOM metadata to include

    Returns:
        Formatted markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# M3D-LaMed Chained Analysis Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Analysis Mode**: Chained Query",
        f"**Modality**: {results.get('modality', 'Unknown')}",
        "",
    ]

    # Add metadata section if provided
    if metadata:
        lines.extend([
            "## Study Information",
            "",
            f"- **Patient ID**: {metadata.get('PatientID', 'Unknown')}",
            f"- **Study Date**: {metadata.get('StudyDate', 'Unknown')}",
            f"- **Modality**: {metadata.get('Modality', 'Unknown')}",
            f"- **Series**: {metadata.get('SeriesDescription', 'Unknown')}",
            f"- **Dimensions**: {metadata.get('Rows', '?')} x {metadata.get('Columns', '?')} x {metadata.get('NumberOfSlices', '?')}",
            "",
        ])

    # Analysis chain results
    if "chain_results" in results and results["chain_results"]:
        lines.extend([
            "## Analysis Chain Results",
            "",
        ])

        # Display in order with questions and responses
        for step_name, step_data in results["chain_results"].items():
            title = step_name.replace("_", " ").title()
            lines.extend([
                f"### {title}",
                "",
                f"**Question**: {step_data.get('question', 'N/A')}",
                "",
                f"**Response**: {step_data.get('response', 'No response')}",
                "",
            ])

    # Summary
    if "summary" in results and results["summary"]:
        lines.extend([
            "## Clinical Summary",
            "",
            results["summary"],
            "",
        ])

    # Disclaimer
    lines.extend([
        "---",
        "",
        "**Disclaimer**: This analysis was generated by an AI model (M3D-LaMed) ",
        "for research purposes only. It should NOT be used as the sole basis for ",
        "clinical decision-making. Always verify findings with qualified radiologists ",
        "and correlate with clinical presentation.",
        "",
    ])

    return "\n".join(lines)


def format_pathology_report(results: Dict, metadata: Optional[Dict] = None) -> str:
    """
    Format pathology screening results as a readable markdown report.

    Args:
        results: Output from M3DLaMedInference.screen_pathology()
        metadata: Optional DICOM metadata to include

    Returns:
        Formatted markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# M3D-LaMed Pathology Screening Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Analysis Mode**: Pathology Screening",
        f"**Body Region**: {results.get('region', 'Unknown').title()}",
        f"**Modality**: {results.get('modality', 'Unknown')}",
        "",
    ]

    # Add metadata section if provided
    if metadata:
        lines.extend([
            "## Study Information",
            "",
            f"- **Patient ID**: {metadata.get('PatientID', 'Unknown')}",
            f"- **Study Date**: {metadata.get('StudyDate', 'Unknown')}",
            f"- **Modality**: {metadata.get('Modality', 'Unknown')}",
            f"- **Series**: {metadata.get('SeriesDescription', 'Unknown')}",
            f"- **Dimensions**: {metadata.get('Rows', '?')} x {metadata.get('Columns', '?')} x {metadata.get('NumberOfSlices', '?')}",
            "",
        ])

    # Summary of findings
    lines.extend([
        "## Screening Summary",
        "",
    ])

    positive = results.get("positive_findings", [])
    uncertain = results.get("uncertain_findings", [])
    total_screened = len(results.get("screenings", {}))

    if positive:
        lines.append(f"**⚠️ Positive Findings**: {len(positive)}")
    if uncertain:
        lines.append(f"**? Uncertain Findings**: {len(uncertain)}")
    lines.append(f"**Total Pathologies Screened**: {total_screened}")
    lines.append("")

    # Detailed positive findings
    if positive:
        lines.extend([
            "## Positive Findings (Require Attention)",
            "",
        ])
        for finding in positive:
            screen_data = results["screenings"].get(finding, {})
            lines.extend([
                f"### {finding.replace('_', ' ').title()}",
                "",
                f"**Question**: {screen_data.get('question', 'N/A')}",
                "",
                f"**Model Response**: {screen_data.get('raw_response', 'Detected')}",
                "",
            ])

    # Detailed uncertain findings
    if uncertain:
        lines.extend([
            "## Uncertain Findings (May Require Follow-up)",
            "",
        ])
        for finding in uncertain:
            screen_data = results["screenings"].get(finding, {})
            lines.extend([
                f"### {finding.replace('_', ' ').title()}",
                "",
                f"**Question**: {screen_data.get('question', 'N/A')}",
                "",
                f"**Model Response**: {screen_data.get('raw_response', 'Possible')}",
                "",
            ])

    # Negative findings
    negative = [
        name for name in results.get("screenings", {}).keys()
        if name not in positive and name not in uncertain
    ]
    if negative:
        lines.extend([
            "## Negative Findings",
            "",
            "The following pathologies were screened and not detected:",
            "",
        ])
        for finding in negative:
            lines.append(f"- {finding.replace('_', ' ').title()}")
        lines.append("")

    # Full screening details
    lines.extend([
        "## Detailed Screening Results",
        "",
        "| Pathology | Result | Model Response |",
        "|-----------|--------|----------------|",
    ])

    for name, screen_data in results.get("screenings", {}).items():
        answer = screen_data.get("answer_letter", "?")
        if answer == "A":
            result = "✓ Positive"
        elif answer == "B":
            result = "? Uncertain"
        elif answer == "C":
            result = "✗ Negative"
        else:
            result = "— Cannot assess"

        # Truncate response for table
        raw = screen_data.get("raw_response", "")
        raw_short = raw[:50] + "..." if len(raw) > 50 else raw
        lines.append(f"| {name.replace('_', ' ').title()} | {result} | {raw_short} |")

    lines.append("")

    # Disclaimer
    lines.extend([
        "---",
        "",
        "**Disclaimer**: This analysis was generated by an AI model (M3D-LaMed) ",
        "for research purposes only. It should NOT be used as the sole basis for ",
        "clinical decision-making. Always verify findings with qualified radiologists ",
        "and correlate with clinical presentation.",
        "",
    ])

    return "\n".join(lines)


# =============================================================================
# SEGMENTATION UTILITIES
# =============================================================================

def save_segmentation_nifti(
    segmentation: np.ndarray,
    output_path: str,
    reference_dicom_dir: Optional[str] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
) -> str:
    """
    Save a segmentation mask as a NIfTI file.

    The segmentation mask from M3D-LaMed is at model resolution (32, 256, 256).
    This function can optionally resample it to match the original DICOM dimensions.

    Args:
        segmentation: 3D numpy array (D, H, W) with values 0-1 or binary
        output_path: Path to save the .nii.gz file
        reference_dicom_dir: Optional path to original DICOM for spacing/orientation
        target_shape: Optional target shape to resample to (D, H, W)

    Returns:
        Path to saved NIfTI file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .nii.gz extension
    if not str(output_path).endswith('.nii.gz'):
        if str(output_path).endswith('.nii'):
            output_path = Path(str(output_path) + '.gz')
        else:
            output_path = Path(str(output_path) + '.nii.gz')

    # Convert to binary mask if needed
    seg_binary = (segmentation > 0.5).astype(np.uint8)

    # Default affine (identity with 1mm spacing)
    affine = np.eye(4)

    # Try to get spatial info from reference DICOM
    if reference_dicom_dir:
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(reference_dicom_dir))
            if dicom_names:
                reader.SetFileNames(dicom_names)
                ref_image = reader.Execute()

                # Get spacing and origin
                spacing = ref_image.GetSpacing()  # (X, Y, Z) = (W, H, D)
                origin = ref_image.GetOrigin()
                direction = ref_image.GetDirection()

                # Build affine matrix
                # NIfTI uses RAS orientation, DICOM typically LPS
                # SimpleITK direction is a 9-element tuple (3x3 matrix flattened)
                dir_matrix = np.array(direction).reshape(3, 3)

                affine = np.eye(4)
                affine[:3, :3] = dir_matrix * np.array(spacing)
                affine[:3, 3] = origin

                # Resample segmentation to original size if needed
                original_size = ref_image.GetSize()  # (W, H, D)
                original_shape = (original_size[2], original_size[1], original_size[0])  # (D, H, W)

                if target_shape is None and seg_binary.shape != original_shape:
                    target_shape = original_shape

        except Exception as e:
            print(f"Warning: Could not read reference DICOM for affine: {e}")

    # Resample to target shape if specified
    if target_shape and seg_binary.shape != target_shape:
        seg_binary = resample_segmentation(seg_binary, target_shape)

    # Create NIfTI image
    # NIfTI expects (X, Y, Z) ordering, our array is (D, H, W) = (Z, Y, X)
    # Transpose to (W, H, D) = (X, Y, Z) for NIfTI
    seg_nifti = np.transpose(seg_binary, (2, 1, 0))

    nii_img = nib.Nifti1Image(seg_nifti, affine)

    # Save
    nib.save(nii_img, str(output_path))

    return str(output_path)


def resample_segmentation(
    segmentation: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resample a segmentation mask to a target shape using nearest-neighbor interpolation.

    Args:
        segmentation: 3D numpy array (D, H, W)
        target_shape: Target shape (D, H, W)

    Returns:
        Resampled segmentation array
    """
    from scipy.ndimage import zoom

    # Calculate zoom factors
    factors = tuple(t / s for t, s in zip(target_shape, segmentation.shape))

    # Use nearest-neighbor interpolation for binary mask
    resampled = zoom(segmentation.astype(np.float32), factors, order=0)

    # Ensure binary
    return (resampled > 0.5).astype(np.uint8)


def get_segmentation_stats(segmentation: np.ndarray, spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
    """
    Calculate statistics for a segmentation mask.

    Args:
        segmentation: 3D binary numpy array (D, H, W)
        spacing_mm: Voxel spacing in mm (D, H, W)

    Returns:
        Dictionary with volume, voxel count, bounding box, etc.
    """
    binary = (segmentation > 0.5).astype(np.uint8)

    # Voxel count
    voxel_count = int(binary.sum())

    # Volume in mm³ and ml
    voxel_volume_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    # Bounding box
    if voxel_count > 0:
        nonzero = np.argwhere(binary)
        bbox_min = nonzero.min(axis=0).tolist()
        bbox_max = nonzero.max(axis=0).tolist()
        bbox_size = [mx - mn + 1 for mn, mx in zip(bbox_min, bbox_max)]
    else:
        bbox_min = [0, 0, 0]
        bbox_max = [0, 0, 0]
        bbox_size = [0, 0, 0]

    return {
        "voxel_count": voxel_count,
        "volume_mm3": round(volume_mm3, 2),
        "volume_ml": round(volume_ml, 3),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_size_voxels": bbox_size,
        "bbox_size_mm": [round(s * sp, 1) for s, sp in zip(bbox_size, spacing_mm)],
    }


def format_classifier_report(results: Dict, metadata: Optional[Dict] = None, model_name: str = "classifier") -> str:
    """
    Format classifier model results as a readable markdown report.

    Args:
        results: Output from classifier model's analyse_scan()
        metadata: Optional DICOM metadata to include
        model_name: Name of the classifier model

    Returns:
        Formatted markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {model_name.upper()} Classification Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Model**: {model_name}",
        f"**Model Type**: Classifier",
        "",
    ]

    # Add metadata section if provided
    if metadata:
        lines.extend([
            "## Study Information",
            "",
            f"- **Patient ID**: {metadata.get('PatientID', 'Unknown')}",
            f"- **Study Date**: {metadata.get('StudyDate', 'Unknown')}",
            f"- **Modality**: {metadata.get('Modality', 'Unknown')}",
            f"- **Series**: {metadata.get('SeriesDescription', 'Unknown')}",
            "",
        ])

    # Overall findings (formatted summary from classifier)
    if results.get("overall_findings"):
        lines.extend([
            "## Classification Results",
            "",
            results["overall_findings"],
            "",
        ])

    # Detailed analysis if available
    analysis = results.get("analysis", {})
    if analysis:
        # Pathology probabilities table
        probs = analysis.get("pathology_probabilities", {})
        if probs:
            lines.extend([
                "## Detailed Pathology Probabilities",
                "",
                "| Pathology | Probability |",
                "|-----------|-------------|",
            ])
            for pathology, prob in sorted(probs.items(), key=lambda x: -x[1]):
                emoji = "⚠️" if prob > 0.5 else "❓" if prob > 0.3 else ""
                lines.append(f"| {pathology.replace('_', ' ').title()} | {prob*100:.1f}% {emoji} |")
            lines.append("")

        # Positive findings
        positive = analysis.get("positive_findings", {})
        if positive:
            lines.extend([
                "## Positive Findings (>50% probability)",
                "",
            ])
            for pathology, prob in sorted(positive.items(), key=lambda x: -x[1]):
                lines.append(f"- **{pathology.replace('_', ' ').title()}**: {prob*100:.1f}%")
            lines.append("")

        # Uncertain findings
        uncertain = analysis.get("uncertain_findings", {})
        if uncertain:
            lines.extend([
                "## Uncertain Findings (30-50% probability)",
                "",
            ])
            for pathology, prob in sorted(uncertain.items(), key=lambda x: -x[1]):
                lines.append(f"- **{pathology.replace('_', ' ').title()}**: {prob*100:.1f}%")
            lines.append("")

    # Recommendations
    if results.get("recommendations"):
        lines.extend([
            "## Recommendations",
            "",
            results["recommendations"],
            "",
        ])

    # Disclaimer
    lines.extend([
        "---",
        "",
        "**DISCLAIMER**: This is an AI-generated analysis for research purposes only. ",
        "All findings must be verified by a qualified radiologist. ",
        "Do not use for clinical decision-making without professional review.",
    ])

    return "\n".join(lines)
