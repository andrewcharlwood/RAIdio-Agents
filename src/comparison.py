"""
Multi-Model Comparison Report Generator

Generates comparison reports when multiple models analyse the same scan,
highlighting agreements and disagreements between model outputs.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def generate_comparison_report(
    series_path: str,
    model_results: Dict[str, Dict],
    metadata: Optional[Dict] = None,
) -> str:
    """
    Generate a comparison report for multiple model outputs on the same series.

    Args:
        series_path: Path to the DICOM series
        model_results: Dict mapping model names to their results
            {
                "m3d-lamed": {"results": {...}, "report_path": "...", "summary_key": "..."},
                "med3dvlm": {"results": {...}, "report_path": "...", "summary_key": "..."},
            }
        metadata: Optional DICOM metadata

    Returns:
        Markdown formatted comparison report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models_used = list(model_results.keys())

    lines = [
        "# Multi-Model Comparison Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Series**: {series_path}",
        f"**Models Compared**: {', '.join(models_used)}",
        "",
    ]

    # Study information
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

    # Side-by-side summaries
    lines.extend([
        "## Model Summaries",
        "",
    ])

    for model_name, data in model_results.items():
        results = data.get("results", {})
        summary_key = data.get("summary_key", "overall_findings")

        # Get summary text
        if summary_key and summary_key in results:
            summary = results[summary_key]
        elif "overall_findings" in results:
            summary = results["overall_findings"]
        else:
            summary = "No summary available"

        # Truncate if too long
        if isinstance(summary, str) and len(summary) > 500:
            summary = summary[:500] + "..."

        lines.extend([
            f"### {model_name.upper()}",
            "",
            summary if summary else "_No findings reported_",
            "",
        ])

    # Individual report links
    lines.extend([
        "## Full Reports",
        "",
    ])
    for model_name, data in model_results.items():
        report_path = data.get("report_path", "N/A")
        lines.append(f"- **{model_name}**: `{report_path}`")
    lines.append("")

    # Try to identify common findings
    common_findings = _find_common_findings(model_results)
    if common_findings["agreements"]:
        lines.extend([
            "## Agreement Analysis",
            "",
            "### Findings Mentioned by Multiple Models",
            "",
        ])
        for finding in common_findings["agreements"]:
            lines.append(f"- {finding}")
        lines.append("")

    if common_findings["unique"]:
        lines.extend([
            "### Unique Findings (Single Model Only)",
            "",
        ])
        for model_name, findings in common_findings["unique"].items():
            if findings:
                lines.append(f"**{model_name}**:")
                for finding in findings:
                    lines.append(f"- {finding}")
                lines.append("")

    # Classifier results if any
    classifier_results = _extract_classifier_results(model_results)
    if classifier_results:
        lines.extend([
            "## Classifier Pathology Detection",
            "",
            "| Pathology | " + " | ".join(classifier_results.keys()) + " |",
            "|-----------|" + "|".join(["----------" for _ in classifier_results]) + "|",
        ])

        # Collect all pathologies
        all_pathologies = set()
        for model_data in classifier_results.values():
            all_pathologies.update(model_data.keys())

        for pathology in sorted(all_pathologies):
            row = f"| {pathology.replace('_', ' ').title()} |"
            for model_name, probs in classifier_results.items():
                prob = probs.get(pathology, 0.0)
                if prob > 0.5:
                    row += f" **{prob*100:.0f}%** |"
                elif prob > 0.3:
                    row += f" {prob*100:.0f}%? |"
                else:
                    row += f" {prob*100:.0f}% |"
            lines.append(row)
        lines.append("")

    # Disclaimer
    lines.extend([
        "---",
        "",
        "**DISCLAIMER**: This comparison is for research purposes only. ",
        "Disagreements between models should be reviewed by a qualified radiologist. ",
        "Agreement does not guarantee accuracy.",
    ])

    return "\n".join(lines)


def _find_common_findings(model_results: Dict[str, Dict]) -> Dict:
    """
    Attempt to identify common findings across models.

    Simple keyword matching - looks for common medical terms.
    """
    # Keywords to look for
    finding_keywords = [
        "normal", "abnormal", "mass", "nodule", "lesion", "tumor",
        "effusion", "opacity", "hemorrhage", "fracture", "edema",
        "atelectasis", "consolidation", "pneumonia", "cardiomegaly",
        "calcification", "cyst", "enhancement", "infarct", "stroke",
    ]

    model_findings = {}

    for model_name, data in model_results.items():
        results = data.get("results", {})
        findings = set()

        # Search in various result fields
        text_to_search = ""
        for key in ["overall_findings", "summary", "report", "recommendations"]:
            if key in results and isinstance(results[key], str):
                text_to_search += " " + results[key].lower()

        # Also search in analysis dict
        if "analysis" in results:
            for value in results["analysis"].values():
                if isinstance(value, str):
                    text_to_search += " " + value.lower()

        # Find keywords
        for keyword in finding_keywords:
            if keyword in text_to_search:
                findings.add(keyword)

        model_findings[model_name] = findings

    # Find agreements (mentioned by 2+ models)
    all_findings = set()
    for findings in model_findings.values():
        all_findings.update(findings)

    agreements = []
    for finding in all_findings:
        count = sum(1 for f in model_findings.values() if finding in f)
        if count >= 2:
            agreements.append(f"{finding.title()} (mentioned by {count} models)")

    # Find unique findings
    unique = {}
    for model_name, findings in model_findings.items():
        other_findings = set()
        for other_model, other_f in model_findings.items():
            if other_model != model_name:
                other_findings.update(other_f)

        unique_to_model = findings - other_findings
        if unique_to_model:
            unique[model_name] = list(unique_to_model)

    return {"agreements": agreements, "unique": unique}


def _extract_classifier_results(model_results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Extract pathology probabilities from classifier models.
    """
    classifier_data = {}

    for model_name, data in model_results.items():
        results = data.get("results", {})

        # Check if this is classifier output
        if results.get("model_type") == "classifier":
            analysis = results.get("analysis", {})
            probs = analysis.get("pathology_probabilities", {})
            if probs:
                classifier_data[model_name] = probs

    return classifier_data


def save_comparison_report(
    series_path: str,
    model_results: Dict[str, Dict],
    output_dir: Path,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Generate and save a comparison report.

    Args:
        series_path: Path to the DICOM series
        model_results: Dict mapping model names to their results
        output_dir: Directory to save the comparison report
        metadata: Optional DICOM metadata

    Returns:
        Path to the saved comparison report
    """
    report = generate_comparison_report(series_path, model_results, metadata)

    # Save to _COMPARISON.md in the output directory
    comparison_path = output_dir / "_COMPARISON.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(report)

    return str(comparison_path)
