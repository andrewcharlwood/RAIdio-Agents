"""
OpenRouter API Integration for Report Summarisation

Post-processing module to summarise M3D-LaMed analysis reports
using LLMs via OpenRouter API.
"""

import os
import json
from pathlib import Path
from typing import Optional

import requests


# Default OpenRouter settings
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"  # Good balance of quality/cost


def get_api_key() -> Optional[str]:
    """
    Get OpenRouter API key from environment variable.

    Returns:
        API key string or None if not set
    """
    return os.environ.get("OPENROUTER_API_KEY")


def summarise_reports(
    report_contents: list[dict],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
) -> dict:
    """
    Summarise multiple analysis reports into a consolidated, readable format.

    Args:
        report_contents: List of dicts with 'filename' and 'content' keys
        model: OpenRouter model ID to use
        api_key: OpenRouter API key (or uses OPENROUTER_API_KEY env var)
        custom_prompt: Optional custom system prompt for summarisation

    Returns:
        Dictionary with:
        - success: bool
        - summary: str (the summarised report)
        - model: str (model used)
        - error: str (if success=False)
    """
    api_key = api_key or get_api_key()

    if not api_key:
        return {
            "success": False,
            "summary": "",
            "model": model,
            "error": "No API key provided. Set OPENROUTER_API_KEY environment variable.",
        }

    if not report_contents:
        return {
            "success": False,
            "summary": "",
            "model": model,
            "error": "No reports provided to summarise.",
        }

    # Build the combined report text
    combined_text = []
    for report in report_contents:
        combined_text.append(f"=== {report['filename']} ===\n{report['content']}\n")

    reports_text = "\n".join(combined_text)

    # System prompt for medical report summarisation
    system_prompt = custom_prompt or """You are a medical imaging report summarisation assistant. Your task is to consolidate multiple AI-generated analysis reports from the same medical imaging study (CT or MRI scan) into a single, coherent summary.

Guidelines:
1. Synthesise findings from all series/sequences into one unified report
2. Identify consistent findings across multiple series (higher confidence)
3. Note any contradictory findings between series
4. Prioritise clinically significant findings
5. Use clear, professional medical terminology
6. Structure the output with clear sections
7. Highlight any urgent or critical findings prominently
8. Include appropriate caveats about AI-generated analysis

Output format:
- Executive Summary (2-3 sentences of key findings)
- Detailed Findings (organised by anatomical region or finding type)
- Areas of Concern (if any)
- Limitations & Caveats
- Recommended Follow-up (if applicable)

Remember: These are AI-generated preliminary analyses. Always emphasise that findings must be verified by qualified radiologists."""

    user_prompt = f"""Please summarise the following AI-generated medical imaging analysis reports from a single scan study. These reports cover different series/sequences from the same examination.

{reports_text}

Please provide a consolidated summary following the guidelines in your instructions."""

    # Make API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/AI_Diag",  # Required by OpenRouter
        "X-Title": "M3D-LaMed Report Summariser",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,  # Lower temperature for more consistent medical output
        "max_tokens": 4096,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120,  # 2 minute timeout for longer responses
        )
        response.raise_for_status()

        result = response.json()
        summary = result["choices"][0]["message"]["content"]

        return {
            "success": True,
            "summary": summary,
            "model": model,
            "usage": result.get("usage", {}),
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "summary": "",
            "model": model,
            "error": "Request timed out. Try again or use a faster model.",
        }
    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        try:
            error_detail = response.json().get("error", {}).get("message", str(e))
            error_msg = error_detail
        except Exception:
            pass
        return {
            "success": False,
            "summary": "",
            "model": model,
            "error": f"API error: {error_msg}",
        }
    except Exception as e:
        return {
            "success": False,
            "summary": "",
            "model": model,
            "error": f"Unexpected error: {str(e)}",
        }


def collect_reports_for_scan(output_dir: str, date: str, modality: str) -> list[dict]:
    """
    Collect all report files for a specific scan (date + modality combination).

    Args:
        output_dir: Base output directory (e.g., "Output")
        date: Study date in YYYYMMDD format
        modality: "CT" or "MRI"

    Returns:
        List of dicts with 'filename' and 'content' keys
    """
    scan_dir = Path(output_dir) / date / modality

    if not scan_dir.exists():
        return []

    reports = []
    for report_file in sorted(scan_dir.glob("*.md")):
        # Skip summary files
        if report_file.stem.startswith("_SUMMARY"):
            continue

        try:
            content = report_file.read_text(encoding="utf-8")
            reports.append({
                "filename": report_file.name,
                "content": content,
                "path": str(report_file),
            })
        except Exception as e:
            print(f"  Warning: Could not read {report_file}: {e}")

    return reports


def list_available_scans(output_dir: str) -> list[dict]:
    """
    List all available scans in the output directory.

    Args:
        output_dir: Base output directory

    Returns:
        List of dicts with 'date', 'modality', and 'report_count' keys
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return []

    scans = []
    for date_dir in sorted(output_path.iterdir()):
        if not date_dir.is_dir():
            continue
        # Check it looks like a date folder (8 digits)
        if not date_dir.name.isdigit() or len(date_dir.name) != 8:
            continue

        for modality_dir in sorted(date_dir.iterdir()):
            if not modality_dir.is_dir():
                continue
            if modality_dir.name not in ["CT", "MRI", "MR"]:
                continue

            # Count non-summary reports
            report_count = len([
                f for f in modality_dir.glob("*.md")
                if not f.stem.startswith("_SUMMARY")
            ])

            if report_count > 0:
                scans.append({
                    "date": date_dir.name,
                    "modality": modality_dir.name,
                    "report_count": report_count,
                    "path": str(modality_dir),
                })

    return scans


def format_summary_report(
    summary_result: dict,
    scan_info: dict,
    report_files: list[str],
) -> str:
    """
    Format the summarised output as a markdown report.

    Args:
        summary_result: Output from summarise_reports()
        scan_info: Dict with 'date' and 'modality'
        report_files: List of source report filenames

    Returns:
        Formatted markdown string
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = scan_info.get("date", "Unknown")

    # Format date nicely
    try:
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    except Exception:
        date_formatted = date_str

    lines = [
        "# Consolidated Scan Summary",
        "",
        f"**Generated**: {timestamp}",
        f"**Study Date**: {date_formatted}",
        f"**Modality**: {scan_info.get('modality', 'Unknown')}",
        f"**Model Used**: {summary_result.get('model', 'Unknown')}",
        f"**Source Reports**: {len(report_files)}",
        "",
        "---",
        "",
    ]

    if summary_result.get("success"):
        lines.extend([
            summary_result.get("summary", "No summary generated."),
            "",
        ])
    else:
        lines.extend([
            "## Error",
            "",
            f"Failed to generate summary: {summary_result.get('error', 'Unknown error')}",
            "",
        ])

    # Source reports section
    lines.extend([
        "---",
        "",
        "## Source Reports",
        "",
        "This summary was generated from the following analysis reports:",
        "",
    ])
    for filename in report_files:
        lines.append(f"- {filename}")
    lines.append("")

    # Disclaimer
    lines.extend([
        "---",
        "",
        "**Disclaimer**: This summary was generated by AI (M3D-LaMed + LLM summarisation) ",
        "for research purposes only. It should NOT be used as the sole basis for ",
        "clinical decision-making. Always verify findings with qualified radiologists ",
        "and correlate with clinical presentation.",
        "",
    ])

    return "\n".join(lines)


def summarise_scan(
    output_dir: str,
    date: str,
    modality: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    save_summary: bool = True,
) -> dict:
    """
    High-level function to summarise all reports for a specific scan.

    Args:
        output_dir: Base output directory
        date: Study date in YYYYMMDD format
        modality: "CT" or "MRI"
        model: OpenRouter model ID
        api_key: Optional API key
        save_summary: Whether to save the summary to a file

    Returns:
        Dictionary with summary results and metadata
    """
    # Collect reports
    reports = collect_reports_for_scan(output_dir, date, modality)

    if not reports:
        return {
            "success": False,
            "error": f"No reports found for {date}/{modality}",
            "summary_path": None,
        }

    print(f"  Found {len(reports)} reports to summarise")

    # Generate summary
    result = summarise_reports(reports, model=model, api_key=api_key)

    # Format and save
    scan_info = {"date": date, "modality": modality}
    report_files = [r["filename"] for r in reports]
    formatted = format_summary_report(result, scan_info, report_files)

    summary_path = None
    if save_summary and result.get("success"):
        summary_path = Path(output_dir) / date / modality / "_SUMMARY.md"
        summary_path.write_text(formatted, encoding="utf-8")
        print(f"  Summary saved to: {summary_path}")

    return {
        "success": result.get("success", False),
        "summary": result.get("summary", ""),
        "formatted": formatted,
        "error": result.get("error"),
        "summary_path": str(summary_path) if summary_path else None,
        "model": model,
        "source_reports": report_files,
    }
