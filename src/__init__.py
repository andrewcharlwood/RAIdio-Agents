"""
M3D-LaMed Project - Medical 3D Image Analysis

This package provides tools for analysing head/neck CT and MRI scans
using the M3D-LaMed-Phi-3-4B model.
"""

from .preprocessing import DICOMPreprocessor, preprocess_for_m3d
from .inference import M3DLaMedInference
from .utils import find_dicom_series, get_dicom_metadata, format_analysis_report

__version__ = "0.1.0"
__all__ = [
    "DICOMPreprocessor",
    "preprocess_for_m3d",
    "M3DLaMedInference",
    "find_dicom_series",
    "get_dicom_metadata",
    "format_analysis_report",
]
