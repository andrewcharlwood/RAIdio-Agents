"""
Model Inference Wrapper for M3D-LaMed

This module provides backwards-compatible access to the M3D-LaMed model
while supporting the new multi-model architecture.

For new code, prefer using the infer module directly:
    from src.infer import get_model, list_models
    model = get_model("m3d-lamed")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Import the new model system
from .infer import get_model, list_models

# Patient context support
from .patient_context import (
    PatientContext,
    load_patient_context,
    enhance_question_with_context,
    build_clinical_context_string,
)
from .infer.m3d_lamed import M3DLaMedModel, DEFAULT_ANALYSIS_QUESTIONS

# Re-export for backwards compatibility
from .preprocessing import preprocess_for_m3d
from .prompts import (
    SYSTEM_PROMPT_GENERAL,
    SYSTEM_PROMPT_RADIOLOGIST,
    SYSTEM_PROMPT_SCREENING,
    get_analysis_chain,
    get_quick_analysis_chain,
    get_pathology_screen,
    format_closed_question,
    REPORT_GENERATION_PROMPTS,
    PLANE_QUESTIONS,
    PHASE_QUESTIONS,
    ABNORMALITY_QUESTIONS,
)

# Default paths (for backwards compatibility)
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "M3D-LaMed-Phi-3-4B"
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_GENERAL


class M3DLaMedInference:
    """
    Backwards-compatible wrapper for M3D-LaMed model.

    This class wraps the new M3DLaMedModel to maintain compatibility
    with existing code. For new code, consider using the infer module:

        from src.infer import get_model
        model = get_model("m3d-lamed")
        model.load_model()
        response = model.generate_response(image, question, modality)

    All existing functionality is preserved.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 512,
    ):
        """
        Initialise the inference wrapper.

        Args:
            model_path: Path to model directory (defaults to models/M3D-LaMed-Phi-3-4B)
            device: Device to run on ("cuda" works with ROCm)
            dtype: Model dtype (float16 recommended for efficiency)
            max_new_tokens: Maximum tokens to generate
        """
        # Use the new model implementation
        self._model = M3DLaMedModel(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

        # Expose internal attributes for backwards compatibility
        self.model_path = self._model.model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

    @property
    def model(self):
        """Access the underlying HuggingFace model."""
        return self._model.model

    @property
    def tokenizer(self):
        """Access the tokenizer."""
        return self._model.tokenizer

    @property
    def _loaded(self):
        """Check if model is loaded."""
        return self._model._loaded

    def load_model(self):
        """Load the model and tokenizer."""
        self._model.load_model()

    def unload_model(self):
        """Unload model to free GPU memory."""
        self._model.unload_model()

    def prepare_image_tensor(
        self,
        image: Union[str, np.ndarray],
        modality: str = "CT"
    ) -> torch.Tensor:
        """
        Prepare image tensor for model input.

        Args:
            image: Either path to DICOM directory or preprocessed numpy array
            modality: "CT" or "MRI" (used if preprocessing needed)

        Returns:
            Tensor with shape (B, C, D, H, W)
        """
        return self._model.preprocess_tensor(image, modality)

    def generate_response(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        modality: str = "CT",
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response for a question about an image.

        Args:
            image: DICOM path, preprocessed array, or tensor
            question: Question to ask about the image
            modality: "CT" or "MRI"
            system_prompt: Unused (kept for API compatibility)

        Returns:
            Model's response string
        """
        return self._model.generate_response(image, question, modality)

    def generate_with_segmentation(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        modality: str = "CT",
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate a response with optional segmentation mask.

        Args:
            image: DICOM path, preprocessed array, or tensor
            question: Question to ask about the image
            modality: "CT" or "MRI"

        Returns:
            Tuple of (response_text, segmentation_mask or None)
        """
        return self._model.generate_with_segmentation(image, question, modality)

    def analyse_scan(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
        analysis_questions: Optional[List[tuple]] = None,
        segment: bool = False,
    ) -> Dict:
        """
        Run comprehensive scan analysis with open-ended questions.

        Args:
            image: DICOM path, preprocessed array, or tensor
            modality: "CT" or "MRI"
            analysis_questions: List of (name, question) tuples
            segment: If True, generate segmentation masks

        Returns:
            Dictionary with analysis results
        """
        return self._model.analyse_scan(image, modality, analysis_questions, segment)

    def analyse_chained(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
        quick: bool = False,
    ) -> Dict:
        """
        Run chained query analysis following paper recommendations.

        Args:
            image: DICOM path, preprocessed array, or tensor
            modality: "CT" or "MRI"
            quick: Use shortened analysis chain

        Returns:
            Dictionary with chained analysis results
        """
        return self._model.analyse_chained(image, modality, quick)

    def ask_closed_question(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        choices: List[str],
        modality: str = "CT",
    ) -> Dict:
        """
        Ask a closed-ended (multiple choice) question.

        Args:
            image: DICOM path, preprocessed array, or tensor
            question: The question to ask
            choices: List of choices
            modality: "CT" or "MRI"

        Returns:
            Dictionary with question, choices, and model's answer
        """
        return self._model.ask_closed_question(image, question, choices, modality)

    def screen_pathology(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        region: str = "head",
        modality: str = "CT",
        pathologies: Optional[List[str]] = None,
    ) -> Dict:
        """
        Screen for specific pathologies using closed-ended questions.

        Args:
            image: DICOM path, preprocessed array, or tensor
            region: Body region ("head", "chest", "abdomen")
            modality: "CT" or "MRI"
            pathologies: Specific pathologies to screen

        Returns:
            Dictionary with screening results
        """
        return self._model.screen_pathology(image, region, modality, pathologies)

    def generate_report(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
    ) -> str:
        """
        Generate a findings report for the image.

        Args:
            image: DICOM path, preprocessed array, or tensor
            modality: "CT" or "MRI"

        Returns:
            Generated report text
        """
        return self._model.generate_report(image, modality)

    def comprehensive_analysis(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
        region: str = "head",
        include_pathology_screen: bool = True,
        segment: bool = False,
    ) -> Dict:
        """
        Run comprehensive analysis combining multiple approaches.

        Args:
            image: DICOM path, preprocessed array, or tensor
            modality: "CT" or "MRI"
            region: Body region for pathology screening
            include_pathology_screen: Whether to include pathology screening
            segment: If True, generate segmentation masks

        Returns:
            Comprehensive analysis dictionary
        """
        return self._model.comprehensive_analysis(
            image, modality, region, include_pathology_screen, segment
        )


# ============================================================================
# Multi-model support utilities
# ============================================================================

def get_available_models() -> List[str]:
    """
    Get list of available model names.

    Returns:
        List of registered model names (e.g., ["m3d-lamed", "med3dvlm", ...])
    """
    return list_models()


def load_model(
    name: str = "m3d-lamed",
    model_path: Optional[str] = None,
    **kwargs
):
    """
    Load a model by name.

    Args:
        name: Model name (e.g., "m3d-lamed", "med3dvlm", "radfm", "ct-clip")
        model_path: Optional path to model files
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Model instance (Medical3DModel subclass)

    Example:
        model = load_model("m3d-lamed")
        model.load_model()
        response = model.generate_response(image, "What abnormalities are visible?")
    """
    return get_model(name, model_path=model_path, **kwargs)
