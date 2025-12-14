"""
Abstract Base Class for 3D Medical Image Models

Provides a unified interface for different 3D medical VLM models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch


class Medical3DModel(ABC):
    """
    Abstract base class for 3D medical image models.

    All model implementations (M3D-LaMed, Med3DVLM, RadFM, CT-CLIP)
    should inherit from this class and implement the abstract methods.
    """

    # Class attributes to be set by subclasses
    name: str = "base"
    model_type: Literal["vqa", "classifier"] = "vqa"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 512,
    ):
        """
        Initialise the model wrapper.

        Args:
            model_path: Path to model directory/checkpoint
            device: Device to run on ("cuda" works with ROCm)
            dtype: Model dtype (float16 recommended)
            max_new_tokens: Maximum tokens to generate (VQA models only)
        """
        self.model_path = Path(model_path) if model_path else self.get_default_model_path()
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None
        self._loaded = False

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @classmethod
    @abstractmethod
    def get_default_model_path(cls) -> Path:
        """Return the default path to the model files."""
        pass

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Return the expected input tensor shape (D, H, W).

        Returns:
            Tuple of (depth, height, width) expected by the model
        """
        pass

    @abstractmethod
    def get_channels(self) -> int:
        """
        Return the number of input channels expected.

        Returns:
            Number of channels (1 for grayscale, 3 for RGB)
        """
        pass

    @abstractmethod
    def preprocess_tensor(
        self,
        volume: np.ndarray,
        modality: str = "CT"
    ) -> torch.Tensor:
        """
        Preprocess a volume array for this specific model.

        Takes a raw DICOM volume (after DICOM loading and resampling) and
        applies model-specific preprocessing (windowing, normalization,
        channel expansion, etc.)

        Args:
            volume: Raw volume array (D, H, W) or (C, D, H, W)
            modality: "CT" or "MRI"

        Returns:
            Tensor ready for model input with shape (B, C, D, H, W)
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and tokenizer into memory.

        Should set self._loaded = True when complete.
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload model from memory and clear GPU cache.

        Should set self._loaded = False when complete.
        """
        pass

    # =========================================================================
    # VQA model methods - implement for VQA models, raise NotImplementedError for classifiers
    # =========================================================================

    def generate_response(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        modality: str = "CT",
    ) -> str:
        """
        Generate a text response for a question about an image.

        Args:
            image: DICOM path, preprocessed array, or tensor
            question: Question to ask about the image
            modality: "CT" or "MRI"

        Returns:
            Model's response string

        Raises:
            NotImplementedError: If model is a classifier
        """
        if self.model_type == "classifier":
            raise NotImplementedError(
                f"{self.name} is a classifier model and does not support generate_response(). "
                "Use classify() instead."
            )
        raise NotImplementedError("Subclass must implement generate_response()")

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
        # Default: no segmentation support
        return self.generate_response(image, question, modality), None

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
            segment: If True, generate segmentation masks (if supported)

        Returns:
            Dictionary with analysis results
        """
        raise NotImplementedError("Subclass must implement analyse_scan()")

    # =========================================================================
    # Classifier model methods - implement for classifiers, raise NotImplementedError for VQA
    # =========================================================================

    def classify(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
    ) -> Dict[str, float]:
        """
        Classify image and return pathology probabilities.

        Args:
            image: DICOM path, preprocessed array, or tensor
            modality: "CT" or "MRI"

        Returns:
            Dictionary mapping pathology names to probabilities

        Raises:
            NotImplementedError: If model is a VQA model
        """
        if self.model_type == "vqa":
            raise NotImplementedError(
                f"{self.name} is a VQA model and does not support classify(). "
                "Use generate_response() instead."
            )
        raise NotImplementedError("Subclass must implement classify()")

    # =========================================================================
    # Common utility methods
    # =========================================================================

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if not self._loaded:
            self.load_model()

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dict with 'allocated_gb' and 'reserved_gb' keys
        """
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            }
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.model_type}', loaded={self._loaded})"
