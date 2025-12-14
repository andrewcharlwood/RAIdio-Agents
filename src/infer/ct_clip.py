"""
CT-CLIP Model Implementation

Wraps the CT-CLIP model for 3D chest CT pathology classification.
Based on repository: https://github.com/ibrahimethemhamamci/CT-CLIP

Features:
- Input: 3D chest CT volumes
- Classification: 18 pathologies
- Zero-shot and fine-tuned variants available
- Fast inference (~0.5-1.5 seconds per volume)

NOTE: This is a CLASSIFIER model, not a VQA model.
It outputs pathology probabilities, not text responses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from . import register_model
from .base import Medical3DModel


# CT-CLIP pathology labels
CT_CLIP_PATHOLOGIES = [
    "lung_opacity",
    "pleural_effusion",
    "atelectasis",
    "consolidation",
    "pneumothorax",
    "emphysema",
    "pulmonary_fibrosis",
    "nodule",
    "mass",
    "cardiomegaly",
    "pericardial_effusion",
    "lymphadenopathy",
    "pulmonary_edema",
    "bronchiectasis",
    "interstitial_lung_disease",
    "lung_cancer",
    "pleural_thickening",
    "calcification",
]


@register_model("ct-clip")
class CTCLIPModel(Medical3DModel):
    """
    CT-CLIP model for 3D chest CT pathology classification.

    This is a CLASSIFIER model - it outputs pathology probabilities,
    NOT text responses like VQA models.

    Features:
    - Classifies 18 chest pathologies
    - Zero-shot capability via CLIP-style architecture
    - Fast inference: 0.5-1.5 seconds per volume

    Use classify() method instead of generate_response().
    """

    name = "ct-clip"
    model_type = "classifier"

    @classmethod
    def get_default_model_path(cls) -> Path:
        """Return the default path to CT-CLIP model files."""
        return Path(__file__).parent.parent.parent / "models" / "CT-CLIP"

    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Return expected input shape (D, H, W).

        CT-CLIP input size is not strictly documented.
        Using common 3D medical imaging dimensions.
        """
        return (64, 256, 256)

    def get_channels(self) -> int:
        """Return number of input channels (1 for grayscale)."""
        return 1

    def preprocess_tensor(
        self,
        volume: np.ndarray,
        modality: str = "CT"
    ) -> torch.Tensor:
        """
        Preprocess volume for CT-CLIP input.

        CT-CLIP is designed for chest CT scans.
        """
        from ..preprocessing import DICOMPreprocessor

        # If string path, preprocess from DICOM
        if isinstance(volume, str):
            preprocessor = DICOMPreprocessor(
                target_spacing=(1.0, 1.0, 1.0),
                target_size=self.get_input_shape(),
                modality="CT",  # CT-CLIP is CT-only
            )
            volume = preprocessor.process(volume)

        # Already a tensor
        if isinstance(volume, torch.Tensor):
            tensor = volume.to(dtype=self.dtype, device=self.device)
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
            return tensor

        # Numpy array
        if volume.ndim == 4:
            volume = volume[np.newaxis, ...]

        return torch.from_numpy(volume).to(dtype=self.dtype, device=self.device)

    def load_model(self) -> None:
        """Load CT-CLIP model."""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"CT-CLIP model not found at {self.model_path}. "
                "Please download from https://github.com/ibrahimethemhamamci/CT-CLIP\n"
                "Installation: pip install CT_CLIP transformer_maskgit"
            )

        try:
            # Try to import CT-CLIP
            from CT_CLIP import CTCLIP

            print(f"Loading CT-CLIP model from {self.model_path}...")
            print(f"  Device: {self.device}")
            print(f"  Dtype: {self.dtype}")

            # Load model
            self.model = CTCLIP.from_pretrained(str(self.model_path))
            self.model = self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            print("CT-CLIP model loaded successfully!")

        except ImportError:
            raise ImportError(
                "CT-CLIP package not found. Install with:\n"
                "  cd CT_CLIP && pip install -e .\n"
                "See: https://github.com/ibrahimethemhamamci/CT-CLIP"
            )

        mem = self.get_memory_usage()
        print(f"GPU Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['reserved_gb']:.2f}GB reserved")

    def unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("CT-CLIP model unloaded.")

    def classify(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
    ) -> Dict[str, float]:
        """
        Classify image and return pathology probabilities.

        Args:
            image: DICOM path, preprocessed array, or tensor
            modality: Should be "CT" for CT-CLIP

        Returns:
            Dictionary mapping pathology names to probabilities (0.0-1.0)
        """
        if modality != "CT":
            print(f"Warning: CT-CLIP is designed for CT scans, not {modality}")

        self.ensure_loaded()

        # Prepare image tensor
        image_tensor = self.preprocess_tensor(image, "CT")

        with torch.no_grad():
            # Get model predictions
            outputs = self.model(image_tensor)

            # Convert logits to probabilities
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("predictions", None))
            else:
                logits = outputs

            if logits is None:
                raise ValueError("Could not extract predictions from CT-CLIP output")

            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Map to pathology names
        results = {}
        for i, pathology in enumerate(CT_CLIP_PATHOLOGIES):
            if i < len(probs):
                results[pathology] = float(probs[i])

        return results

    def analyse_scan(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
        analysis_questions: Optional[List[tuple]] = None,
        segment: bool = False,
    ) -> Dict:
        """
        Run pathology classification on the scan.

        For CT-CLIP (a classifier), this returns classification results
        formatted similarly to VQA model outputs for compatibility.
        """
        if modality != "CT":
            print(f"Warning: CT-CLIP is designed for CT scans, not {modality}")

        # Get classifications
        pathology_probs = self.classify(image, "CT")

        # Find significant findings (probability > 0.5)
        positive_findings = {
            k: v for k, v in pathology_probs.items() if v > 0.5
        }
        uncertain_findings = {
            k: v for k, v in pathology_probs.items() if 0.3 <= v <= 0.5
        }

        # Format as analysis results
        results = {
            "model": self.name,
            "model_type": "classifier",
            "overall_findings": self._format_classification_summary(pathology_probs),
            "analysis": {
                "pathology_probabilities": pathology_probs,
                "positive_findings": positive_findings,
                "uncertain_findings": uncertain_findings,
            },
            "recommendations": self._generate_recommendations(positive_findings),
            "segmentations": None,
        }

        return results

    def _format_classification_summary(self, probs: Dict[str, float]) -> str:
        """Format classification probabilities as a summary."""
        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])

        lines = ["CT-CLIP Classification Results:"]
        lines.append("")

        # High probability findings
        high = [(k, v) for k, v in sorted_probs if v > 0.5]
        if high:
            lines.append("POSITIVE FINDINGS (>50% probability):")
            for pathology, prob in high:
                lines.append(f"  - {pathology.replace('_', ' ').title()}: {prob*100:.1f}%")
            lines.append("")

        # Uncertain findings
        uncertain = [(k, v) for k, v in sorted_probs if 0.3 <= v <= 0.5]
        if uncertain:
            lines.append("UNCERTAIN FINDINGS (30-50% probability):")
            for pathology, prob in uncertain:
                lines.append(f"  - {pathology.replace('_', ' ').title()}: {prob*100:.1f}%")
            lines.append("")

        # Negative findings (low probability)
        if not high and not uncertain:
            lines.append("No significant pathologies detected (all <30% probability)")

        return "\n".join(lines)

    def _generate_recommendations(self, positive_findings: Dict[str, float]) -> str:
        """Generate recommendations based on positive findings."""
        if not positive_findings:
            return "No significant findings detected. Routine follow-up as clinically indicated."

        lines = ["Based on positive findings:"]

        # Map pathologies to recommendations
        rec_map = {
            "lung_cancer": "Urgent oncology referral recommended. Consider PET-CT staging.",
            "mass": "Further characterization with contrast CT or PET-CT recommended.",
            "nodule": "Follow LUNG-RADS guidelines for nodule management.",
            "pneumothorax": "Clinical correlation required. Consider chest tube if large.",
            "pleural_effusion": "Consider thoracentesis if symptomatic.",
            "pulmonary_edema": "Evaluate cardiac function. Consider echocardiogram.",
            "cardiomegaly": "Recommend echocardiogram for cardiac evaluation.",
            "emphysema": "Pulmonary function tests recommended.",
            "pulmonary_fibrosis": "Recommend pulmonology consultation.",
            "interstitial_lung_disease": "Recommend pulmonology consultation.",
        }

        for pathology in positive_findings:
            if pathology in rec_map:
                lines.append(f"  - {pathology.replace('_', ' ').title()}: {rec_map[pathology]}")

        if len(lines) == 1:
            lines.append("  - Clinical correlation recommended for detected findings.")

        return "\n".join(lines)

    # Override VQA methods to raise helpful errors
    def generate_response(self, *args, **kwargs) -> str:
        """CT-CLIP is a classifier. Use classify() instead."""
        raise NotImplementedError(
            "CT-CLIP is a classifier model, not a VQA model. "
            "Use classify() to get pathology probabilities, or "
            "use analyse_scan() for a formatted report."
        )

    def generate_with_segmentation(self, *args, **kwargs):
        """CT-CLIP doesn't support segmentation."""
        raise NotImplementedError(
            "CT-CLIP does not support segmentation. "
            "Use classify() for pathology detection."
        )
