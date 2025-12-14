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

    def __init__(self, **kwargs):
        """Initialize CT-CLIP with float32 dtype (required by model architecture)."""
        # CT-CLIP requires float32 - override any dtype setting
        kwargs['dtype'] = torch.float32
        super().__init__(**kwargs)

    @classmethod
    def get_default_model_path(cls) -> Path:
        """Return the default path to CT-CLIP model files."""
        return Path(__file__).parent.parent.parent / "models" / "CT-CLIP"

    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Return expected input shape (D, H, W).

        CT-CLIP uses CTViT with image_size=480 and expects 240 depth slices.
        """
        return (240, 480, 480)

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

        CT-CLIP expects:
        - Shape: (B, C, D, H, W) = (1, 1, 240, 480, 480)
        - HU values clipped to [-1000, 1000] and normalized by /1000
        """
        from scipy.ndimage import zoom

        target_shape = self.get_input_shape()  # (240, 480, 480)

        # Handle string path
        if isinstance(volume, str):
            from ..preprocessing import DICOMPreprocessor
            preprocessor = DICOMPreprocessor(
                target_spacing=(1.5, 0.75, 0.75),  # CT-CLIP default spacing
                target_size=None,  # Don't resize yet, we'll do custom resize
                modality="CT",
            )
            volume = preprocessor.process(volume)

        # Convert tensor to numpy
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()

        volume = np.asarray(volume, dtype=np.float32)

        # Remove batch/channel dimensions to get (D, H, W)
        while volume.ndim > 3:
            volume = volume[0]

        # Ensure HU range [-1000, 1000] and normalize
        # If already normalized to [0,1], denormalize first
        if volume.min() >= 0 and volume.max() <= 1:
            # Assume it was normalized with soft tissue window
            # Convert back to approximate HU range for CT-CLIP normalization
            volume = volume * 2000 - 1000  # Map [0,1] to [-1000, 1000]

        volume = np.clip(volume, -1000, 1000)
        volume = volume / 1000.0  # CT-CLIP normalization

        # Resize to target shape (240, 480, 480)
        current_shape = volume.shape
        if current_shape != target_shape:
            zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
            volume = zoom(volume, zoom_factors, order=1)

        # Add channel and batch dimensions: (D, H, W) -> (1, 1, D, H, W)
        volume = volume[np.newaxis, np.newaxis, ...]

        return torch.from_numpy(volume.astype(np.float32)).to(
            dtype=self.dtype, device=self.device
        )

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
            # Import required modules
            from ct_clip import CTCLIP
            from transformer_maskgit import CTViT
            from transformers import BertTokenizer, BertModel

            print(f"Loading CT-CLIP model from {self.model_path}...")
            print(f"  Device: {self.device}")
            print(f"  Dtype: {self.dtype}")

            # Check for model checkpoint file
            ckpt_path = self.model_path / "CT-CLIP_v2.pt"
            if not ckpt_path.exists():
                # Check alternate locations
                alt_paths = [
                    self.model_path / "models" / "CT-CLIP-Related" / "CT-CLIP_v2.pt",
                    self.model_path / "CT_CLIP_v2.pt",
                ]
                for alt in alt_paths:
                    if alt.exists():
                        ckpt_path = alt
                        break
                else:
                    raise FileNotFoundError(
                        f"CT-CLIP checkpoint not found at {self.model_path}.\n"
                        "Please download from HuggingFace (requires authentication):\n"
                        "  1. Request access at: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE\n"
                        "  2. Login: huggingface-cli login\n"
                        "  3. Download: huggingface-cli download ibrahimhamamci/CT-RATE "
                        "models/CT-CLIP-Related/CT-CLIP_v2.pt --repo-type dataset "
                        f"--local-dir {self.model_path}"
                    )

            # Initialize tokenizer for text encoding
            print("  Loading BiomedVLP-CXR-BERT tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained(
                'microsoft/BiomedVLP-CXR-BERT-specialized',
                do_lower_case=True
            )

            # Initialize text encoder (BERT)
            print("  Loading BiomedVLP-CXR-BERT text encoder...")
            text_encoder = BertModel.from_pretrained(
                "microsoft/BiomedVLP-CXR-BERT-specialized"
            )
            text_encoder.resize_token_embeddings(len(self.tokenizer))

            # Initialize image encoder (CTViT)
            print("  Initializing CTViT image encoder...")
            image_encoder = CTViT(
                dim=512,
                codebook_size=8192,
                image_size=480,
                patch_size=20,
                temporal_patch_size=10,
                spatial_depth=4,
                temporal_depth=4,
                dim_head=32,
                heads=8
            )

            # Initialize CT-CLIP with correct dimensions
            print("  Initializing CT-CLIP model...")
            self.model = CTCLIP(
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                dim_image=294912,
                dim_text=768,
                dim_latent=512,
                extra_latent_projection=False,
                use_mlm=False,
                downsample_image_embeds=False,
                use_all_token_embeds=False
            )

            # Load pretrained weights
            print(f"  Loading checkpoint from {ckpt_path}...")
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
            # Use strict=False to handle position_ids buffer mismatch
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            if missing:
                print(f"  Note: {len(missing)} missing keys (expected for new components)")
            if unexpected:
                print(f"  Note: {len(unexpected)} unexpected keys (expected for buffer changes)")
            self.model = self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            print("CT-CLIP model loaded successfully!")

        except ImportError as e:
            raise ImportError(
                f"CT-CLIP package or dependency not found: {e}\n"
                "Install with:\n"
                "  cd external/CT-CLIP/transformer_maskgit && pip install -e .\n"
                "  cd external/CT-CLIP/CT_CLIP && pip install -e .\n"
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

        CT-CLIP uses zero-shot classification by comparing text prompts:
        "X is present" vs "X is not present" for each pathology.

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

        # CT-CLIP pathology names (more readable versions)
        pathology_names = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
            'Pericardial effusion', 'Coronary artery wall calcification',
            'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
            'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
            'Pleural effusion', 'Mosaic attenuation pattern',
            'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
            'Interlobular septal thickening'
        ]

        results = {}
        softmax = torch.nn.Softmax(dim=0)

        with torch.no_grad():
            for i, pathology_name in enumerate(pathology_names):
                # Create text prompts for zero-shot classification
                text = [f"{pathology_name} is present.", f"{pathology_name} is not present."]
                text_tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Run model forward pass
                output = self.model(text_tokens, image_tensor, device=self.device)

                # Apply softmax to get probability of "is present"
                probs = softmax(output)
                prob_present = float(probs[0].cpu().numpy())

                # Map to our standard pathology names
                std_name = CT_CLIP_PATHOLOGIES[i] if i < len(CT_CLIP_PATHOLOGIES) else pathology_name.lower().replace(' ', '_')
                results[std_name] = prob_present

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
