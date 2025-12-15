"""
RadFM Model Implementation

Wraps the RadFM model for 3D medical image VQA.
Based on paper and repository: https://github.com/chaoyi-wu/RadFM

Features:
- Input: 3-channel RGB, flexible dimensions (typically 512x512xD)
- Base LLM: LLaMA
- Vision encoder: 3D ViT + Perceiver
- 32 tokens per image
- Supports: VQA, Report Generation, Multi-image analysis
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from . import register_model
from .base import Medical3DModel
from .radfm_helpers import get_tokenizer, format_prompt_with_image, preprocess_volume_for_radfm


# Default analysis questions
DEFAULT_ANALYSIS_QUESTIONS = [
    ("abnormalities", "Identify any abnormalities or pathological findings visible in this scan. Describe their location, appearance, and clinical significance."),
    ("key_findings", "What are the most clinically significant findings in this image? List them in order of importance."),
    ("differential", "Based on the findings in this scan, what differential diagnoses would you consider?"),
]


@register_model("radfm")
class RadFMModel(Medical3DModel):
    """
    RadFM model for 3D medical image VQA.

    Features:
    - Input: 3-channel (RGB), 512x512xD flexible dimensions
    - Base LLM: LLaMA
    - 3D ViT + Perceiver visual encoding
    - 32 tokens per image for multimodal fusion
    - Supports multi-image queries
    """

    name = "radfm"
    model_type = "vqa"

    # RadFM uses 32 tokens per image
    TOKENS_PER_IMAGE = 32

    @classmethod
    def get_default_model_path(cls) -> Path:
        """Return the default path to RadFM model files."""
        return Path(__file__).parent.parent.parent / "models" / "RadFM"

    @classmethod
    def get_external_radfm_path(cls) -> Path:
        """Return the path to external RadFM Quick_demo directory."""
        return Path(__file__).parent.parent.parent / "external" / "RadFM" / "Quick_demo"

    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Return expected input shape (D, H, W).

        RadFM is flexible on dimensions but typically uses 512x512.
        Depth is variable based on the scan.
        """
        return (32, 512, 512)  # Default, but flexible

    def get_channels(self) -> int:
        """Return number of input channels (3 for RGB)."""
        return 3

    def preprocess_tensor(
        self,
        volume: np.ndarray,
        modality: str = "CT"
    ) -> torch.Tensor:
        """
        Preprocess volume for RadFM input.

        RadFM expects (B, S, C, H, W, D) format where D is LAST.
        This method converts our standard (D, H, W) format to RadFM's format.
        """
        from ..preprocessing import DICOMPreprocessor

        # If string path, preprocess from DICOM
        if isinstance(volume, str):
            preprocessor = DICOMPreprocessor(
                target_spacing=(1.0, 1.0, 1.0),
                target_size=(32, 512, 512),  # (D, H, W)
                modality=modality,
            )
            volume = preprocessor.process(volume)

        # Convert to RadFM format (B, S, C, H, W, D)
        # Using helper function which handles dimension transformations
        tensor = preprocess_volume_for_radfm(
            volume,
            target_shape=(512, 512, 4),  # (H, W, D)
            modality=modality
        )

        return tensor.to(dtype=self.dtype, device=self.device)

    def load_model(self) -> None:
        """Load RadFM model and tokenizer."""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RadFM model not found at {self.model_path}. "
                "Please download from https://github.com/chaoyi-wu/RadFM"
            )

        print(f"Loading RadFM model from {self.model_path}...")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")

        # Add external RadFM directory to path to import custom model
        external_path = self.get_external_radfm_path()
        sys.path.insert(0, str(external_path))

        try:
            # Import RadFM's custom model class (capital M in Model)
            from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

            # Look for Language_files in model_path or external fallback
            lang_files = self.model_path / "Language_files"
            if not lang_files.exists():
                lang_files = external_path / "Language_files"
                if not lang_files.exists():
                    raise FileNotFoundError(
                        f"Language_files not found in {self.model_path} or {external_path}"
                    )

            print(f"  Language files: {lang_files}")

            # Load tokenizer with special image tokens using our helper
            self.tokenizer, self.image_padding_tokens = get_tokenizer(
                str(lang_files),
                max_img_size=100,
                image_num=32
            )

            # Initialize model
            self.model = MultiLLaMAForCausalLM(
                lang_model_path=str(lang_files)
            )

            # Load checkpoint weights if available
            ckpt_path = self.model_path / "pytorch_model.bin"
            if ckpt_path.exists():
                print(f"  Loading weights from {ckpt_path}")
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                self.model.load_state_dict(ckpt)
            else:
                print(f"  Warning: No weights found at {ckpt_path}, using random initialization")

            self.model = self.model.to(self.device)
            self.model.eval()

        except ImportError as e:
            print(f"Failed to import RadFM model: {e}")
            print("Make sure external/RadFM/Quick_demo exists with Model/RadFM/multimodality_model.py")
            raise

        finally:
            # Remove path from sys.path
            if str(external_path) in sys.path:
                sys.path.remove(str(external_path))

        self._loaded = True
        print("RadFM model loaded successfully!")

        mem = self.get_memory_usage()
        print(f"GPU Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['reserved_gb']:.2f}GB reserved")

    def unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if hasattr(self, 'image_padding_tokens'):
            del self.image_padding_tokens
            self.image_padding_tokens = None

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("RadFM model unloaded.")

    def generate_response(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        modality: str = "CT",
    ) -> str:
        """
        Generate a response for a question about an image.

        Args:
            image: DICOM path, preprocessed array, or tensor
            question: Question to ask about the image
            modality: "CT" or "MRI"

        Returns:
            Model's response string
        """
        self.ensure_loaded()

        # Prepare image tensor in RadFM format (B, S, C, H, W, D)
        vision_x = self.preprocess_tensor(image, modality)

        # Format prompt with image placeholder tokens using helper
        prompt = format_prompt_with_image(
            question,
            self.image_padding_tokens,
            image_index=0,
            position=0  # Prepend image before question
        )

        # Tokenize the prompt
        lang_x = self.tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        )['input_ids'].to(self.device)

        with torch.no_grad():
            # RadFM generate uses positional args: model.generate(lang_x, vision_x)
            generation = self.model.generate(lang_x, vision_x)

        # Decode response
        response = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

        # Remove the question from response if present
        if question in response:
            response = response.replace(question, "").strip()

        return response.strip()

    def analyse_scan(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
        analysis_questions: Optional[List[tuple]] = None,
        segment: bool = False,
    ) -> Dict:
        """
        Run comprehensive scan analysis with open-ended questions.
        """
        self.ensure_loaded()

        if analysis_questions is None:
            analysis_questions = DEFAULT_ANALYSIS_QUESTIONS

        # Prepare image tensor once
        image_tensor = self.preprocess_tensor(image, modality)

        results = {
            "model": self.name,
            "overall_findings": "",
            "analysis": {},
            "recommendations": "",
            "segmentations": None,  # RadFM doesn't support segmentation
        }

        # Overall findings
        print("Generating overall findings...")
        overall_question = (
            f"Please provide a comprehensive analysis of this {modality} scan. "
            "Describe the key anatomical structures visible, any abnormalities, "
            "and overall image quality."
        )
        results["overall_findings"] = self.generate_response(
            image_tensor, overall_question, modality
        )

        # Run each analysis question
        print("Running analysis...")
        for name, question in tqdm(analysis_questions, desc="Analysis"):
            full_question = f"{question} Be specific about location and confidence level."
            results["analysis"][name] = self.generate_response(
                image_tensor, full_question, modality
            )

        # Recommendations
        print("Generating recommendations...")
        rec_question = (
            f"Based on your analysis of this {modality} scan and any findings identified, "
            "what clinical recommendations would you suggest?"
        )
        results["recommendations"] = self.generate_response(
            image_tensor, rec_question, modality
        )

        return results

    def generate_report(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
    ) -> str:
        """Generate a findings report for the image."""
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)

        prompt = "Generate a detailed radiology report describing the findings in this medical image."

        return self.generate_response(image_tensor, prompt, modality)
