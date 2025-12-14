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

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from . import register_model
from .base import Medical3DModel


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

        RadFM expects 3-channel RGB input (H, W, D, 3).
        For grayscale medical images, we repeat across channels.
        """
        from ..preprocessing import DICOMPreprocessor

        # If string path, preprocess from DICOM
        if isinstance(volume, str):
            preprocessor = DICOMPreprocessor(
                target_spacing=(1.0, 1.0, 1.0),
                target_size=(32, 512, 512),  # RadFM typical size
                modality=modality,
            )
            volume = preprocessor.process(volume)

        # Already a tensor
        if isinstance(volume, torch.Tensor):
            tensor = volume.to(dtype=self.dtype, device=self.device)

            # RadFM needs 3 channels - repeat grayscale if needed
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1, 1)  # (3, D, H, W)
            elif tensor.ndim == 5 and tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1, 1)  # (B, 3, D, H, W)

            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)

            return tensor

        # Numpy array
        if volume.ndim == 3:
            # (D, H, W) -> (3, D, H, W)
            volume = np.stack([volume, volume, volume], axis=0)
        elif volume.ndim == 4 and volume.shape[0] == 1:
            # (1, D, H, W) -> (3, D, H, W)
            volume = np.concatenate([volume, volume, volume], axis=0)

        if volume.ndim == 4:
            volume = volume[np.newaxis, ...]

        return torch.from_numpy(volume).to(dtype=self.dtype, device=self.device)

    def load_model(self) -> None:
        """Load RadFM model and tokenizer."""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RadFM model not found at {self.model_path}. "
                "Please download from https://github.com/chaoyi-wu/RadFM"
            )

        # RadFM has a custom model class
        import sys
        sys.path.insert(0, str(self.model_path))

        try:
            # Try to import RadFM's custom model class
            from model.RadFM import MultiLLaMAForCausalLM
            from model.tokenizer import get_tokenizer

            print(f"Loading RadFM model from {self.model_path}...")
            print(f"  Device: {self.device}")
            print(f"  Dtype: {self.dtype}")

            # Load tokenizer with special image tokens
            self.tokenizer = get_tokenizer(str(self.model_path / "Language_files"))

            # Load model
            self.model = MultiLLaMAForCausalLM(
                lang_model_path=str(self.model_path / "Language_files")
            )

            # Load checkpoint
            ckpt_path = self.model_path / "pytorch_model.bin"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                self.model.load_state_dict(ckpt)

            self.model = self.model.to(self.device)
            self.model.eval()

        except ImportError:
            # Fallback: try using standard transformers if custom import fails
            print("Custom RadFM import failed, trying standard transformers...")
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

        finally:
            # Remove path from sys.path
            if str(self.model_path) in sys.path:
                sys.path.remove(str(self.model_path))

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

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("RadFM model unloaded.")

    def _format_prompt_with_image(self, question: str) -> str:
        """
        Format prompt with image placeholder tokens.

        RadFM uses <image></image> tags to mark image positions.
        """
        return f"<image></image>{question}"

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

        # Prepare image tensor
        image_tensor = self.preprocess_tensor(image, modality)

        # Format prompt with image markers
        prompt = self._format_prompt_with_image(question)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # RadFM generate
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                images=image_tensor,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        if prompt in response:
            response = response.replace(prompt, "")

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
