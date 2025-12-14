"""
Med3DVLM Model Implementation

Wraps the Med3DVLM model for 3D medical image VQA.
Based on paper: arXiv:2503.20047

Features:
- Input: 128x256x256 single-channel volumes
- Base LLM: Qwen 2.5-7B
- Vision encoder: DCFormer with decomposed 3D convolutions
- Supports: VQA, Report Generation, Image-Text Retrieval
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


@register_model("med3dvlm")
class Med3DVLMModel(Medical3DModel):
    """
    Med3DVLM model for 3D medical image VQA.

    Features:
    - Input: 128x256x256 single-channel volumes (4x depth vs M3D-LaMed)
    - Base LLM: Qwen 2.5-7B
    - Uses AutoProcessor for tokenization
    - Dual-stream MLP-Mixer projector for multimodal fusion
    """

    name = "med3dvlm"
    model_type = "vqa"

    @classmethod
    def get_default_model_path(cls) -> Path:
        """Return the default path to Med3DVLM model files."""
        return Path(__file__).parent.parent.parent / "models" / "Med3DVLM"

    def get_input_shape(self) -> Tuple[int, int, int]:
        """Return expected input shape (D, H, W)."""
        return (128, 256, 256)

    def get_channels(self) -> int:
        """Return number of input channels (1 for grayscale)."""
        return 1

    def preprocess_tensor(
        self,
        volume: np.ndarray,
        modality: str = "CT"
    ) -> torch.Tensor:
        """
        Preprocess volume for Med3DVLM input.

        Med3DVLM expects 128x256x256 volumes (NOT 32x256x256 like M3D-LaMed).
        This method handles:
        - Resizing to target dimensions
        - Normalization
        - Adding batch/channel dimensions
        - Shape validation
        """
        from ..preprocessing import DICOMPreprocessor
        from scipy.ndimage import zoom

        target_shape = self.get_input_shape()  # (D, H, W) = (128, 256, 256)

        # If string path, preprocess from DICOM
        if isinstance(volume, str):
            preprocessor = DICOMPreprocessor(
                target_spacing=(1.0, 1.0, 1.0),
                target_size=target_shape,
                modality=modality,
            )
            volume = preprocessor.process(volume)

        # Convert tensor to numpy for resizing if needed
        if isinstance(volume, torch.Tensor):
            volume = volume.float().cpu().numpy()  # Convert bfloat16 to float32 first

        # Ensure we have a numpy array at this point
        volume = np.asarray(volume, dtype=np.float32)

        # Handle different input shapes and resize to target
        # Expected final shape: (1, 1, D, H, W) = (1, 1, 128, 256, 256)
        if volume.ndim == 3:
            # (D, H, W) -> add channel and batch
            current_shape = volume.shape
        elif volume.ndim == 4:
            # (C, D, H, W) or (1, D, H, W)
            current_shape = volume.shape[1:]  # Get (D, H, W)
        elif volume.ndim == 5:
            # (N, C, D, H, W)
            current_shape = volume.shape[2:]  # Get (D, H, W)
        else:
            raise ValueError(f"Unexpected volume dimensions: {volume.ndim}")

        # Resize if dimensions don't match
        if current_shape != target_shape:
            # Calculate zoom factors for spatial dimensions only
            zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))

            if volume.ndim == 3:
                volume = zoom(volume, zoom_factors, order=1)
            elif volume.ndim == 4:
                # Resize each channel (typically just 1)
                resized = zoom(volume[0], zoom_factors, order=1)
                volume = resized[np.newaxis, ...]
            elif volume.ndim == 5:
                # Resize the spatial dimensions for each batch/channel
                resized = zoom(volume[0, 0], zoom_factors, order=1)
                volume = resized[np.newaxis, np.newaxis, ...]

        # Ensure proper shape: (1, 1, D, H, W)
        if volume.ndim == 3:
            volume = volume[np.newaxis, np.newaxis, ...]
        elif volume.ndim == 4:
            volume = volume[np.newaxis, ...]

        tensor = torch.from_numpy(volume.astype(np.float32)).to(
            dtype=torch.bfloat16, device=self.device
        )

        # Validate final shape - Med3DVLM REQUIRES 128 depth, not 32!
        expected_shape = (1, 1, 128, 256, 256)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Med3DVLM input shape error: got {tuple(tensor.shape)}, "
                f"expected {expected_shape}. "
                f"Note: Med3DVLM uses 128 depth slices, NOT 32 like M3D-LaMed!"
            )

        return tensor

    def load_model(self) -> None:
        """Load Med3DVLM model and tokenizer."""
        if self._loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers package required for Med3DVLM")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Med3DVLM model not found at {self.model_path}. "
                "Please download with: python scripts/download_model.py --model med3dvlm"
            )

        print(f"Loading Med3DVLM tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )

        print(f"Loading Med3DVLM model from {self.model_path}...")
        print(f"  Device: {self.device}")
        print(f"  Dtype: bfloat16")
        print(f"  Note: Model is ~33GB, loading may take a few minutes...")

        # Med3DVLM uses bfloat16 and has custom modeling code
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()
        self._loaded = True
        print("Med3DVLM model loaded successfully!")

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

        print("Med3DVLM model unloaded.")

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

        # Prepare image tensor (includes shape validation)
        image_tensor = self.preprocess_tensor(image, modality)

        # Tokenize the question
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # Med3DVLM may use different parameter names for image input
        # Try 'images' first (most likely), fall back to 'pixel_values'
        generation_kwargs = {
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.7,  # Lowered from 1.0 for more coherent output
            "do_sample": True,
            "top_p": 0.9,
        }

        with torch.no_grad():
            try:
                # Attempt 1: Use 'images' and 'inputs' parameters
                outputs = self.model.generate(
                    images=image_tensor,
                    inputs=inputs.input_ids,
                    **generation_kwargs,
                )
            except TypeError:
                try:
                    # Attempt 2: Use 'pixel_values' and 'input_ids' parameters
                    outputs = self.model.generate(
                        pixel_values=image_tensor,
                        input_ids=inputs.input_ids,
                        **generation_kwargs,
                    )
                except TypeError:
                    # Attempt 3: Use positional arguments
                    outputs = self.model.generate(
                        image_tensor,
                        inputs.input_ids,
                        **generation_kwargs,
                    )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input question from response if present
        if question in response:
            response = response.split(question)[-1]

        response = response.strip()

        # Check for common error indicators
        if response in ["?", "", "."]:
            # Return a more informative message instead of cryptic output
            return "(Model produced no meaningful response - possible input format issue)"

        return response

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
            "segmentations": None,  # Med3DVLM doesn't support segmentation
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
        """
        Generate a findings report for the image.

        Med3DVLM achieves 36.42% METEOR on report generation.
        """
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)

        prompt = "Generate a detailed radiology report describing the findings in this medical image."

        return self.generate_response(image_tensor, prompt, modality)
