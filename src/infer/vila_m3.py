"""
VILA-M3 Model Implementation

Wraps the MONAI VILA-M3 vision-language model framework.
Supports VQA with expert model integration (VISTA3D, CXR, BRATS).

Features:
- Input: 2D images or slices from 3D volumes (.nii.gz)
- Base LLM: Llama-3 (8B) or Vicuna (3B, 13B)
- Expert models: VISTA3D (3D CT segmentation), CXR (chest X-ray), BRATS (brain tumors)
- Supports: VQA, Report Generation, Segmentation via experts
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm


# MODEL_CARDS: Expert model descriptions (required for proper model behavior)
# The VILA-M3 model was trained with this context and expects it in the prompt
MODEL_CARDS = """Here is a list of available expert models:
<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None
<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'
<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None
<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None
Give the model <NAME(args)> when selecting a suitable expert model.
"""

# Patterns for cleaning VISTA3D/expert model output leakage
VISTA3D_PATTERNS = [
    r'identified by VISTA3D:\s*[^.]*\.',  # "identified by VISTA3D: red: liver, ..."
    r'as identified by VISTA3D[^.]*\.',
    r'VISTA3D identifies[^.]*\.',
    r'VISTA3D segmentation[^.]*\.',
    r'(?:red|blue|green|yellow|orange|purple|pink|gray|grey|brown|white):\s*\w+(?:\s+\w+)*,?\s*',  # color labels
]

# Modality confusion patterns
MODALITY_PATTERNS = {
    "MRI": [r'\bCT image\b', r'\bCT scan\b', r'\bcomputed tomography\b'],
    "CT": [r'\bMRI image\b', r'\bMRI scan\b', r'\bmagnetic resonance\b'],
}

from . import register_model
from .base import Medical3DModel


# Default analysis questions
DEFAULT_ANALYSIS_QUESTIONS = [
    ("description", "Describe this medical image in detail. What anatomical structures are visible?"),
    ("abnormalities", "Identify any abnormalities or pathological findings visible in this scan."),
    ("key_findings", "What are the most clinically significant findings in this image?"),
    ("differential", "Based on the findings, what differential diagnoses would you consider?"),
]


def _setup_vila_path():
    """Add VILA framework to Python path."""
    project_root = Path(__file__).parent.parent.parent
    vila_root = project_root / "external" / "vila-m3"
    vila_thirdparty = vila_root / "thirdparty" / "VILA"
    demo_dir = vila_root / "m3" / "demo"

    paths_to_add = [
        str(vila_root),
        str(vila_thirdparty),
        str(demo_dir),
    ]

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)

    return vila_root, demo_dir


@register_model("vila-m3")
@register_model("vila-m3-8b")
class VILAm3Model(Medical3DModel):
    """
    VILA-M3 model for medical image VQA with expert integration.

    Features:
    - Works with 2D images and slices from 3D volumes
    - Expert model support: VISTA3D, CXR analysis, BRATS
    - Based on Llama-3 (8B) or Vicuna (3B, 13B)
    """

    name = "vila-m3"
    model_type = "vqa"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 1024,
        model_size: str = "8b",  # "3b", "8b", or "13b"
        enable_experts: bool = True,
    ):
        """
        Initialize VILA-M3 model.

        Args:
            model_path: Path to local model or HuggingFace ID
            device: Device to run on
            dtype: Model dtype
            max_new_tokens: Maximum tokens to generate
            model_size: Model size ("3b", "8b", or "13b")
            enable_experts: Enable expert model integration
        """
        # Set model size and conv_mode before calling parent init
        self.model_size = model_size.lower()
        self.conv_mode = "llama_3" if self.model_size == "8b" else "vicuna_v1"
        self.enable_experts = enable_experts

        # Model path: use HuggingFace ID if not specified
        if model_path is None:
            model_path = f"MONAI/Llama3-VILA-M3-{self.model_size.upper()}"

        super().__init__(
            model_path=str(model_path),
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

        self.generator = None
        self._vila_root = None
        self._demo_dir = None
        self._temp_dir = None

    @classmethod
    def get_default_model_path(cls) -> Path:
        """Return the default HuggingFace model ID."""
        return Path("MONAI/Llama3-VILA-M3-8B")

    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        VILA-M3 works with 2D slices, not full 3D tensors.
        Returns a nominal shape for compatibility.
        """
        return (1, 512, 512)  # Single slice

    def get_channels(self) -> int:
        """Return number of input channels."""
        return 3  # RGB images

    def preprocess_tensor(
        self,
        volume: np.ndarray,
        modality: str = "CT"
    ) -> torch.Tensor:
        """
        VILA-M3 uses its own image processing pipeline.
        This method is not used directly - images are passed as file paths.
        """
        raise NotImplementedError(
            "VILA-M3 uses file-based image processing. "
            "Pass image paths directly to generate_response()."
        )

    def _convert_dicom_to_slice_image(
        self,
        dicom_path: str,
        modality: str = "CT",
        slice_index: Optional[int] = None,
    ) -> str:
        """
        Convert DICOM series to a 2D slice image (JPG) for VILA-M3.

        VILA-M3 uses PIL for image loading, which only supports 2D formats.
        This method extracts a representative slice from the DICOM volume,
        applies appropriate windowing, and saves as JPG.

        Args:
            dicom_path: Path to DICOM series directory
            modality: "CT" or "MRI" for appropriate windowing
            slice_index: Specific slice to extract (None = middle slice)

        Returns:
            Path to JPG slice image
        """
        try:
            import SimpleITK as sitk
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("SimpleITK and Pillow required for DICOM conversion")

        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="vila_m3_")

        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_path)

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_path}")

        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        # Get array (SimpleITK uses [Z, Y, X] ordering)
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        # Select slice (middle by default)
        num_slices = array.shape[0]
        if slice_index is None:
            slice_index = num_slices // 2
        slice_index = max(0, min(slice_index, num_slices - 1))

        slice_2d = array[slice_index, :, :]

        # Apply windowing based on modality
        if modality.upper() == "CT":
            # Soft tissue window (WL=50, WW=400) - matches VILA-M3's utils.py
            window_center = 50
            window_width = 400
            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2
            slice_2d = np.clip(slice_2d, min_val, max_val)
            slice_2d = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        elif modality.upper() == "MRI":
            # Percentile-based scaling for MRI
            p2, p98 = np.percentile(slice_2d, (2, 98))
            slice_2d = np.clip(slice_2d, p2, p98)
            if p98 > p2:
                slice_2d = ((slice_2d - p2) / (p98 - p2) * 255).astype(np.uint8)
            else:
                slice_2d = np.zeros_like(slice_2d, dtype=np.uint8)
        else:
            # Generic min-max scaling
            min_val, max_val = slice_2d.min(), slice_2d.max()
            if max_val > min_val:
                slice_2d = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                slice_2d = np.zeros_like(slice_2d, dtype=np.uint8)

        # Rotate to match standard viewing orientation (RAS)
        slice_2d = np.rot90(slice_2d, k=1)
        slice_2d = np.flipud(slice_2d)

        # Convert to RGB (PIL expects RGB for .jpg)
        rgb_slice = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)

        # Save as JPG
        jpg_path = os.path.join(self._temp_dir, f"slice_{slice_index}.jpg")
        pil_image = PILImage.fromarray(rgb_slice, mode='RGB')
        pil_image.save(jpg_path, quality=95)

        return jpg_path

    def load_model(self) -> None:
        """Load VILA-M3 model and framework."""
        if self._loaded:
            return

        # Setup Python path for VILA framework
        self._vila_root, self._demo_dir = _setup_vila_path()

        # Check framework exists
        if not self._vila_root.exists():
            raise FileNotFoundError(
                f"VILA-M3 framework not found at {self._vila_root}. "
                "Clone it with: git clone --recursive "
                "https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework "
                "external/vila-m3"
            )

        print(f"Loading VILA-M3 model ({self.model_size.upper()})...")
        print(f"  Model: {self.model_path}")
        print(f"  Conv mode: {self.conv_mode}")
        print(f"  Experts enabled: {self.enable_experts}")

        try:
            # Import from VILA framework
            from gradio_m3 import M3Generator
            from experts.utils import ImageCache

            # Determine source type
            model_path_str = str(self.model_path)
            if model_path_str.startswith("MONAI/") or "/" not in model_path_str:
                source = "huggingface"
            else:
                source = "local"

            print(f"  Source: {source}")

            # Create generator
            self.generator = M3Generator(
                source=source,
                model_path=model_path_str,
                conv_mode=self.conv_mode,
            )

            self._loaded = True
            print("VILA-M3 model loaded successfully!")

            mem = self.get_memory_usage()
            print(f"GPU Memory: {mem['allocated_gb']:.2f}GB allocated")

        except Exception as e:
            raise RuntimeError(f"Failed to load VILA-M3: {e}")

    def unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if self.generator is not None:
            del self.generator
            self.generator = None

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cleanup temp directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

        print("VILA-M3 model unloaded.")

    def _prepare_image(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str,
        slice_index: Optional[int] = None,
    ) -> str:
        """
        Prepare image for VILA-M3 processing.

        VILA-M3 uses PIL for loading images, so we convert everything to
        2D JPG/PNG format. For 3D volumes (DICOM or arrays), we extract
        a representative slice.

        Args:
            image: DICOM path, image path, or array
            modality: CT or MRI (for windowing)
            slice_index: Specific slice to extract from 3D volumes

        Returns:
            Path to 2D image file (.jpg/.png)
        """
        if isinstance(image, str):
            path = Path(image)

            # Check if it's a DICOM directory
            if path.is_dir():
                # Convert DICOM to 2D slice image (JPG)
                return self._convert_dicom_to_slice_image(image, modality, slice_index)

            # Check if it's already a 2D image
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                return str(path)

            # NIfTI files need slice extraction
            if path.suffix.lower() in ['.nii', '.gz']:
                return self._convert_nifti_to_slice_image(str(path), modality, slice_index)

            raise ValueError(f"Unsupported image path: {image}")

        elif isinstance(image, (np.ndarray, torch.Tensor)):
            # Convert array to 2D slice image
            return self._convert_array_to_slice_image(image, modality, slice_index)

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _convert_nifti_to_slice_image(
        self,
        nifti_path: str,
        modality: str = "CT",
        slice_index: Optional[int] = None,
    ) -> str:
        """Convert NIfTI file to a 2D slice image (JPG)."""
        try:
            import nibabel as nib
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("nibabel and Pillow required for NIfTI conversion")

        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="vila_m3_")

        # Load NIfTI
        nii = nib.load(nifti_path)
        array = nii.get_fdata().astype(np.float32)

        # Select slice (middle by default)
        num_slices = array.shape[2]  # nibabel uses [X, Y, Z]
        if slice_index is None:
            slice_index = num_slices // 2
        slice_index = max(0, min(slice_index, num_slices - 1))

        slice_2d = array[:, :, slice_index]

        # Apply windowing (same as DICOM conversion)
        slice_2d = self._apply_windowing(slice_2d, modality)

        # Rotate to match viewing orientation
        slice_2d = np.rot90(slice_2d, k=1)

        # Convert to RGB
        rgb_slice = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)

        # Save as JPG
        jpg_path = os.path.join(self._temp_dir, f"nifti_slice_{slice_index}.jpg")
        pil_image = PILImage.fromarray(rgb_slice, mode='RGB')
        pil_image.save(jpg_path, quality=95)

        return jpg_path

    def _convert_array_to_slice_image(
        self,
        array: Union[np.ndarray, torch.Tensor],
        modality: str = "CT",
        slice_index: Optional[int] = None,
    ) -> str:
        """Convert numpy/torch array to a 2D slice image (JPG)."""
        from PIL import Image as PILImage

        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="vila_m3_")

        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()

        # Remove batch/channel dims if present
        while array.ndim > 3:
            array = array[0]

        array = array.astype(np.float32)

        # Select slice if 3D
        if array.ndim == 3:
            num_slices = array.shape[0]  # Assuming [Z, Y, X] or [D, H, W]
            if slice_index is None:
                slice_index = num_slices // 2
            slice_index = max(0, min(slice_index, num_slices - 1))
            slice_2d = array[slice_index, :, :]
        else:
            slice_2d = array

        # Apply windowing
        slice_2d = self._apply_windowing(slice_2d, modality)

        # Convert to RGB
        rgb_slice = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)

        # Save as JPG
        jpg_path = os.path.join(self._temp_dir, f"array_slice.jpg")
        pil_image = PILImage.fromarray(rgb_slice, mode='RGB')
        pil_image.save(jpg_path, quality=95)

        return jpg_path

    def _apply_windowing(self, slice_2d: np.ndarray, modality: str) -> np.ndarray:
        """Apply appropriate windowing based on modality."""
        if modality.upper() == "CT":
            # Soft tissue window (WL=50, WW=400)
            window_center = 50
            window_width = 400
            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2
            slice_2d = np.clip(slice_2d, min_val, max_val)
            return ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        elif modality.upper() == "MRI":
            # Percentile-based scaling for MRI
            p2, p98 = np.percentile(slice_2d, (2, 98))
            slice_2d = np.clip(slice_2d, p2, p98)
            if p98 > p2:
                return ((slice_2d - p2) / (p98 - p2) * 255).astype(np.uint8)
            return np.zeros_like(slice_2d, dtype=np.uint8)
        else:
            # Generic min-max scaling
            min_val, max_val = slice_2d.min(), slice_2d.max()
            if max_val > min_val:
                return ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            return np.zeros_like(slice_2d, dtype=np.uint8)

    def _clean_response(self, response: str, modality: str = "CT") -> str:
        """
        Clean VILA-M3 response by removing expert model artifacts.

        Removes:
        - VISTA3D segmentation labels (e.g., "identified by VISTA3D: red: liver...")
        - Color-coded anatomical labels (e.g., "red: spinal cord, yellow: skull")
        - Corrects modality confusion (MRI described as CT or vice versa)

        Args:
            response: Raw model response
            modality: Actual modality of the scan

        Returns:
            Cleaned response string
        """
        cleaned = response.strip()

        # Remove VISTA3D labels and color-coded anatomy descriptions
        for pattern in VISTA3D_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Fix modality confusion (e.g., MRI scan called "CT image")
        if modality in MODALITY_PATTERNS:
            wrong_modality_terms = MODALITY_PATTERNS[modality]
            correct_term = f"{modality} image" if "image" in wrong_modality_terms[0] else f"{modality} scan"

            for wrong_pattern in wrong_modality_terms:
                cleaned = re.sub(wrong_pattern, correct_term, cleaned, flags=re.IGNORECASE)

        # Clean up multiple spaces and empty sentences
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\.\s*\.', '.', cleaned)  # Multiple periods
        cleaned = re.sub(r'^\s*,\s*', '', cleaned)  # Leading comma
        cleaned = cleaned.strip()

        return cleaned

    def generate_response(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        modality: str = "CT",
        slice_index: Optional[int] = None,
        use_experts: Optional[bool] = None,
        clean_output: bool = True,
        include_model_cards: bool = True,
    ) -> str:
        """
        Generate a response for a question about an image.

        The prompt format follows the original VILA-M3 implementation:
        [MODEL_CARDS]<image>This is a {modality} image.\n{question}

        Args:
            image: DICOM path, image path, or array
            question: Question to ask about the image
            modality: "CT" or "MRI"
            slice_index: Specific slice to analyze (for 3D volumes)
            use_experts: Whether to enable expert models (overrides init setting)
            clean_output: If True, clean response to remove expert model artifacts
            include_model_cards: If True, include MODEL_CARDS context (recommended)

        Returns:
            Model's response string
        """
        self.ensure_loaded()

        # Prepare image (converts to 2D JPG slice for PIL compatibility)
        image_path = self._prepare_image(image, modality, slice_index)

        # Build prompt following original VILA-M3 format:
        # model_cards + "<image>" + mod_msg + prompt
        # This is how gradio_m3.py constructs prompts (lines 427-438)
        model_cards = MODEL_CARDS if include_model_cards else ""
        mod_msg = f"This is a {modality} image.\n" if modality != "Unknown" else ""

        # Construct the full text: model_cards + <image> + mod_msg + question
        # Note: <image> token MUST come before the modality message
        prompt_text = f"{model_cards}<image>{mod_msg}{question}"

        # Build message structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_path", "image_path": image_path},
                ]
            }
        ]

        # Generate response
        response = self.generator.generate_response(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=0.9,
        )

        response = response.strip()

        # Clean response to remove expert model artifacts
        if clean_output:
            response = self._clean_response(response, modality)

        return response

    def generate_with_segmentation(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        modality: str = "CT",
        segmentation_target: str = "everything",
    ) -> Tuple[str, Optional[str]]:
        """
        Generate response with VISTA3D segmentation.

        Args:
            image: DICOM path or array
            question: Question about the image
            modality: "CT" or "MRI"
            segmentation_target: What to segment ("everything", "organs", etc.)

        Returns:
            Tuple of (response_text, path_to_segmentation_file)
        """
        self.ensure_loaded()

        # Prepare image
        image_path = self._prepare_image(image, modality)

        # First, ask the model to trigger VISTA3D
        seg_prompt = f"Please segment the {segmentation_target} in this image using VISTA3D."

        # Use correct prompt format with MODEL_CARDS
        mod_msg = f"This is a {modality} image.\n" if modality != "Unknown" else ""
        prompt_text = f"{MODEL_CARDS}<image>{mod_msg}{seg_prompt}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_path", "image_path": image_path},
                ]
            }
        ]

        # Generate - this should trigger VISTA3D if experts are enabled
        response = self.generator.generate_response(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=0.9,
        )

        # Check if segmentation was created
        seg_path = None
        if self._temp_dir:
            potential_seg = os.path.join(self._temp_dir, "segmentation.nii.gz")
            if os.path.exists(potential_seg):
                seg_path = potential_seg

        # Now answer the original question
        if question != seg_prompt:
            final_response = self.generate_response(image, question, modality)
        else:
            final_response = response

        return final_response, seg_path

    def analyse_scan(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
        analysis_questions: Optional[List[tuple]] = None,
        segment: bool = False,
    ) -> Dict:
        """
        Run comprehensive scan analysis.

        Args:
            image: DICOM path or array
            modality: "CT" or "MRI"
            analysis_questions: List of (name, question) tuples
            segment: Whether to run VISTA3D segmentation

        Returns:
            Dictionary with analysis results
        """
        self.ensure_loaded()

        if analysis_questions is None:
            analysis_questions = DEFAULT_ANALYSIS_QUESTIONS

        # Prepare image once
        image_path = self._prepare_image(image, modality)

        results = {
            "model": self.name,
            "model_size": self.model_size,
            "overall_findings": "",
            "analysis": {},
            "recommendations": "",
            "segmentation_path": None,
        }

        # Overall findings
        print("Generating overall findings...")
        overall_question = (
            f"Please provide a comprehensive analysis of this {modality} scan. "
            "Describe the key anatomical structures visible, any abnormalities, "
            "and overall image quality."
        )
        results["overall_findings"] = self.generate_response(
            image_path, overall_question, modality
        )

        # Run segmentation if requested
        if segment and modality == "CT":
            print("Running VISTA3D segmentation...")
            try:
                _, seg_path = self.generate_with_segmentation(
                    image_path, "Segment everything", modality, "everything"
                )
                results["segmentation_path"] = seg_path
            except Exception as e:
                print(f"  Segmentation failed: {e}")

        # Run each analysis question
        print("Running analysis...")
        for name, question in tqdm(analysis_questions, desc="Analysis"):
            results["analysis"][name] = self.generate_response(
                image_path, question, modality
            )

        # Recommendations
        print("Generating recommendations...")
        rec_question = (
            f"Based on your analysis of this {modality} scan and any findings, "
            "what clinical recommendations would you suggest?"
        )
        results["recommendations"] = self.generate_response(
            image_path, rec_question, modality
        )

        return results


@register_model("vila-m3-3b")
class VILAm3Model3B(VILAm3Model):
    """VILA-M3 3B variant."""

    name = "vila-m3-3b"

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        if model_path is None:
            model_path = "MONAI/Llama3-VILA-M3-3B"
        super().__init__(model_path=model_path, model_size="3b", **kwargs)


@register_model("vila-m3-13b")
class VILAm3Model13B(VILAm3Model):
    """VILA-M3 13B variant."""

    name = "vila-m3-13b"

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        if model_path is None:
            model_path = "MONAI/Llama3-VILA-M3-13B"
        super().__init__(model_path=model_path, model_size="13b", **kwargs)
