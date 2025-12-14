"""
M3D-LaMed Model Implementation

Wraps the M3D-LaMed-Phi-3-4B model for 3D medical image VQA.
Based on paper: arXiv:2404.00578v1
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import register_model
from .base import Medical3DModel
from ..preprocessing import preprocess_for_m3d
from ..prompts import (
    SYSTEM_PROMPT_RADIOLOGIST,
    SYSTEM_PROMPT_SCREENING,
    get_analysis_chain,
    get_quick_analysis_chain,
    get_pathology_screen,
    format_closed_question,
    REPORT_GENERATION_PROMPTS,
    PLANE_QUESTIONS,
    PHASE_QUESTIONS,
)


# Default analysis questions for open-ended findings
DEFAULT_ANALYSIS_QUESTIONS = [
    ("abnormalities", "Identify any abnormalities or pathological findings visible in this scan. Describe their location, appearance, and clinical significance."),
    ("key_findings", "What are the most clinically significant findings in this image? List them in order of importance."),
    ("differential", "Based on the findings in this scan, what differential diagnoses would you consider?"),
]


@register_model("m3d-lamed")
class M3DLaMedModel(Medical3DModel):
    """
    M3D-LaMed-Phi-3-4B model for 3D medical image VQA.

    Features:
    - Input: 32x256x256 single-channel volumes
    - Base LLM: Microsoft Phi-3-4B
    - Supports: VQA, report generation, segmentation
    - 256 <im_patch> tokens for image embedding
    """

    name = "m3d-lamed"
    model_type = "vqa"

    # Model-specific constants
    IMAGE_PATCH_COUNT = 256
    IMAGE_PATCH_TOKEN = "<im_patch>"
    SEG_TOKEN_ID = 32014

    @classmethod
    def get_default_model_path(cls) -> Path:
        """Return the default path to M3D-LaMed model files."""
        return Path(__file__).parent.parent.parent / "models" / "M3D-LaMed-Phi-3-4B"

    def get_input_shape(self) -> Tuple[int, int, int]:
        """Return expected input shape (D, H, W)."""
        return (32, 256, 256)

    def get_channels(self) -> int:
        """Return number of input channels (1 for grayscale)."""
        return 1

    def preprocess_tensor(
        self,
        volume: np.ndarray,
        modality: str = "CT"
    ) -> torch.Tensor:
        """
        Preprocess volume for M3D-LaMed input.

        Uses the existing preprocessing pipeline which handles:
        - CT windowing (soft tissue window WW=350, WL=40)
        - MRI normalization (percentile clipping + z-score)
        - Output shape: (B, 1, 32, 256, 256)
        """
        # If string path, use full preprocessing pipeline
        if isinstance(volume, str):
            volume = preprocess_for_m3d(volume, modality=modality)

        # Already a tensor - ensure proper shape and device
        if isinstance(volume, torch.Tensor):
            tensor = volume.to(dtype=self.dtype, device=self.device)
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
            return tensor

        # Numpy array - add batch dimension if needed
        if volume.ndim == 4:
            volume = volume[np.newaxis, ...]

        return torch.from_numpy(volume).to(dtype=self.dtype, device=self.device)

    def load_model(self) -> None:
        """Load M3D-LaMed model and tokenizer."""
        if self._loaded:
            return

        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        print(f"Loading M3D-LaMed model from {self.model_path}...")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")

        # Use eager attention (SDPA not supported by this model)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )

        self.model.eval()
        self._loaded = True
        print("M3D-LaMed model loaded successfully!")

        # Print memory usage
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

        print("M3D-LaMed model unloaded.")

    def _build_prompt(self, question: str) -> str:
        """
        Build prompt with image patch tokens.

        IMPORTANT: Uses simple format, NOT chat template!
        The model's prepare_inputs_for_multimodal expects:
          - Position 0: BOS token
          - Positions 1-256: <im_patch> tokens (replaced by image features)
          - Position 257+: question text
        """
        image_tokens = self.IMAGE_PATCH_TOKEN * self.IMAGE_PATCH_COUNT
        return f"{image_tokens}{question}"

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

        # Build prompt with image patch tokens
        prompt = self._build_prompt(question)

        # Tokenize
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)

        input_len = input_ids.shape[1]

        # Suppress segmentation token from text output
        seg_token_id = getattr(self.model.config, 'seg_token_id', self.SEG_TOKEN_ID)

        with torch.no_grad():
            result = self.model.generate(
                image_tensor,
                input_ids,
                seg_enable=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=self.max_new_tokens,
                bad_words_ids=[[seg_token_id]],
            )
            outputs = result[0] if isinstance(result, tuple) else result

        # Extract only new tokens
        if outputs.shape[1] > input_len:
            new_tokens = outputs[0][input_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.strip()

    def generate_with_segmentation(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        modality: str = "CT",
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate a response with optional segmentation mask.

        When the model identifies something to segment, it outputs [SEG] token
        and produces a 3D segmentation mask.

        Args:
            image: DICOM path, preprocessed array, or tensor
            question: Question to ask about the image
            modality: "CT" or "MRI"

        Returns:
            Tuple of (response_text, segmentation_mask or None)
        """
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)
        prompt = self._build_prompt(question)

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)

        input_len = input_ids.shape[1]

        try:
            with torch.no_grad():
                result = self.model.generate(
                    image_tensor,
                    input_ids,
                    seg_enable=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=self.max_new_tokens,
                )

            if isinstance(result, tuple) and len(result) == 2:
                output_ids, seg_logits = result

                # Decode text response
                if output_ids.shape[1] > input_len:
                    new_tokens = output_ids[0][input_len:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                else:
                    response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # Convert segmentation logits to binary mask
                if seg_logits is not None:
                    seg_probs = torch.sigmoid(seg_logits)
                    seg_mask = (seg_probs > 0.5).float()

                    if seg_mask.sum() > 0:
                        seg_np = seg_mask[0, 0].cpu().numpy()
                        return response.strip(), seg_np

                return response.strip(), None
            else:
                outputs = result[0] if isinstance(result, tuple) else result
                if outputs.shape[1] > input_len:
                    new_tokens = outputs[0][input_len:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                else:
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.strip(), None

        except RuntimeError as e:
            if "expected a non-empty list" in str(e) or "cat()" in str(e):
                return self.generate_response(image_tensor, question, modality), None
            raise

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
            segment: If True, generate segmentation masks for abnormalities

        Returns:
            Dictionary with analysis results
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
            "segmentations": {} if segment else None,
        }

        # Overall findings
        print("Generating overall findings...")
        overall_question = (
            f"Please provide a comprehensive analysis of this {modality} scan. "
            "Describe the key anatomical structures visible, any abnormalities, "
            "and overall image quality."
        )

        if segment:
            response, seg_mask = self.generate_with_segmentation(
                image_tensor, overall_question, modality
            )
            results["overall_findings"] = response
            if seg_mask is not None:
                results["segmentations"]["overall"] = seg_mask
        else:
            results["overall_findings"] = self.generate_response(
                image_tensor, overall_question, modality
            )

        # Run each analysis question
        print("Running analysis...")
        for name, question in tqdm(analysis_questions, desc="Analysis"):
            full_question = f"{question} Be specific about location and confidence level."

            if segment:
                response, seg_mask = self.generate_with_segmentation(
                    image_tensor, full_question, modality
                )
                results["analysis"][name] = response
                if seg_mask is not None:
                    results["segmentations"][name] = seg_mask
            else:
                results["analysis"][name] = self.generate_response(
                    image_tensor, full_question, modality
                )

        # Recommendations
        print("Generating recommendations...")
        rec_question = (
            f"Based on your analysis of this {modality} scan and any findings identified, "
            "what clinical recommendations would you suggest? "
            "Include suggestions for additional imaging, consultations, "
            "or urgent actions if warranted. If no significant findings, state that clearly."
        )
        results["recommendations"] = self.generate_response(
            image_tensor, rec_question, modality
        )

        return results

    # =========================================================================
    # M3D-LaMed specific methods (not part of base interface)
    # =========================================================================

    def analyse_chained(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
        quick: bool = False,
    ) -> Dict:
        """
        Run chained query analysis following paper recommendations.

        Starts with high-accuracy queries (plane, phase) before moving
        to lower-accuracy queries (abnormality, location).
        """
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)

        if quick:
            chain = get_quick_analysis_chain(modality)
        else:
            chain = get_analysis_chain(modality)

        results = {
            "model": self.name,
            "modality": modality,
            "chain_results": {},
            "summary": "",
        }

        print(f"Running {'quick ' if quick else ''}chained analysis...")
        for name, question, qtype in tqdm(chain, desc="Analysis chain"):
            response = self.generate_response(image_tensor, question, modality)
            results["chain_results"][name] = {
                "question": question,
                "response": response,
                "type": qtype,
            }

        # Generate summary
        print("Generating summary...")
        findings_text = "\n".join([
            f"- {name}: {r['response']}"
            for name, r in results["chain_results"].items()
        ])
        summary_question = (
            f"Based on the following analysis of this {modality} scan:\n"
            f"{findings_text}\n\n"
            "Provide a concise clinical summary and any recommendations."
        )
        results["summary"] = self.generate_response(image_tensor, summary_question, modality)

        return results

    def ask_closed_question(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        question: str,
        choices: List[str],
        modality: str = "CT",
    ) -> Dict:
        """
        Ask a closed-ended (multiple choice) question.

        Closed-ended questions have higher accuracy (75.78% mean) than open-ended.
        """
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)
        full_question = format_closed_question(question, choices)

        response = self.generate_response(image_tensor, full_question, modality)

        # Extract choice letter
        answer_letter = None
        response_upper = response.upper().strip()
        for choice in choices:
            letter = choice.split(".")[0].strip().upper()
            if response_upper.startswith(letter) or f" {letter}." in response_upper:
                answer_letter = letter
                break

        return {
            "question": question,
            "choices": choices,
            "full_question": full_question,
            "raw_response": response,
            "answer_letter": answer_letter,
        }

    def screen_pathology(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        region: str = "head",
        modality: str = "CT",
        pathologies: Optional[List[str]] = None,
    ) -> Dict:
        """
        Screen for specific pathologies using closed-ended questions.
        """
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)
        pathology_questions = get_pathology_screen(region)

        if pathologies:
            pathology_questions = {
                k: v for k, v in pathology_questions.items()
                if k in pathologies
            }

        results = {
            "model": self.name,
            "region": region,
            "modality": modality,
            "screenings": {},
            "positive_findings": [],
            "uncertain_findings": [],
        }

        print(f"Screening for {len(pathology_questions)} pathologies...")
        for name, pq in tqdm(pathology_questions.items(), desc="Pathology screening"):
            answer = self.ask_closed_question(
                image_tensor, pq["question"], pq["choices"], modality
            )
            results["screenings"][name] = answer

            if answer["answer_letter"] == "A":
                results["positive_findings"].append(name)
            elif answer["answer_letter"] == "B":
                results["uncertain_findings"].append(name)

        return results

    def generate_report(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        modality: str = "CT",
    ) -> str:
        """Generate a findings report for the image."""
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)

        import random
        prompt = random.choice(REPORT_GENERATION_PROMPTS)

        return self.generate_response(image_tensor, prompt, modality)

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
        """
        self.ensure_loaded()

        image_tensor = self.preprocess_tensor(image, modality)

        results = {
            "model": self.name,
            "modality": modality,
            "region": region,
            "image_parameters": {},
            "findings": {},
            "pathology_screening": None,
            "report": "",
            "recommendations": "",
            "segmentations": {} if segment else None,
        }

        # Step 1: Image parameters
        print("Establishing image parameters...")
        plane_response = self.generate_response(
            image_tensor, PLANE_QUESTIONS["open"][0], modality
        )
        results["image_parameters"]["plane"] = plane_response

        if modality == "CT":
            phase_response = self.generate_response(
                image_tensor, PHASE_QUESTIONS["open"][0], modality
            )
            results["image_parameters"]["phase"] = phase_response

        # Step 2: Findings analysis
        print("Analysing findings...")
        for name, question in DEFAULT_ANALYSIS_QUESTIONS:
            full_question = f"{question} Be specific about location and confidence level."

            if segment:
                response, seg_mask = self.generate_with_segmentation(
                    image_tensor, full_question, modality
                )
                results["findings"][name] = response
                if seg_mask is not None:
                    results["segmentations"][name] = seg_mask
            else:
                results["findings"][name] = self.generate_response(
                    image_tensor, full_question, modality
                )

        # Step 3: Pathology screening
        if include_pathology_screen:
            print("Running pathology screening...")
            results["pathology_screening"] = self.screen_pathology(
                image_tensor, region=region, modality=modality
            )

        # Step 4: Report
        print("Generating report...")
        results["report"] = self.generate_report(image_tensor, modality)

        # Step 5: Recommendations
        print("Generating recommendations...")
        rec_question = (
            f"Based on your analysis of this {modality} scan and any findings identified, "
            "what clinical recommendations would you suggest? "
            "Include suggestions for additional imaging, consultations, "
            "or urgent actions if warranted. If no significant findings, state that clearly."
        )
        results["recommendations"] = self.generate_response(
            image_tensor, rec_question, modality
        )

        return results
