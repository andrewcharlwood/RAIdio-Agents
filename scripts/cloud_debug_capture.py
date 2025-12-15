#!/usr/bin/env python3
"""
Cloud Debug Capture Script

Run this ONCE on cloud to capture all debug data for local analysis.
Outputs: debug_capture_{timestamp}.json with comprehensive diagnostic info for all models.

Usage:
    python scripts/cloud_debug_capture.py --dicom-path /path/to/dicom --output debug_output.json
    python scripts/cloud_debug_capture.py --test  # Uses synthetic test data
"""

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# Med3DVLM cleaning patterns (from med3dvlm.py)
BOUNDING_BOX_PATTERN = re.compile(r'\[\s*[\d.]+(?:\s*,\s*[\d.]+)*\s*\]')
CONVERSATIONAL_PREFIXES = [
    "Sure, it is ", "Sure, ", "Certainly, ", "Of course, ",
    "The answer is ", "Based on the image, ", "Looking at the image, ",
]
VQA_PATTERNS = [
    r'^(?:The\s+)?(?:answer|response)\s+(?:is|would be)[:\s]+',
    r'^(?:It\s+)?(?:appears|looks|seems)\s+(?:to be|like)[:\s]+',
]


def capture_m3d_lamed_debug(
    dicom_path: Optional[str],
    question: str = "What anatomical structures are visible in this scan?",
    modality: str = "CT",
    use_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Capture M3D-LaMed internals for debugging.

    Returns detailed info about:
    - Tokenized input IDs
    - <im_patch> token positions (should be 1-256)
    - BOS token position (should be 0)
    - Image tensor shape
    - Generation config
    - Raw output
    """
    result = {
        "model": "m3d-lamed",
        "status": "error",
        "error": None,
        "debug_data": {},
    }

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Find model path
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "M3D-LaMed-Phi-3-4B"

        if not model_path.exists():
            result["error"] = f"Model not found at {model_path}"
            return result

        print("Loading M3D-LaMed tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,  # Prevent overwriting fine-tuned model files
        )

        # Constants
        IMAGE_PATCH_COUNT = 256
        IMAGE_PATCH_TOKEN = "<im_patch>"
        SEG_TOKEN_ID = 32014

        # Build prompt (SIMPLE format, not chat template)
        image_tokens = IMAGE_PATCH_TOKEN * IMAGE_PATCH_COUNT
        prompt = f"{image_tokens}{question}"

        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Get token IDs for analysis
        im_patch_id = tokenizer.convert_tokens_to_ids(IMAGE_PATCH_TOKEN)
        bos_id = tokenizer.bos_token_id

        # Find positions
        input_ids_list = input_ids[0].tolist()
        im_patch_positions = [i for i, tid in enumerate(input_ids_list) if tid == im_patch_id]
        bos_positions = [i for i, tid in enumerate(input_ids_list) if tid == bos_id]

        # Check for chat template indicators (role tokens)
        vocab = tokenizer.get_vocab()
        role_tokens = ["<|user|>", "<|assistant|>", "<|system|>", "[INST]", "[/INST]"]
        found_role_tokens = {}
        for rt in role_tokens:
            if rt in vocab:
                rt_id = vocab[rt]
                rt_positions = [i for i, tid in enumerate(input_ids_list) if tid == rt_id]
                if rt_positions:
                    found_role_tokens[rt] = rt_positions

        # Store tokenizer debug info
        result["debug_data"]["tokenizer"] = {
            "im_patch_token": IMAGE_PATCH_TOKEN,
            "im_patch_id": im_patch_id,
            "bos_token_id": bos_id,
            "vocab_size": len(vocab),
            "input_length": len(input_ids_list),
            "first_10_tokens": input_ids_list[:10],
            "last_10_tokens": input_ids_list[-10:],
        }

        result["debug_data"]["token_positions"] = {
            "im_patch_positions": im_patch_positions[:5] + ["..."] + im_patch_positions[-5:] if len(im_patch_positions) > 10 else im_patch_positions,
            "im_patch_count": len(im_patch_positions),
            "expected_im_patch_count": IMAGE_PATCH_COUNT,
            "im_patch_positions_correct": (
                len(im_patch_positions) == IMAGE_PATCH_COUNT and
                im_patch_positions[0] == 1 and
                im_patch_positions[-1] == IMAGE_PATCH_COUNT
            ),
            "bos_positions": bos_positions,
            "bos_at_position_0": 0 in bos_positions,
            "chat_template_tokens_found": found_role_tokens,
            "chat_template_used": len(found_role_tokens) > 0,
        }

        # Load model and run inference
        print("Loading M3D-LaMed model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,  # Prevent overwriting fine-tuned model files
            attn_implementation="eager",
        )
        model.eval()

        # Create image tensor
        if use_synthetic or dicom_path is None:
            print("Using synthetic test volume...")
            image_tensor = torch.randn(1, 1, 32, 256, 256, dtype=torch.float16, device="cuda")
            result["debug_data"]["preprocessing"] = {"synthetic": True}
        else:
            import SimpleITK as sitk
            print(f"Loading DICOM from {dicom_path}...")

            # First, capture raw DICOM info BEFORE preprocessing
            try:
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(dicom_path)

                if dicom_files:
                    # Read just the first file to get metadata
                    first_file = sitk.ReadImage(dicom_files[0])

                    # Try to get raw pixel stats before windowing
                    reader.SetFileNames(dicom_files)
                    raw_image = reader.Execute()
                    raw_array = sitk.GetArrayFromImage(raw_image)

                    result["debug_data"]["preprocessing"] = {
                        "dicom_file_count": len(dicom_files),
                        "raw_image_size": list(raw_image.GetSize()),  # (X, Y, Z)
                        "raw_image_spacing": list(raw_image.GetSpacing()),
                        "raw_pixel_type": str(raw_image.GetPixelIDTypeAsString()),
                        "raw_min_value": float(raw_array.min()),
                        "raw_max_value": float(raw_array.max()),
                        "raw_mean_value": float(raw_array.mean()),
                        "raw_std_value": float(raw_array.std()),
                        "series_description": first_file.GetMetaData("0008|103e") if first_file.HasMetaDataKey("0008|103e") else "unknown",
                        "modality_tag": first_file.GetMetaData("0008|0060") if first_file.HasMetaDataKey("0008|0060") else "unknown",
                        "is_likely_scout": (
                            raw_array.shape[0] <= 3 or  # Very few slices
                            "topo" in str(first_file.GetMetaData("0008|103e") if first_file.HasMetaDataKey("0008|103e") else "").lower() or
                            "scout" in str(first_file.GetMetaData("0008|103e") if first_file.HasMetaDataKey("0008|103e") else "").lower() or
                            "localizer" in str(first_file.GetMetaData("0008|103e") if first_file.HasMetaDataKey("0008|103e") else "").lower()
                        ),
                    }

                    # Check if values look like HU (typically -1024 to +3000)
                    if raw_array.min() >= 0 and raw_array.max() < 5000:
                        result["debug_data"]["preprocessing"]["warning"] = (
                            "Raw values may not be in Hounsfield Units! "
                            f"Range [{raw_array.min()}, {raw_array.max()}] - expected negative values for CT."
                        )
            except Exception as e:
                result["debug_data"]["preprocessing"] = {"error": str(e)}

            from src.preprocessing import preprocess_for_m3d
            volume = preprocess_for_m3d(dicom_path, modality=modality)
            image_tensor = torch.from_numpy(volume).to(dtype=torch.float16, device="cuda")
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(0)

        result["debug_data"]["image_tensor"] = {
            "shape": list(image_tensor.shape),
            "expected_shape": [1, 1, 32, 256, 256],
            "shape_correct": list(image_tensor.shape) == [1, 1, 32, 256, 256],
            "dtype": str(image_tensor.dtype),
            "device": str(image_tensor.device),
            "min_value": float(image_tensor.min()),
            "max_value": float(image_tensor.max()),
            "mean_value": float(image_tensor.mean()),
            "std_value": float(image_tensor.std()),
            "is_constant": float(image_tensor.std()) < 0.001,
            "value_range_ok": float(image_tensor.min()) >= 0 and float(image_tensor.max()) <= 1 and float(image_tensor.std()) > 0.01,
        }

        # Run generation
        print("Running generation...")
        input_ids = input_ids.to("cuda")
        input_len = input_ids.shape[1]

        seg_token_id = getattr(model.config, 'seg_token_id', SEG_TOKEN_ID)

        with torch.no_grad():
            outputs = model.generate(
                image_tensor,
                input_ids,
                seg_enable=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=256,
                bad_words_ids=[[seg_token_id]],
            )

            if isinstance(outputs, tuple):
                output_ids = outputs[0]
            else:
                output_ids = outputs

        # Decode response
        if output_ids.shape[1] > input_len:
            new_tokens = output_ids[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        result["debug_data"]["generation"] = {
            "input_length": input_len,
            "output_length": output_ids.shape[1],
            "new_tokens_count": output_ids.shape[1] - input_len,
            "generation_config": {
                "seg_enable": False,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 256,
                "seg_token_id": seg_token_id,
            },
            "output_ids_sample": output_ids[0][-20:].tolist(),
        }

        result["debug_data"]["response"] = {
            "raw_response": response.strip(),
            "response_length": len(response),
            "starts_with_sure": response.strip().lower().startswith("sure"),
            "contains_coordinates": "[0." in response or "[1." in response,
            "contains_bounding_box": bool(any(
                f"[{d}." in response for d in "0123456789"
            ) and "]" in response),
        }

        # Cleanup
        del model
        torch.cuda.empty_cache()

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def capture_vila_m3_debug(
    dicom_path: Optional[str],
    question: str = "What anatomical structures are visible in this scan?",
    modality: str = "CT",
    use_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Capture VILA-M3 internals for debugging.

    Returns detailed info about:
    - Message format
    - Expert model calls/outputs
    - Modality handling
    - Final response
    """
    result = {
        "model": "vila-m3",
        "status": "error",
        "error": None,
        "debug_data": {},
    }

    try:
        project_root = Path(__file__).parent.parent
        vila_root = project_root / "external" / "vila-m3"

        if not vila_root.exists():
            result["error"] = f"VILA-M3 framework not found at {vila_root}"
            return result

        # Add VILA to path
        import sys
        vila_paths = [
            str(vila_root),
            str(vila_root / "thirdparty" / "VILA"),
            str(vila_root / "m3" / "demo"),
        ]
        for p in vila_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

        print("Loading VILA-M3 framework...")
        from gradio_m3 import M3Generator

        # Create generator
        generator = M3Generator(
            source="huggingface",
            model_path="MONAI/Llama3-VILA-M3-8B",
            conv_mode="llama_3",
        )

        # Build message structure
        if use_synthetic or dicom_path is None:
            # Create temp synthetic image
            import tempfile
            import SimpleITK as sitk

            temp_dir = tempfile.mkdtemp()
            synthetic_vol = np.random.randn(64, 128, 128).astype(np.float32)
            sitk_image = sitk.GetImageFromArray(synthetic_vol)
            image_path = f"{temp_dir}/synthetic.nii.gz"
            sitk.WriteImage(sitk_image, image_path)
        else:
            # Convert DICOM to 2D JPG slice (VILA-M3 uses PIL which can't read NIfTI)
            import SimpleITK as sitk
            import tempfile
            from PIL import Image as PILImage

            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_path)
            reader.SetFileNames(dicom_files)
            image = reader.Execute()

            # Get array and extract middle slice
            array = sitk.GetArrayFromImage(image).astype(np.float32)
            num_slices = array.shape[0]
            slice_idx = num_slices // 2
            slice_2d = array[slice_idx, :, :]

            # Apply CT soft tissue windowing (WL=50, WW=400)
            window_center = 50
            window_width = 400
            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2
            slice_2d = np.clip(slice_2d, min_val, max_val)
            slice_2d = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # Rotate and convert to RGB
            slice_2d = np.rot90(slice_2d, k=1)
            slice_2d = np.flipud(slice_2d)
            rgb_slice = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)

            # Save as JPG
            temp_dir = tempfile.mkdtemp()
            image_path = f"{temp_dir}/slice_{slice_idx}.jpg"
            pil_image = PILImage.fromarray(rgb_slice, mode='RGB')
            pil_image.save(image_path, quality=95)

            result["debug_data"]["slice_extraction"] = {
                "total_slices": num_slices,
                "selected_slice": slice_idx,
                "slice_shape": list(slice_2d.shape),
                "image_path": image_path,
            }

        # MODEL_CARDS: Expert model descriptions (required for proper model behavior)
        # The VILA-M3 model was trained with this context and expects it in the prompt
        MODEL_CARDS = """Here is a list of available expert models:
<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None
<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'
<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None
<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None
Give the model <NAME(args)> when selecting a suitable expert model.
"""

        # Build prompt following original VILA-M3 format (gradio_m3.py lines 427-438):
        # model_cards + "<image>" + mod_msg + prompt
        mod_msg = f"This is a {modality} image.\n"
        prompt_text = f"{MODEL_CARDS}<image>{mod_msg}{question}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_path", "image_path": image_path},
                ]
            }
        ]

        result["debug_data"]["message_format"] = {
            "messages": str(messages)[:500] + "...",  # Truncate for readability
            "prompt_format": "MODEL_CARDS + <image> + mod_msg + question",
            "mod_msg": mod_msg.strip(),
            "has_model_cards": True,
            "modality_passed": modality,
        }

        # Generate response
        print("Running VILA-M3 generation...")
        response = generator.generate_response(
            messages=messages,
            max_tokens=256,
            temperature=0.0,
            top_p=0.9,
        )

        result["debug_data"]["response"] = {
            "raw_response": response.strip(),
            "response_length": len(response),
            "contains_vista3d": "VISTA3D" in response or "vista3d" in response.lower(),
            "contains_ct_reference": "CT" in response.upper(),
            "modality_confusion": (modality == "MRI" and "CT image" in response),
        }

        # Check for expert model integration
        result["debug_data"]["expert_integration"] = {
            "experts_enabled": hasattr(generator, 'experts') or hasattr(generator, 'expert_models'),
            "response_has_color_labels": any(color in response.lower() for color in ["red:", "blue:", "yellow:", "green:"]),
            "response_has_anatomical_labels": "identified by" in response.lower(),
        }

        # Cleanup
        del generator
        torch.cuda.empty_cache()

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def _validate_med3dvlm_prompt(
    input_ids: torch.Tensor,
    proj_out_num: int,
    tokenizer,
) -> Dict[str, Any]:
    """
    Validate that Med3DVLM prompt is in correct simple format (not chat template).

    Checks:
    - <im_patch> tokens are present and correct count
    - Tokens are contiguous (not scattered due to chat template)

    Returns dict with validation results for debug output.
    """
    result = {
        "valid": False,
        "im_patch_count": 0,
        "expected_count": proj_out_num,
        "im_patch_positions": [],
        "is_contiguous": False,
        "warnings": [],
    }

    try:
        # Get <im_patch> token ID
        im_patch_id = tokenizer.convert_tokens_to_ids("<im_patch>")
        result["im_patch_token_id"] = im_patch_id

        # Convert to list for analysis
        ids_list = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()

        # Find all positions where <im_patch> appears
        im_patch_positions = [i for i, tid in enumerate(ids_list) if tid == im_patch_id]
        result["im_patch_count"] = len(im_patch_positions)
        result["im_patch_positions"] = im_patch_positions[:10]  # First 10 for brevity

        # Validate count matches proj_out_num
        if len(im_patch_positions) != proj_out_num:
            result["warnings"].append(
                f"Expected {proj_out_num} <im_patch> tokens, found {len(im_patch_positions)}"
            )
        else:
            # Validate contiguity
            if len(im_patch_positions) >= 2:
                expected_contiguous = (
                    im_patch_positions[-1] - im_patch_positions[0] == proj_out_num - 1
                )
                result["is_contiguous"] = expected_contiguous
                if not expected_contiguous:
                    result["warnings"].append(
                        "<im_patch> tokens are not contiguous - possible chat template issue"
                    )
                else:
                    result["valid"] = True
            elif len(im_patch_positions) == 1:
                result["is_contiguous"] = True
                result["valid"] = True

    except Exception as e:
        result["warnings"].append(f"Validation error: {str(e)}")

    return result


def _clean_med3dvlm_response(response: str) -> str:
    """
    Clean Med3DVLM response by removing artifacts.

    Removes:
    - Bounding boxes: [0.094, 0.348, ...]
    - Conversational prefixes: "Sure, it is...", "Certainly, ..."
    - Extra whitespace
    """
    cleaned = response.strip()

    # Remove bounding boxes using regex pattern
    cleaned = BOUNDING_BOX_PATTERN.sub('', cleaned)

    # Remove conversational prefixes
    for prefix in CONVERSATIONAL_PREFIXES:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):]
            break

    # Also handle VQA-style patterns
    for pattern in VQA_PATTERNS:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            cleaned = cleaned[match.end():]
            break

    # Normalize whitespace and strip
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def capture_med3dvlm_debug(
    dicom_path: Optional[str],
    question: str = "What anatomical structures are visible in this scan?",
    modality: str = "CT",
    use_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Capture Med3DVLM internals for debugging.

    Returns detailed info about:
    - Input tensor shape (should be 128x256x256, NOT 32x256x256)
    - Generation parameters
    - Raw output
    """
    result = {
        "model": "med3dvlm",
        "status": "error",
        "error": None,
        "debug_data": {},
    }

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from scipy.ndimage import zoom

        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "Med3DVLM"

        if not model_path.exists():
            result["error"] = f"Model not found at {model_path}"
            return result

        print("Loading Med3DVLM tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        print("Loading Med3DVLM model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Create/load image tensor
        TARGET_SHAPE = (128, 256, 256)  # Med3DVLM expects 128 depth, not 32!

        if use_synthetic or dicom_path is None:
            print("Using synthetic test volume...")
            volume = np.random.randn(*TARGET_SHAPE).astype(np.float32)
        else:
            from src.preprocessing import preprocess_for_m3d
            print(f"Loading DICOM from {dicom_path}...")
            volume = preprocess_for_m3d(dicom_path, modality=modality)

            # Volume comes as (1, D, H, W) - extract spatial dims
            if volume.ndim == 4:
                volume = volume[0]

            current_shape = volume.shape

            # Resize to Med3DVLM expected shape
            if current_shape != TARGET_SHAPE:
                zoom_factors = tuple(t / c for t, c in zip(TARGET_SHAPE, current_shape))
                volume = zoom(volume, zoom_factors, order=1)

        # Add batch and channel dims: (1, 1, 128, 256, 256)
        image_tensor = torch.from_numpy(volume[np.newaxis, np.newaxis, ...]).to(
            dtype=torch.bfloat16, device="cuda"
        )

        result["debug_data"]["image_tensor"] = {
            "shape": list(image_tensor.shape),
            "expected_shape": [1, 1, 128, 256, 256],
            "shape_correct": list(image_tensor.shape) == [1, 1, 128, 256, 256],
            "dtype": str(image_tensor.dtype),
            "device": str(image_tensor.device),
        }

        # Get proj_out_num from model config (number of image tokens)
        # Try multiple fallback locations (matching med3dvlm.py)
        proj_out_num = None
        proj_out_num_source = "default"

        try:
            proj_out_num = model.get_model().config.proj_out_num
            proj_out_num_source = "model.get_model().config"
        except (AttributeError, TypeError):
            pass

        if proj_out_num is None:
            try:
                proj_out_num = model.config.proj_out_num
                proj_out_num_source = "model.config"
            except (AttributeError, TypeError):
                pass

        if proj_out_num is None:
            proj_out_num = 256
            proj_out_num_source = "default (256)"

        # Build prompt with image tokens (same format as M3D-LaMed)
        # Format: <im_patch> * proj_out_num + question
        image_tokens = "<im_patch>" * proj_out_num
        prompt = image_tokens + question

        # Tokenize the full prompt (image tokens + question)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda")

        result["debug_data"]["prompt_format"] = {
            "proj_out_num": proj_out_num,
            "proj_out_num_source": proj_out_num_source,
            "image_tokens_count": proj_out_num,
            "prompt_length": len(prompt),
        }

        # Capture input length for proper response extraction
        input_len = inputs.input_ids.shape[1]

        result["debug_data"]["tokenizer"] = {
            "input_length": input_len,
            "input_ids_sample": inputs.input_ids[0][:20].tolist(),
            "expected_min_length": proj_out_num + 5,  # image tokens + question tokens
        }

        # Validate prompt format (matching med3dvlm.py _validate_prompt_format)
        prompt_validation = _validate_med3dvlm_prompt(
            inputs.input_ids, proj_out_num, tokenizer
        )
        result["debug_data"]["prompt_validation"] = prompt_validation

        # Run generation
        print("Running Med3DVLM generation...")

        # Try different parameter names
        generation_attempts = []

        # Generation parameters (matching med3dvlm.py)
        generation_kwargs = {
            "max_new_tokens": 256,
            "temperature": 1.0,  # Match original Med3DVLM (was 0.7 before fix)
            "do_sample": True,
            "top_p": 0.9,
            "use_cache": True,
        }

        # Attempt 1: Using 'images' parameter
        try:
            with torch.no_grad():
                outputs = model.generate(
                    images=image_tensor,
                    inputs=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_kwargs,
                )
            # Extract only new tokens using input_len (matching med3dvlm.py)
            if outputs.shape[1] > input_len:
                new_tokens = outputs[0][input_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            else:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_attempts.append({
                "param_name": "images",
                "success": True,
                "response": response[:500],  # Store more for debugging
                "output_length": outputs.shape[1],
                "new_tokens_count": outputs.shape[1] - input_len,
            })
        except Exception as e:
            generation_attempts.append({
                "param_name": "images",
                "success": False,
                "error": str(e),
            })

        # Attempt 2: Using 'pixel_values' parameter
        try:
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values=image_tensor,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_kwargs,
                )
            # Extract only new tokens using input_len (matching med3dvlm.py)
            if outputs.shape[1] > input_len:
                new_tokens = outputs[0][input_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            else:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_attempts.append({
                "param_name": "pixel_values",
                "success": True,
                "response": response[:500],  # Store more for debugging
                "output_length": outputs.shape[1],
                "new_tokens_count": outputs.shape[1] - input_len,
            })
        except Exception as e:
            generation_attempts.append({
                "param_name": "pixel_values",
                "success": False,
                "error": str(e),
            })

        result["debug_data"]["generation_attempts"] = generation_attempts

        # Find successful attempt
        successful = [a for a in generation_attempts if a.get("success")]
        if successful:
            raw_response = successful[0]["response"]

            # Clean response (matching med3dvlm.py _clean_response)
            cleaned_response = _clean_med3dvlm_response(raw_response)

            result["debug_data"]["response"] = {
                "raw_response": raw_response,
                "cleaned_response": cleaned_response,
                "was_cleaned": raw_response != cleaned_response,
                "response_length": len(cleaned_response),
                "working_param_name": successful[0]["param_name"],
                "output_length": successful[0].get("output_length"),
                "new_tokens_count": successful[0].get("new_tokens_count"),
            }

            # Enhanced error detection (matching med3dvlm.py)
            result["debug_data"]["response_analysis"] = {
                "is_empty": len(cleaned_response.strip()) == 0,
                "is_single_char": len(cleaned_response.strip()) <= 1,
                "is_question_mark": cleaned_response.strip() == "?",
                "is_error_response": cleaned_response.strip() in ["?", "", "."],
                "prompt_valid": prompt_validation.get("valid", False),
            }

            result["status"] = "success"
        else:
            result["debug_data"]["response"] = {
                "error": "No successful generation attempt",
            }
            result["debug_data"]["response_analysis"] = {
                "is_empty": True,
                "is_error_response": True,
                "prompt_valid": prompt_validation.get("valid", False),
            }

        # Cleanup
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def capture_radfm_debug(
    dicom_path: Optional[str],
    question: str = "What anatomical structures are visible in this scan?",
    modality: str = "CT",
    use_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Capture RadFM internals for debugging.

    Returns detailed info about:
    - Model loading from Language_files
    - Tokenizer special token count (should be 3202+ with image tokens)
    - Image tensor shape (should be (B, S, C, H, W, D) with D=depth as LAST dim)
    - Prompt format with <image></image> markers
    - Generation attempt and response
    """
    result = {
        "model": "radfm",
        "status": "error",
        "error": None,
        "debug_data": {},
    }

    try:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "RadFM"

        # Check both possible locations for Language_files
        language_files_paths = [
            model_path / "Language_files",
            project_root / "external" / "RadFM" / "Quick_demo" / "Language_files",
        ]

        language_files_path = None
        for path in language_files_paths:
            if path.exists():
                language_files_path = path
                break

        result["debug_data"]["paths"] = {
            "model_path": str(model_path),
            "model_path_exists": model_path.exists(),
            "language_files_path": str(language_files_path) if language_files_path else None,
            "language_files_exists": language_files_path.exists() if language_files_path else False,
            "pytorch_model_bin_exists": (model_path / "pytorch_model.bin").exists() if model_path.exists() else False,
        }

        if not model_path.exists():
            result["error"] = f"Model path not found at {model_path}"
            return result

        if not language_files_path:
            result["error"] = "Language_files directory not found in any expected location"
            return result

        # Import RadFM's custom tokenizer function
        import sys
        radfm_root = project_root / "external" / "RadFM" / "Quick_demo"
        if str(radfm_root) not in sys.path:
            sys.path.insert(0, str(radfm_root))

        try:
            from transformers import LlamaTokenizer

            # RadFM tokenizer constants
            MAX_IMG_SIZE = 100  # Maximum number of images
            IMAGE_NUM = 32  # Tokens per image

            print("Loading RadFM tokenizer...")
            # Replicate get_tokenizer function from test.py
            image_padding_tokens = []
            tokenizer = LlamaTokenizer.from_pretrained(str(language_files_path))

            special_token = {"additional_special_tokens": ["<image>", "</image>"]}

            # Generate image tokens
            for i in range(MAX_IMG_SIZE):
                image_padding_token = ""
                for j in range(IMAGE_NUM):
                    image_token = f"<image{i * IMAGE_NUM + j}>"
                    image_padding_token += image_token
                    special_token["additional_special_tokens"].append(image_token)
                image_padding_tokens.append(image_padding_token)
                tokenizer.add_special_tokens(special_token)

            # Configure LLaMA special tokens
            tokenizer.pad_token_id = 0
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2

            vocab_size = len(tokenizer)

            result["debug_data"]["tokenizer"] = {
                "vocab_size": vocab_size,
                "expected_min_size": 32000 + (MAX_IMG_SIZE * IMAGE_NUM) + 2,  # LLaMA base + image tokens + markers
                "image_tokens_per_image": IMAGE_NUM,
                "max_images_supported": MAX_IMG_SIZE,
                "has_image_start_token": "<image>" in tokenizer.get_vocab(),
                "has_image_end_token": "</image>" in tokenizer.get_vocab(),
                "image_token_0_id": tokenizer.convert_tokens_to_ids("<image0>") if "<image0>" in tokenizer.get_vocab() else None,
                "first_image_padding": image_padding_tokens[0][:50] + "..." if image_padding_tokens else None,
            }

            # Create image tensor
            # RadFM expects: (B, S, C, H, W, D) where D is DEPTH (last dimension)
            # S = number of images in sequence
            # For 2D images: D=4 (as per test.py)
            # For 3D CT: D can be variable

            if use_synthetic or dicom_path is None:
                print("Using synthetic test volume...")
                # Create synthetic 3D volume: (1, 1, 3, 512, 512, 4)
                # B=1, S=1, C=3 (RGB), H=512, W=512, D=4
                image_tensor = torch.randn(1, 1, 3, 512, 512, 4, dtype=torch.float32)
                result["debug_data"]["preprocessing"] = {"synthetic": True}
            else:
                import SimpleITK as sitk
                from scipy.ndimage import zoom

                print(f"Loading DICOM from {dicom_path}...")
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(dicom_path)

                if not dicom_files:
                    result["error"] = "No DICOM files found"
                    return result

                reader.SetFileNames(dicom_files)
                image = reader.Execute()
                array = sitk.GetArrayFromImage(image)  # (Z, Y, X)

                result["debug_data"]["preprocessing"] = {
                    "dicom_file_count": len(dicom_files),
                    "raw_array_shape": list(array.shape),
                    "raw_spacing": list(image.GetSpacing()),
                }

                # Apply CT windowing if CT
                if modality == "CT":
                    # Soft tissue window
                    window_center = 40
                    window_width = 350
                    min_val = window_center - window_width / 2
                    max_val = window_center + window_width / 2
                    array = np.clip(array, min_val, max_val)
                    array = (array - min_val) / (max_val - min_val)
                else:
                    # MRI: normalize
                    array = (array - array.min()) / (array.max() - array.min() + 1e-8)

                # Resize to 512x512x4 (typical RadFM 2D-like dimensions)
                target_shape = (4, 512, 512)  # (D, H, W)
                if array.shape != target_shape:
                    zoom_factors = tuple(t / c for t, c in zip(target_shape, array.shape))
                    array = zoom(array, zoom_factors, order=1)

                # Convert to RGB by repeating channels: (D, H, W) -> (C, H, W, D)
                # RadFM expects (C, H, W, D) per image
                array = array.transpose(1, 2, 0)  # (H, W, D)
                array = np.stack([array, array, array], axis=0)  # (3, H, W, D)

                # Add batch and sequence dims: (B, S, C, H, W, D)
                image_tensor = torch.from_numpy(array[np.newaxis, np.newaxis, ...]).to(dtype=torch.float32)

            result["debug_data"]["image_tensor"] = {
                "shape": list(image_tensor.shape),
                "expected_format": "[B, S, C, H, W, D] where D=depth (LAST)",
                "expected_shape_example": [1, 1, 3, 512, 512, 4],
                "depth_is_last": True,  # RadFM uses D as last dimension
                "dtype": str(image_tensor.dtype),
                "channels": image_tensor.shape[2] if image_tensor.ndim >= 3 else None,
                "expected_channels": 3,  # RGB
            }

            # Build prompt with image markers
            # Format: <image><image0><image1>...<image31></image>question
            prompt_with_image = f"<image>{image_padding_tokens[0]}</image>{question}"

            result["debug_data"]["prompt_format"] = {
                "format": "<image>[32 image tokens]</image>question",
                "image_tokens_count": IMAGE_NUM,
                "prompt_sample": prompt_with_image[:100] + "...",
                "prompt_length": len(prompt_with_image),
            }

            # Try to load model and generate
            try:
                from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

                print("Loading RadFM model...")
                model = MultiLLaMAForCausalLM(
                    lang_model_path=str(language_files_path),
                )

                # Try to load checkpoint
                ckpt_path = model_path / "pytorch_model.bin"
                if ckpt_path.exists():
                    print(f"Loading checkpoint from {ckpt_path}...")
                    ckpt = torch.load(str(ckpt_path), map_location="cpu")
                    model.load_state_dict(ckpt)
                    result["debug_data"]["model_loading"] = {
                        "checkpoint_loaded": True,
                        "checkpoint_path": str(ckpt_path),
                    }
                else:
                    result["debug_data"]["model_loading"] = {
                        "checkpoint_loaded": False,
                        "warning": f"Checkpoint not found at {ckpt_path}",
                    }

                model = model.to("cuda")
                model.eval()

                # Tokenize prompt
                print("Running generation...")
                lang_x = tokenizer(
                    prompt_with_image,
                    max_length=2048,
                    truncation=True,
                    return_tensors="pt"
                )['input_ids'].to('cuda')

                vision_x = image_tensor.to('cuda')

                input_len = lang_x.shape[1]

                # Generate (using positional args like test.py)
                with torch.no_grad():
                    generation = model.generate(lang_x, vision_x)

                # Decode response
                response = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

                result["debug_data"]["generation"] = {
                    "input_length": input_len,
                    "output_length": generation.shape[1],
                    "new_tokens_count": generation.shape[1] - input_len,
                    "generation_method": "model.generate(lang_x, vision_x)",  # Positional args
                }

                result["debug_data"]["response"] = {
                    "raw_response": response.strip(),
                    "response_length": len(response),
                    "is_empty": len(response.strip()) == 0,
                }

                # Cleanup
                del model
                torch.cuda.empty_cache()

                result["status"] = "success"

            except ImportError as e:
                result["debug_data"]["model_loading"] = {
                    "error": f"Failed to import RadFM model: {str(e)}",
                    "suggestion": "Ensure RadFM repository is cloned to external/RadFM",
                }
                result["status"] = "partial_success"  # Tokenizer worked

        finally:
            if str(radfm_root) in sys.path:
                sys.path.remove(str(radfm_root))

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Cloud Debug Capture Script")
    parser.add_argument(
        "--dicom-path",
        type=str,
        default=None,
        help="Path to DICOM series directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use synthetic test data instead of real DICOM",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="m3d-lamed,vila-m3,med3dvlm,radfm",
        help="Comma-separated list of models to test (default: all)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="CT",
        choices=["CT", "MRI"],
        help="Image modality",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What anatomical structures are visible in this scan?",
        help="Question to ask the model",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"debug_capture_{timestamp}.json")

    models_to_test = [m.strip().lower() for m in args.models.split(",")]
    use_synthetic = args.test or args.dicom_path is None

    print("=" * 60)
    print("Cloud Debug Capture Script")
    print("=" * 60)
    print(f"Models: {models_to_test}")
    print(f"DICOM path: {args.dicom_path or '(synthetic)'}")
    print(f"Modality: {args.modality}")
    print(f"Question: {args.question}")
    print(f"Output: {output_path}")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dicom_path": args.dicom_path,
            "modality": args.modality,
            "question": args.question,
            "use_synthetic": use_synthetic,
        },
        "models": {},
    }

    # Run captures
    model_count = len(models_to_test)
    current = 0

    if "m3d-lamed" in models_to_test:
        current += 1
        print(f"\n[{current}/{model_count}] Capturing M3D-LaMed debug data...")
        results["models"]["m3d-lamed"] = capture_m3d_lamed_debug(
            args.dicom_path, args.question, args.modality, use_synthetic
        )

    if "vila-m3" in models_to_test:
        current += 1
        print(f"\n[{current}/{model_count}] Capturing VILA-M3 debug data...")
        results["models"]["vila-m3"] = capture_vila_m3_debug(
            args.dicom_path, args.question, args.modality, use_synthetic
        )

    if "med3dvlm" in models_to_test:
        current += 1
        print(f"\n[{current}/{model_count}] Capturing Med3DVLM debug data...")
        results["models"]["med3dvlm"] = capture_med3dvlm_debug(
            args.dicom_path, args.question, args.modality, use_synthetic
        )

    if "radfm" in models_to_test:
        current += 1
        print(f"\n[{current}/{model_count}] Capturing RadFM debug data...")
        results["models"]["radfm"] = capture_radfm_debug(
            args.dicom_path, args.question, args.modality, use_synthetic
        )

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)

    # Print summary
    for model_name, model_result in results["models"].items():
        status = model_result.get("status", "unknown")
        print(f"\n{model_name}: {status.upper()}")

        if status == "success" and "debug_data" in model_result:
            debug = model_result["debug_data"]

            if model_name == "m3d-lamed":
                pos_data = debug.get("token_positions", {})
                print(f"  - <im_patch> positions correct: {pos_data.get('im_patch_positions_correct', 'N/A')}")
                print(f"  - BOS at position 0: {pos_data.get('bos_at_position_0', 'N/A')}")
                print(f"  - Chat template used: {pos_data.get('chat_template_used', 'N/A')}")

                resp_data = debug.get("response", {})
                print(f"  - Starts with 'Sure': {resp_data.get('starts_with_sure', 'N/A')}")
                print(f"  - Contains coordinates: {resp_data.get('contains_bounding_box', 'N/A')}")

            elif model_name == "vila-m3":
                resp_data = debug.get("response", {})
                print(f"  - Contains VISTA3D labels: {resp_data.get('contains_vista3d', 'N/A')}")
                print(f"  - Modality confusion: {resp_data.get('modality_confusion', 'N/A')}")

                expert_data = debug.get("expert_integration", {})
                print(f"  - Has color labels: {expert_data.get('response_has_color_labels', 'N/A')}")

            elif model_name == "med3dvlm":
                tensor_data = debug.get("image_tensor", {})
                print(f"  - Shape correct (128,256,256): {tensor_data.get('shape_correct', 'N/A')}")

                resp_data = debug.get("response", {})
                print(f"  - Response is empty: {resp_data.get('is_empty', 'N/A')}")
                print(f"  - Working param name: {resp_data.get('working_param_name', 'N/A')}")

            elif model_name == "radfm":
                paths_data = debug.get("paths", {})
                print(f"  - Language_files found: {paths_data.get('language_files_exists', 'N/A')}")
                print(f"  - pytorch_model.bin exists: {paths_data.get('pytorch_model_bin_exists', 'N/A')}")

                tokenizer_data = debug.get("tokenizer", {})
                if tokenizer_data:
                    print(f"  - Vocab size: {tokenizer_data.get('vocab_size', 'N/A')}")
                    print(f"  - Image tokens per image: {tokenizer_data.get('image_tokens_per_image', 'N/A')}")

                tensor_data = debug.get("image_tensor", {})
                if tensor_data:
                    print(f"  - Tensor shape: {tensor_data.get('shape', 'N/A')}")
                    print(f"  - Depth is last dim: {tensor_data.get('depth_is_last', 'N/A')}")

                resp_data = debug.get("response", {})
                if resp_data:
                    print(f"  - Response is empty: {resp_data.get('is_empty', 'N/A')}")

        elif model_result.get("error"):
            print(f"  - Error: {model_result['error'][:100]}...")

    print(f"\nResults saved to: {output_path}")
    print("Download this file for local analysis.")


if __name__ == "__main__":
    main()
