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
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<image>{question}"},
                    {"type": "image_path", "image_path": image_path},
                ]
            }
        ]

        result["debug_data"]["message_format"] = {
            "messages": str(messages),
            "modality_in_prompt": modality in question,
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

        # Tokenize question
        inputs = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda")

        result["debug_data"]["tokenizer"] = {
            "input_length": inputs.input_ids.shape[1],
            "input_ids_sample": inputs.input_ids[0][:20].tolist(),
        }

        # Run generation
        print("Running Med3DVLM generation...")

        # Try different parameter names
        generation_attempts = []

        # Attempt 1: Using 'images' parameter
        try:
            with torch.no_grad():
                outputs = model.generate(
                    images=image_tensor,
                    inputs=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=256,
                    temperature=1.0,
                    do_sample=True,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_attempts.append({
                "param_name": "images",
                "success": True,
                "response": response[:200],
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
                    max_new_tokens=256,
                    temperature=1.0,
                    do_sample=True,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_attempts.append({
                "param_name": "pixel_values",
                "success": True,
                "response": response[:200],
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
            response = successful[0]["response"]
            result["debug_data"]["response"] = {
                "raw_response": response,
                "response_length": len(response),
                "is_empty": len(response.strip()) == 0,
                "is_question_mark": response.strip() == "?",
                "working_param_name": successful[0]["param_name"],
            }
            result["status"] = "success"
        else:
            result["debug_data"]["response"] = {
                "error": "No successful generation attempt",
            }

        # Cleanup
        del model
        torch.cuda.empty_cache()

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
        default="m3d-lamed,vila-m3,med3dvlm",
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
    if "m3d-lamed" in models_to_test:
        print("\n[1/3] Capturing M3D-LaMed debug data...")
        results["models"]["m3d-lamed"] = capture_m3d_lamed_debug(
            args.dicom_path, args.question, args.modality, use_synthetic
        )

    if "vila-m3" in models_to_test:
        print("\n[2/3] Capturing VILA-M3 debug data...")
        results["models"]["vila-m3"] = capture_vila_m3_debug(
            args.dicom_path, args.question, args.modality, use_synthetic
        )

    if "med3dvlm" in models_to_test:
        print("\n[3/3] Capturing Med3DVLM debug data...")
        results["models"]["med3dvlm"] = capture_med3dvlm_debug(
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

        elif model_result.get("error"):
            print(f"  - Error: {model_result['error'][:100]}...")

    print(f"\nResults saved to: {output_path}")
    print("Download this file for local analysis.")


if __name__ == "__main__":
    main()
