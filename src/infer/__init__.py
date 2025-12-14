"""
Model Registry and Factory for 3D Medical Image Models

Provides unified access to all supported models through a registry pattern.
"""

from typing import Dict, List, Optional, Type

from .base import Medical3DModel

# Registry of available models
_MODEL_REGISTRY: Dict[str, Type[Medical3DModel]] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the registry.

    Usage:
        @register_model("m3d-lamed")
        class M3DLaMedModel(Medical3DModel):
            ...
    """
    def decorator(cls: Type[Medical3DModel]):
        _MODEL_REGISTRY[name.lower()] = cls
        cls.name = name.lower()
        return cls
    return decorator


def get_model(
    name: str,
    model_path: Optional[str] = None,
    **kwargs
) -> Medical3DModel:
    """
    Factory function to get a model instance by name.

    Args:
        name: Model name (e.g., "m3d-lamed", "med3dvlm", "radfm", "ct-clip")
        model_path: Optional path to model files (uses default if not specified)
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available or 'none registered'}"
        )

    model_class = _MODEL_REGISTRY[name_lower]
    return model_class(model_path=model_path, **kwargs)


def list_models() -> List[str]:
    """
    List all registered model names.

    Returns:
        List of model names
    """
    return sorted(_MODEL_REGISTRY.keys())


def list_vqa_models() -> List[str]:
    """
    List registered VQA models.

    Returns:
        List of VQA model names
    """
    return sorted([
        name for name, cls in _MODEL_REGISTRY.items()
        if cls.model_type == "vqa"
    ])


def list_classifier_models() -> List[str]:
    """
    List registered classifier models.

    Returns:
        List of classifier model names
    """
    return sorted([
        name for name, cls in _MODEL_REGISTRY.items()
        if cls.model_type == "classifier"
    ])


def get_model_info(name: str) -> Dict:
    """
    Get information about a registered model.

    Args:
        name: Model name

    Returns:
        Dict with model info (name, type, input_shape, channels)

    Raises:
        ValueError: If model name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available or 'none registered'}"
        )

    cls = _MODEL_REGISTRY[name_lower]

    # Create temporary instance to get shape info
    # (don't load model, just get metadata)
    temp = cls.__new__(cls)
    temp.model_path = cls.get_default_model_path()

    return {
        "name": name_lower,
        "class": cls.__name__,
        "type": cls.model_type,
        "input_shape": temp.get_input_shape() if hasattr(temp, 'get_input_shape') else None,
        "channels": temp.get_channels() if hasattr(temp, 'get_channels') else None,
        "default_path": str(cls.get_default_model_path()),
    }


# Import model implementations to trigger registration
# These imports must come after the registry is defined
try:
    from .m3d_lamed import M3DLaMedModel
except ImportError as e:
    print(f"Warning: Could not import M3D-LaMed model: {e}")

try:
    from .med3dvlm import Med3DVLMModel
except ImportError as e:
    print(f"Warning: Could not import Med3DVLM model: {e}")

try:
    from .radfm import RadFMModel
except ImportError as e:
    print(f"Warning: Could not import RadFM model: {e}")

try:
    from .ct_clip import CTCLIPModel
except ImportError as e:
    print(f"Warning: Could not import CT-CLIP model: {e}")

try:
    from .vila_m3 import VILAm3Model, VILAm3Model3B, VILAm3Model13B
except ImportError as e:
    print(f"Warning: Could not import VILA-M3 model: {e}")


__all__ = [
    "Medical3DModel",
    "register_model",
    "get_model",
    "list_models",
    "list_vqa_models",
    "list_classifier_models",
    "get_model_info",
]
