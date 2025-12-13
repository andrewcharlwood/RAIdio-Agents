"""
DICOM Preprocessing Pipeline for M3D-LaMed

Converts DICOM series to normalised tensors suitable for M3D-LaMed input.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
import SimpleITK as sitk


def group_dicom_by_dimensions(dicom_dir: str) -> Dict[Tuple[int, int], List[str]]:
    """
    Group DICOM files by their (Rows, Columns) dimensions.

    Args:
        dicom_dir: Path to directory containing DICOM files

    Returns:
        Dict mapping (rows, cols) to list of file paths.
        Only includes groups with >1 slice (valid volumes).
    """
    dicom_path = Path(dicom_dir)
    groups = defaultdict(list)

    for filepath in sorted(dicom_path.iterdir()):
        if not filepath.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True, force=True)
            if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                dims = (int(ds.Rows), int(ds.Columns))
                groups[dims].append(str(filepath))
        except Exception:
            continue

    # Filter to groups with >1 slice (valid volumes)
    return {dims: files for dims, files in groups.items() if len(files) > 1}


class DICOMPreprocessor:
    """
    Preprocessor for converting DICOM series to normalised tensors.

    Handles both CT and MRI modalities with appropriate windowing/normalisation.
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Optional[Tuple[int, int, int]] = (32, 256, 256),
        modality: str = "CT"
    ):
        """
        Initialise the preprocessor.

        Args:
            target_spacing: Isotropic voxel spacing in mm (D, H, W)
            target_size: Output volume dimensions (D, H, W), or None to skip resizing
            modality: Either "CT" or "MRI"
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.modality = modality.upper()

        if self.modality not in ("CT", "MRI"):
            raise ValueError(f"Modality must be 'CT' or 'MRI', got '{modality}'")

    def load_dicom_files(self, file_list: List[str]) -> sitk.Image:
        """
        Load specific DICOM files as a series.

        Args:
            file_list: List of DICOM file paths to load

        Returns:
            SimpleITK Image object
        """
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_list)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        return reader.Execute()

    def load_dicom_series(
        self, dicom_dir: str
    ) -> Union[sitk.Image, List[Tuple[str, sitk.Image]]]:
        """
        Load a DICOM series from a directory.

        If mixed dimensions are detected, returns separate volumes for each
        dimension group (for series with >1 slice per group).

        Args:
            dicom_dir: Path to directory containing DICOM files

        Returns:
            Either a single SimpleITK Image, or a list of (label, image) tuples
            if mixed dimensions are detected.

        Raises:
            ValueError: If no DICOM series found in directory
        """
        dicom_path = Path(dicom_dir)
        if not dicom_path.exists():
            raise ValueError(f"DICOM directory does not exist: {dicom_dir}")

        reader = sitk.ImageSeriesReader()

        # Get DICOM file names (properly sorted by GDCM for slice ordering)
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))

        if not dicom_names:
            raise ValueError(f"No DICOM series found in directory: {dicom_dir}")

        # Check for mixed dimensions (groups with >1 slice)
        groups = group_dicom_by_dimensions(dicom_dir)

        if len(groups) == 0:
            # No valid multi-slice groups - try loading all files
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            return reader.Execute()

        elif len(groups) == 1:
            # Single dimension group - filter to only files matching this dimension
            # This excludes scout/localizer images with different dimensions
            dims, valid_files = list(groups.items())[0]

            # Normalise paths for comparison (handle mixed path separators)
            valid_files_normalised = {str(Path(f)) for f in valid_files}

            # Filter GDCM-sorted list to preserve proper slice ordering
            filtered_names = [f for f in dicom_names if str(Path(f)) in valid_files_normalised]

            if len(filtered_names) < len(dicom_names):
                excluded = len(dicom_names) - len(filtered_names)
                print(f"  Filtered out {excluded} file(s) with mismatched dimensions")

            reader.SetFileNames(filtered_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            return reader.Execute()

        else:
            # Multiple dimension groups - return list of volumes
            print(f"  Found {len(groups)} dimension groups, processing separately...")
            volumes = []
            for dims, files in sorted(groups.items(), key=lambda x: -len(x[1])):
                label = f"{dims[1]}x{dims[0]}"  # WxH format
                print(f"    - {label}: {len(files)} slices")

                # Normalise paths and sort files by GDCM order for proper slice ordering
                files_normalised = {str(Path(f)) for f in files}
                sorted_files = [f for f in dicom_names if str(Path(f)) in files_normalised]
                image = self.load_dicom_files(sorted_files)
                volumes.append((label, image))

            return volumes

    def resample_to_isotropic(self, image: sitk.Image) -> sitk.Image:
        """
        Resample image to isotropic spacing.

        Args:
            image: Input SimpleITK Image

        Returns:
            Resampled SimpleITK Image with target spacing
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Calculate new size based on spacing change
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, self.target_spacing)
        ]

        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))

        return resampler.Execute(image)

    def resize_volume(self, image: sitk.Image) -> sitk.Image:
        """
        Resize volume to target dimensions.

        Args:
            image: Input SimpleITK Image

        Returns:
            Resized SimpleITK Image
        """
        if self.target_size is None:
            return image

        original_size = image.GetSize()  # (X, Y, Z) = (W, H, D)
        original_spacing = image.GetSpacing()

        # Convert target_size from (D, H, W) to SimpleITK's (X, Y, Z) = (W, H, D)
        sitk_target_size = (self.target_size[2], self.target_size[1], self.target_size[0])

        # Calculate new spacing to achieve target size
        new_spacing = [
            ospc * osz / tsz
            for ospc, osz, tsz in zip(original_spacing, original_size, sitk_target_size)
        ]

        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(sitk_target_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))

        return resampler.Execute(image)

    def apply_ct_windowing(self, image_array: np.ndarray) -> np.ndarray:
        """
        Apply CT windowing for single-channel output.

        Uses soft tissue window for general head/neck imaging.

        Args:
            image_array: Input CT volume as numpy array (D, H, W)

        Returns:
            Single-channel normalised array (1, D, H, W) in range [0, 1]
        """
        # Soft tissue window for general purpose (also good for head/neck)
        ww, wl = 350, 40

        # Calculate window bounds
        lower = wl - ww / 2
        upper = wl + ww / 2

        # Apply windowing
        windowed = np.clip(image_array, lower, upper)

        # Normalise to [0, 1]
        normalised = (windowed - lower) / (upper - lower)

        # Add channel dimension: (1, D, H, W)
        output = normalised[np.newaxis, ...].astype(np.float32)

        return output

    def normalise_mri(self, image_array: np.ndarray) -> np.ndarray:
        """
        Normalise MRI volume.

        Applies percentile clipping and z-score normalisation.

        Args:
            image_array: Input MRI volume as numpy array (D, H, W)

        Returns:
            Normalised array (1, D, H, W) in range [0, 1]
        """
        # Clip to 1st-99th percentile
        p1, p99 = np.percentile(image_array, [1, 99])
        clipped = np.clip(image_array, p1, p99)

        # Z-score normalisation
        mean = np.mean(clipped)
        std = np.std(clipped)
        if std > 0:
            normalised = (clipped - mean) / std
        else:
            normalised = clipped - mean

        # Scale to [0, 1]
        min_val = normalised.min()
        max_val = normalised.max()
        if max_val > min_val:
            scaled = (normalised - min_val) / (max_val - min_val)
        else:
            scaled = np.zeros_like(normalised)

        # Add channel dimension: (1, D, H, W)
        output = scaled[np.newaxis, ...].astype(np.float32)

        return output

    def _process_single_volume(self, image: sitk.Image) -> np.ndarray:
        """
        Process a single SimpleITK image through the preprocessing pipeline.

        Args:
            image: SimpleITK Image object

        Returns:
            Preprocessed array (1, D, H, W)
        """
        # Resample to isotropic spacing
        image = self.resample_to_isotropic(image)

        # Resize to target dimensions
        image = self.resize_volume(image)

        # Convert to numpy array
        image_array = sitk.GetArrayFromImage(image)

        # Apply modality-specific preprocessing
        if self.modality == "CT":
            return self.apply_ct_windowing(image_array)
        else:  # MRI
            return self.normalise_mri(image_array)

    def process(
        self, dicom_dir: str
    ) -> Union[np.ndarray, List[Tuple[str, np.ndarray]]]:
        """
        Main preprocessing pipeline.

        Args:
            dicom_dir: Path to DICOM directory

        Returns:
            Either a single preprocessed array (1, D, H, W), or a list of
            (label, array) tuples if mixed dimensions were detected.
        """
        # Step 1: Load DICOM series (may return multiple volumes)
        result = self.load_dicom_series(dicom_dir)

        if isinstance(result, list):
            # Multiple volumes from mixed-dimension series
            processed = []
            for label, image in result:
                output = self._process_single_volume(image)
                processed.append((label, output))
            return processed
        else:
            # Single volume - standard processing
            return self._process_single_volume(result)


def preprocess_for_m3d(
    dicom_dir: str,
    modality: str = "CT",
    target_size: Optional[Tuple[int, int, int]] = (32, 256, 256),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Union[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    Convenience function to preprocess a DICOM series for M3D-LaMed.

    Args:
        dicom_dir: Path to DICOM directory
        modality: "CT" or "MRI"
        target_size: Output volume dimensions (D, H, W)
        target_spacing: Target voxel spacing in mm

    Returns:
        Either a single preprocessed array (1, D, H, W), or a list of
        (label, array) tuples if mixed dimensions were detected in the series.
    """
    preprocessor = DICOMPreprocessor(
        target_spacing=target_spacing,
        target_size=target_size,
        modality=modality
    )
    return preprocessor.process(dicom_dir)
