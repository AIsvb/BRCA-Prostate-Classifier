import os
import math
import logging
from typing import Union
import numpy as np
from pathlib import Path
from src.utils.slideloader import SlideLoader
from slidesegmenter import SlideSegmenter
from slidesegmenter._model_utils import ModifiedUNet
import torch
from PIL import Image
import cv2
from contextlib import contextmanager
from typing import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_slide(
    slide_path: Path,
    output_folder: Path,
    model_a_checkpoint: Path,
    model_b_checkpoint: Path,
    model_settings: Path,
    magnification: float,
    target_magnification: float
) -> None:
    """Process a single slide and save the result."""
    output_path = output_folder / slide_path.with_suffix('.png').name
    if output_path.is_file():
        logger.debug(f"Skipping existing file: {output_path}")
        return

    logger.info(f"Processing slide: {slide_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize SlideLoader and segmenters per process
    try:
        loader = SlideLoader({'multithreading': True, 'progress_bar': False})
        segmenter_a = SlideSegmenter(
            tissue_segmentation=True,
            pen_marking_segmentation=False,
            separate_cross_sections=False,
            device=device
        )
        segmenter_a._load_model(ModifiedUNet, str(model_a_checkpoint), str(model_settings))
        segmenter_b = SlideSegmenter(
            tissue_segmentation=True,
            pen_marking_segmentation=False,
            separate_cross_sections=False,
            device=device
        )
        segmenter_b._load_model(ModifiedUNet, str(model_b_checkpoint), str(model_settings))
    except Exception as e:
        logger.error(f"Failed to initialize tools for {slide_path}: {e}")
        return

    try:
        # Load slide and image
        with loader_context(loader, slide_path) as slide_loader:
            image = slide_loader.get_image(magnification=magnification)
            true_height, true_width = slide_loader.get_dimensions(magnification * (target_magnification / magnification))

        # Crop image
        image = crop_to_true_dimensions(image, true_height, true_width, target_magnification / magnification)

        # Segment the image
        segmentation_a = segmenter_a.segment(image / 255.0)
        segmentation_b = segmenter_b.segment(image / 255.0)
        intersection = compute_intersection(segmentation_a, segmentation_b)

        # Apply Otsu's thresholding
        otsu = apply_otsu_threshold(image)

        # Combine and save
        combined_image = np.vstack([image, intersection, otsu])
        Image.fromarray(combined_image).save(output_path)
        logger.info(f"Saved: {output_path}")

        return slide_path, output_path
    
    except Exception as e:
        logger.error(f"Error processing {slide_path}: {e}")

def process_slides(
    input_folder: Union[str, Path],
    model_a_checkpoint: Union[str, Path],
    model_b_checkpoint: Union[str, Path],
    model_settings: Union[str, Path],
    output_folder: Union[str, Path],
    magnification: float = 0.625,
    target_magnification: float = 20.0,
    glob_pattern: str = '*.ndpi',
    max_workers: int = None
) -> None:
    """
    Process NDPI slides in parallel using two segmenters and save segmentation results as PNG images.

    Args:
        input_folder (Union[str, Path]): Path to the folder containing NDPI slides.
        model_a_checkpoint (Union[str, Path]): Path to the first segmentation model checkpoint.
        model_b_checkpoint (Union[str, Path]): Path to the second segmentation model checkpoint.
        model_settings (Union[str, Path]): Path to the model settings file.
        output_folder (Union[str, Path]): Path to the folder where processed images will be saved.
        magnification (float, optional): Magnification level for loading images. Defaults to 0.625.
        target_magnification (float, optional): Target magnification for ratio calculation. Defaults to 20.0.
        glob_pattern (str, optional): Glob pattern for slide files (e.g., '*.ndpi', '*.tif'). Defaults to '*.ndpi'.
        max_workers (int, optional): Number of parallel workers. Defaults to None (uses CPU count).

    Raises:
        FileNotFoundError: If input folder or model files do not exist.
        ValueError: If magnification values are invalid or glob pattern is empty.
    """
    # Input validation
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    model_a_checkpoint = Path(model_a_checkpoint)
    model_b_checkpoint = Path(model_b_checkpoint)
    model_settings = Path(model_settings)
    process_list = output_folder / 'processed_slides.txt'

    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_folder}")
    for model_file in (model_a_checkpoint, model_b_checkpoint, model_settings):
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
    if magnification <= 0 or target_magnification <= 0:
        raise ValueError("Magnification and target_magnification must be positive")
    if not glob_pattern or not isinstance(glob_pattern, str):
        raise ValueError("glob_pattern must be a non-empty string")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Collect all slide paths
    slide_paths = list(input_folder.rglob(glob_pattern))
    if not slide_paths:
        logger.warning(f"No slides found in {input_folder} with pattern {glob_pattern}")
        return

    logger.info(f"Found {len(slide_paths)} slides to process")

    # Process slides in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_slide,
                slide_path,
                output_folder,
                model_a_checkpoint,
                model_b_checkpoint,
                model_settings,
                magnification,
                target_magnification
            ): slide_path for slide_path in slide_paths
        }
        for future in as_completed(futures):
            slide_path = futures[future]
            try:
                result = future.result()  # Raises exception if the process failed
                mode = 'a' if process_list.is_file() else 'w'
                with open(process_list, mode) as file_:
                    file_.write(f'{result[0].stem},{result[0]},{result[1]}\n')

            except Exception as e:
                logger.error(f"Failed to process {slide_path}: {e}")

@contextmanager
def loader_context(loader: SlideLoader, slide_path: Path) -> Iterator[SlideLoader]:
    """Context manager for SlideLoader to ensure proper resource cleanup."""
    try:
        loader.load_slide(slide_path)
        yield loader
    finally:
        loader.close()

def crop_to_true_dimensions(
    image: np.ndarray, true_height: int, true_width: int, ratio: float
) -> np.ndarray:
    """Crop image to match true slide dimensions based on magnification ratio."""
    height, width, _ = image.shape
    crop_height = math.ceil(max(0, height * ratio - true_height) / ratio)
    crop_width = math.ceil(max(0, width * ratio - true_width) / ratio)
    if crop_height > 0 or crop_width > 0:
        return image[:height - crop_height, :width - crop_width, :]
    return image

def compute_intersection(segmentation_a: np.ndarray, segmentation_b: np.ndarray) -> np.ndarray:
    """Compute the intersection of two segmentation masks and format as RGB."""
    intersection = np.logical_and(segmentation_a, segmentation_b) * 255
    return np.concatenate([intersection] * 3, axis=-1).astype(np.uint8)

def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu's thresholding to the saturation channel of the image."""
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)
    _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
    return np.repeat(img_otsu[:, :, np.newaxis], 3, axis=2)
