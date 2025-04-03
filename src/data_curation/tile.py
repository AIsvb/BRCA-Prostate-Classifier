from pathlib import Path
import logging
from multiprocessing import Pool
from functools import partial
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from src.utils.slideloader import SlideLoader
from typing import Callable, Optional, Union, Dict, List, Tuple
from torch.nn.functional import conv2d
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import rescale
import h5py
import matplotlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("tiling.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_hdf5(output_path: Union[str, Path], asset_dict: Dict, attr_dict: Optional[Dict] = None, mode: str = 'a', chunk_size: int = 32) -> None:
    """
    Save assets to an HDF5 file with optional attributes.

    Args:
        output_path: Path to the HDF5 file.
        asset_dict: Dictionary of group names to datasets (key-value pairs of arrays).
        attr_dict: Optional dictionary of attributes to store at the file level.
        mode: File mode ('a' for append, 'w' for write).
        chunk_size: Chunk size for resizable datasets (e.g., 'coords', 'features').
    """
    try:
        with h5py.File(Path(output_path), mode) as file:
            for group_name, dictionary in asset_dict.items():
                group = file.require_group(group_name)  # Creates or gets group
                for key, array in dictionary.items():
                    array = np.asarray(array)  # Ensure NumPy array
                    data_shape = array.shape
                    if key not in group:
                        if key in ('coords', 'features'):
                            chunk_shape = (min(chunk_size, data_shape[0]),) + data_shape[1:]
                            maxshape = (None,) + data_shape[1:]
                            dset = group.create_dataset(
                                key, data=array, maxshape=maxshape, chunks=chunk_shape,
                                compression='gzip', compression_opts=4
                            )
                        else:
                            group.create_dataset(key, data=array, compression='gzip')
                    elif mode == 'a' and key in ('coords', 'features'):
                        dset = group[key]
                        new_size = dset.shape[0] + data_shape[0]
                        dset.resize(new_size, axis=0)
                        dset[-data_shape[0]:] = array

            if attr_dict:
                file.attrs.update(attr_dict)
    except Exception as e:
        logger.error(f"Failed to save HDF5 file {output_path}: {str(e)}")
        raise

def combine(
    tissue_segmentation: np.ndarray,
    pen_marking_segmentation: np.ndarray,
    tissue_color: Tuple[float, float, float] = (1, 1, 1),
    pen_color: Tuple[float, float, float] = (0, 0, 1),
    background_color: Tuple[float, float, float] = (0, 0, 0)
) -> np.ndarray:
    """
    Combine tissue and pen marking segmentations into a single RGB image.

    Args:
        tissue_segmentation: Tissue segmentation (H, W) or (H, W, C).
        pen_marking_segmentation: Pen marking segmentation (H, W) or (H, W, C).
        tissue_color: RGB color for tissue regions.
        pen_color: RGB color for pen marking regions.
        background_color: RGB color for background.

    Returns:
        Combined segmentation as (H, W, 3).

    Raises:
        ValueError: If segmentations differ in size.
    """
    tissue_seg = tissue_segmentation.sum(axis=2) if tissue_segmentation.ndim == 3 else tissue_segmentation
    pen_seg = pen_marking_segmentation.sum(axis=2) if pen_marking_segmentation.ndim == 3 else pen_marking_segmentation

    if tissue_seg.shape != pen_seg.shape:
        raise ValueError("Tissue and pen marking segmentations must have the same dimensions.")

    channels = [
        np.where(pen_seg > 0.5, pen_val, np.where(tissue_seg > 0.5, tissue_val, bg_val))
        for tissue_val, pen_val, bg_val in zip(tissue_color, pen_color, background_color)
    ]
    return np.stack(channels, axis=-1)

def visualize_tiling(
    images: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
    tile_information: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int], int]]],
    output_path: Union[str, Path],
    line_color: Union[str, Tuple[int, int, int]] = (255, 0, 0),
    line_width: int = 2,
    downscale_factor: float = 1.0,
    axis: int = 0,
    enum_tiles: bool = False
) -> None:
    """
    Visualize tile outlines on images and save the result.

    Args:
        images: Single image or list of images (H, W, C) or torch.Tensor.
        tile_information: Dict of cross-section to list of (position, location, shape).
        output_path: Path to save the output image.
        line_color: Color of tile outlines or a matplotlib colormap name.
        line_width: Width of tile outlines.
        downscale_factor: Factor to downscale images (< 1.0 means shrink).
        axis: Axis for concatenating multiple images (0=vertical, 1=horizontal).
        enum_tiles: Whether to label tiles with indices.

    Raises:
        ValueError: If images are empty or have mismatched dimensions.
        TypeError: If image type is invalid.
    """
    # Normalize input to list of NumPy arrays
    if not isinstance(images, (list, tuple)):
        images = [images]
    images = [img.numpy() if isinstance(img, torch.Tensor) else img for img in images]
    if not images or not all(isinstance(img, np.ndarray) for img in images):
        raise TypeError("Images must be non-empty list of NumPy arrays or tensors.")
    if len({img.shape[:2] for img in images}) > 1:
        raise ValueError("All images must have the same spatial dimensions.")

    # Downscale images
    downscale_factor = min(1.0, downscale_factor)  # Ensure <= 1.0
    images = [rescale(img, downscale_factor, channel_axis=-1) * 255 for img in images]
    image = np.concatenate(images, axis=axis) if len(images) > 1 else images[0]
    image = image.astype(np.uint8)

    # Handle line color
    use_cmap = line_color in matplotlib.colormaps
    cmap = matplotlib.colormaps[line_color] if use_cmap else None

    # Draw tiles
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    n_sections = len(tile_information)
    dims = images[0].shape[:2]

    for i, tiles in enumerate(tile_information.values()):
        color = tuple(int(c * 255) for c in cmap(i / n_sections)[:3]) if use_cmap else line_color
        for j, (_, (x, y), shape) in enumerate(tiles):
            for n in range(len(images)):
                top_left = (
                    int((x + n * dims[axis] * (axis == 1)) * downscale_factor),
                    int((y + n * dims[axis] * (axis == 0)) * downscale_factor)
                )
                bottom_right = (
                    top_left[0] + int(shape * downscale_factor),
                    top_left[1] + int(shape * downscale_factor)
                )
                draw.rectangle([top_left, bottom_right], outline=color, width=line_width)
                if enum_tiles:
                    font_size = int(shape * downscale_factor / 2)
                    font = ImageFont.load_default() if font_size < 10 else ImageFont.truetype('calibri.ttf', font_size)
                    draw.text(
                        (top_left[0] + shape * downscale_factor // 4, top_left[1] + shape * downscale_factor // 4),
                        str(j), fill=(0, 150, 50), font=font
                    )

    try:
        pil_image.save(Path(output_path))
    except Exception as e:
        logger.error(f"Failed to save visualization {output_path}: {str(e)}")
        raise

def get_bounding_box(array: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return bounding box (rmin, rmax, cmin, cmax) around non-zero elements, or None if empty."""
    if array.ndim > 2:
        array = array.any(axis=-1)
    rows, cols = np.nonzero(array)
    return (rows.min(), rows.max(), cols.min(), cols.max()) if rows.size else None

def compute_extraction_region(
    crop: np.ndarray,
    tile_shape: int,
    stride: int,
    min_tissue_fraction: float = 0.0,
    exclusion_crop: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute binary mask of valid tile regions."""
    filter = torch.ones(1, 1, tile_shape, tile_shape) / (tile_shape ** 2)
    crop_tensor = torch.from_numpy(crop[None, None, ...].astype(np.float32))
    filtered = conv2d(crop_tensor, filter, stride=stride, padding='valid')[0, 0].numpy()
    mask = filtered >= min_tissue_fraction

    if exclusion_crop is not None:
        excl_tensor = torch.from_numpy(exclusion_crop[None, None, ...].astype(np.float32))
        excl_filtered = conv2d(excl_tensor, filter, stride=stride, padding='valid')[0, 0].numpy()
        mask &= excl_filtered == 0

    return mask

def tile(
    segmentation: Union[np.ndarray, torch.Tensor],
    shape: int,
    stride: Optional[int] = None,
    min_tissue_fraction: float = 0.001,
    preprocessing_function: Optional[Callable] = None,
    exclusion_map: Optional[Union[np.ndarray, torch.Tensor]] = None,
    exceed_image: bool = False
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Extract tile coordinates and shapes from a segmentation mask.

    Args:
        segmentation: Binary segmentation (H, W, C).
        shape: Tile size in pixels.
        stride: Step size between tiles (defaults to shape).
        min_tissue_fraction: Minimum tissue fraction for a tile to be included.
        preprocessing_function: Optional preprocessing for segmentation.
        exclusion_map: Optional mask of regions to exclude (H, W) or (H, W, 1).
        exceed_image: Allow tiles to extend beyond image boundaries.

    Returns:
        Tuple of (tile_information, crops, exclusion_crops, top_lefts, outsides).
    """
    segmentation = segmentation.numpy() if isinstance(segmentation, torch.Tensor) else segmentation
    if segmentation.ndim != 3:
        raise ValueError("Segmentation must be 3D (H, W, C).")
    if not exceed_image and (shape > segmentation.shape[0] or shape > segmentation.shape[1]):
        logger.warning("Tile shape exceeds segmentation dimensions; no tiles extracted.")
        return {}, {}, {}, {}, {}

    stride = shape if stride is None else stride
    if exclusion_map is not None:
        exclusion_map = exclusion_map.numpy() if isinstance(exclusion_map, torch.Tensor) else exclusion_map
        if exclusion_map.ndim == 3:
            exclusion_map = exclusion_map[..., 0]
        if exclusion_map.shape[:2] != segmentation.shape[:2]:
            raise ValueError("Exclusion map dimensions must match segmentation.")

    tile_info, crops, excl_crops, top_lefts, outsides = {}, {}, {}, {}, {}
    for i in range(segmentation.shape[-1]):
        cross_section = segmentation[..., i]
        if preprocessing_function:
            cross_section = preprocessing_function(cross_section)

        bbox = get_bounding_box(cross_section)
        if not bbox:
            continue

        top, bottom, left, right = bbox
        height, width = bottom - top + 1, right - left + 1
        grid_h, grid_w = ceil(height / shape), ceil(width / shape)
        pad_h, pad_w = grid_h * shape - height, grid_w * shape - width
        top_adj = top - pad_h // 2
        left_adj = left - pad_w // 2
        crop_h, crop_w = grid_h * shape, grid_w * shape

        if not exceed_image:
            top_adj = max(0, top_adj)
            left_adj = max(0, left_adj)
            crop_h = min(crop_h, segmentation.shape[0] - top_adj)
            crop_w = min(crop_w, segmentation.shape[1] - left_adj)

        crop = cross_section[top_adj:top_adj + crop_h, left_adj:left_adj + crop_w]
        if exceed_image:
            pad_top = max(0, -top_adj)
            pad_left = max(0, -left_adj)
            pad_bottom = max(0, top_adj + crop_h - segmentation.shape[0])
            pad_right = max(0, left_adj + crop_w - segmentation.shape[1])
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

        crops[str(i)] = crop
        excl_crop = None
        if exclusion_map is not None:
            excl_crop = exclusion_map[top_adj:top_adj + crop_h, left_adj:left_adj + crop_w]
            if exceed_image:
                excl_crop = np.pad(excl_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            excl_crops[str(i)] = excl_crop

        mask = compute_extraction_region(crop, shape, stride, min_tissue_fraction, excl_crop)
        y_idx, x_idx = np.nonzero(mask)
        tiles = [
            ((int(x), int(y)), (int(left_adj + x * stride), int(top_adj + y * stride)), shape)
            for x, y in zip(x_idx, y_idx)
        ]
        tile_info[str(i)] = tiles
        top_lefts[str(i)] = (left_adj, top_adj)
        outsides[str(i)] = (pad_left if exceed_image else 0, pad_top if exceed_image else 0)

    return tile_info, crops, excl_crops, top_lefts, outsides

def scale_tile_information(tile_information: Dict, scaling_factor: float) -> Optional[Dict]:
    """Scale tile coordinates and shapes by a factor."""
    if not tile_information:
        return None
    return {
        k: [(pos, (int(loc[0] * scaling_factor), int(loc[1] * scaling_factor)), int(shape * scaling_factor))
            for pos, loc, shape in tiles]
        for k, tiles in tile_information.items()
    }

def ceil(x):
    return int(np.ceil(x))

def process_single_slide(slide_data, save_dir, configs):
    """Process a single slide and return its output dictionary."""
    slide, slide_path, mask_path = slide_data
    tiling_save_dir = save_dir / 'tiles'
    visualization_save_dir = save_dir / 'visualizations'
    
    output = {
        'slide_id': slide, 'slide_location': str(slide_path), 'mask_location': str(mask_path),
        'coords_location': '', 'grid_width': 0, 'grid_height': 0,
        'wsi_crop_width': 0, 'wsi_crop_height': 0, 'n_tiles': 0
    }

    try:
        tiling_path = tiling_save_dir / f'{slide}.h5'
        if configs.auto_skip and tiling_path.is_file():
            logger.info(f"{slide} already exists at {tiling_path}, skipped")
            output['coords_location'] = str(tiling_path)
            return output

        # Load slide and mask
        loader = SlideLoader({'multithreading': True, 'progress_bar': False})
        try:
            loader.load_slide(slide_path)
            true_height, true_width = loader.get_dimensions(configs.tiling_magnification)
            image = loader.get_image(magnification=configs.segmentation_magnification)
        except Exception as e:
            logger.error(f"Failed to load slide {slide}: {str(e)}")
            return output
        finally:
            loader.close()

        # Crop image if needed
        height, width, _ = image.shape
        ratio = configs.tiling_magnification / configs.segmentation_magnification
        crop_height = ceil(max(0, height * ratio - true_height) / ratio)
        crop_width = ceil(max(0, width * ratio - true_width) / ratio)
        if crop_height > 0 or crop_width > 0:
            image = image[:height - crop_height, :width - crop_width, :]

        # Load and preprocess segmentation mask
        try:
            segmentation = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if segmentation is None:
                raise ValueError("Segmentation mask could not be loaded")
            segmentation = (segmentation > 0).astype(np.uint8)[:, :, np.newaxis]
        except Exception as e:
            logger.error(f"Failed to load mask for {slide}: {str(e)}")
            return output

        # Tile the segmentation mask
        tiling_shape = int(configs.shape / ratio)
        tiling_stride = int(configs.stride / ratio)
        try:
            tile_info, crops, _, top_left, outside = tile(
                segmentation=segmentation, exclusion_map=None,
                shape=tiling_shape, stride=tiling_stride,
                min_tissue_fraction=configs.min_tissue_fraction,
                exceed_image=configs.exceed_image
            )
            tile_info_scaled = scale_tile_information(tile_info, ratio)['0']
        except Exception as e:
            logger.error(f"Tiling failed for {slide}: {str(e)}")
            return output

        # Prepare assets for saving
        positions = np.array([x[0] for x in tile_info_scaled])
        coordinates = np.array([x[1] for x in tile_info_scaled])
        shapes = np.array([x[2] for x in tile_info_scaled])
        crop = crops['0']

        output.update({
            'coords_location': str(tiling_path),
            'grid_width': int(positions[:, 0].max() + 1),
            'grid_height': int(positions[:, 1].max() + 1),
            'wsi_crop_width': int(crop.shape[1] * ratio),
            'wsi_crop_height': int(crop.shape[0] * ratio),
            'n_tiles': len(tile_info_scaled)
        })

        asset_dict = {
            '0': {
                'positions': positions, 'coords': coordinates, 'shapes': shapes,
                'crop': crop, 'top_left': np.array(top_left['0']),
                'outside': np.array(outside['0'])
            }
        }
        meta = {k: v for k, v in vars(configs).items() if v is not None}
        meta['slide_ext'] = slide_path.suffix

        # Save to HDF5
        try:
            save_hdf5(tiling_path, asset_dict, meta, 'w')
        except Exception as e:
            logger.error(f"Failed to save HDF5 for {slide}: {str(e)}")
            return output

        # Visualize tiling
        try:
            visualize_tiling(
                images=image, tile_information=tile_info,
                output_path=visualization_save_dir / f'{slide}.png',
                line_color=configs.line_color, line_width=configs.line_width,
                downscale_factor=configs.downscale_factor, axis=configs.axis,
                enum_tiles=configs.enum_tiles
            )
        except Exception as e:
            logger.warning(f"Visualization failed for {slide}: {str(e)}")

        logger.info(f"Successfully processed {slide}")
        return output

    except Exception as e:
        logger.error(f"Unexpected error processing {slide}: {str(e)}")
        return output
    
def process_tiling_list(save_dir, configs, num_workers=4):
    """Process a list of slides in parallel and append results to existing CSV."""
    save_dir = Path(save_dir)
    tiling_save_dir = save_dir / 'tiles'
    visualization_save_dir = save_dir / 'visualizations'
    tiling_save_dir.mkdir(parents=True, exist_ok=True)
    visualization_save_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate CSV with slide list
    try:
        df = pd.read_csv(configs.process_list)
        required_cols = {"slide_id", "slide_location", "mask_location"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain {required_cols}")
        slides_data = df[list(sorted(required_cols))].values.tolist()
        slides_data = [(s, Path(sl), Path(m)) for m, s, sl in slides_data]
    except Exception as e:
        logger.error(f"Failed to load process list {configs.process_list}: {str(e)}")
        return

    # Filter valid paths
    
    slides_data = [(s, sl, m) for s, sl, m in slides_data if sl.is_file() and m.is_file()]
    logger.info(f"Processing {len(slides_data)} slides with valid paths and masks")

    # Load existing CSV if it exists
    csv_path = save_dir / 'processed_slides_autogen.csv'
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            logger.info(f"Found existing CSV with {len(existing_df)} entries")
            # Optionally filter out already processed slides
            processed_slides = set(existing_df['slide_id'])
            slides_data = [(s, sl, m) for s, sl, m in slides_data if s not in processed_slides]
            logger.info(f"After filtering, {len(slides_data)} new slides to process")
        except Exception as e:
            logger.error(f"Failed to read existing CSV {csv_path}: {str(e)}")
            existing_df = pd.DataFrame()  # Start fresh if reading fails
    else:
        existing_df = pd.DataFrame()
        logger.info("No existing CSV found, starting fresh")

    if not slides_data:
        logger.info("No new slides to process")
        return

    # Process slides in parallel
    with Pool(num_workers) as pool:
        func = partial(process_single_slide, save_dir=save_dir, configs=configs)
        results = list(tqdm(pool.imap(func, slides_data), total=len(slides_data), desc="Processing slides"))

    # Combine new results with existing data
    new_df = pd.DataFrame(results)
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Save combined results
    try:
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved updated CSV with {len(combined_df)} total entries")
    except Exception as e:
        logger.error(f"Failed to save updated CSV: {str(e)}")
