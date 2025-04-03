
import logging
from src.data_curation.segment import process_slides
from src.utils.process_segmentations import split_png_stack, create_tiling_process_list
import argparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments for slide processing."""
    parser = argparse.ArgumentParser(description="Process NDPI slides with two segmenters in parallel.")
    parser.add_argument(
        "--input-folder",
        type=str,
        default="data/images",
        help="Path to the folder containing NDPI slides (default: data/images)"
    )
    parser.add_argument(
        "--model-a-checkpoint",
        type=str,
        default="resources/slidesegmenter_files/checkpoint_I59000.tar",
        help="Path to the first segmentation model checkpoint (default: resources/slidesegmenter_files/checkpoint_I59000.tar)"
    )
    parser.add_argument(
        "--model-b-checkpoint",
        type=str,
        default="resources/slidesegmenter_files/segmenter_weights_011024.pth",
        help="Path to the second segmentation model checkpoint (default: resources/slidesegmenter_files/segmenter_weights_011024.pth)"
    )
    parser.add_argument(
        "--model-settings",
        type=str,
        default="resources/slidesegmenter_files/settings.json",
        help="Path to the model settings file (default: resources/slidesegmenter_files/settings.json)"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="tests/segmentation",
        help="Path to the folder where processed images will be saved (default: tests/segmentation)"
    )
    parser.add_argument(
        "--magnification",
        type=float,
        default=0.625,
        help="Magnification level for loading images (default: 0.625)"
    )
    parser.add_argument(
        "--target-magnification",
        type=float,
        default=20.0,
        help="Target magnification for ratio calculation (default: 20.0)"
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="*.ndpi",
        help="Glob pattern for slide files (e.g., '*.ndpi', '*.tif') (default: *.ndpi)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, set to 0 for serial execution)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    process_slides(
        input_folder=args.input_folder,
        model_a_checkpoint=args.model_a_checkpoint,
        model_b_checkpoint=args.model_b_checkpoint,
        model_settings=args.model_settings,
        output_folder=args.output_folder,
        magnification=args.magnification,
        target_magnification=args.target_magnification,
        glob_pattern=args.glob_pattern,
        max_workers=args.max_workers
    )

    split_png_stack(input_folder=args.output_folder)

    create_tiling_process_list(text_file=f'{args.output_folder}/processed_slides.txt', 
                               mask_location=f'{args.output_folder}/segmenter_masks', 
                               output_file=f'{args.output_folder}/tiling_process_list.csv')