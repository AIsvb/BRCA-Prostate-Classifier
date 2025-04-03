from src.data_curation.tile import process_tiling_list
from pathlib import Path
import argparse


# Argument parser (unchanged, omitted for brevity)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( 
        '--process_list', type=str, help='csv file with columns slide_id, slide_location, and mask_location')
    parser.add_argument(
        '--save_dir', type=str, help='location of the root folder in which tiling results will be stored')
    parser.add_argument(
        '--tiling_magnification', type=float, default=20.0, help='magnification at which to extract tiles')
    parser.add_argument(
        '--segmentation_magnification', type=float, default=0.625, help='magnification for which the segmentation masks were created')
    parser.add_argument(
        '--shape', type=int, default=256, help='size in pixels of the tiles square sides')
    parser.add_argument(
        '--stride', type=int, default=256, help='stride, use to manage overlap of tiles')
    parser.add_argument(
        '--min_tissue_fraction', type=float, default=0.001, help='minimum fraction of tissue that a tile must have to be accepted')
    parser.add_argument(
        '--exceed_image', type=bool, default=True, help='whether tiles can partially span beyond the image')
    parser.add_argument(
        '--auto_skip', type=bool, default=True, help='whether to skip files for which the destinatio folder already specifies a tile file')
    parser.add_argument(
        '--line_color', type=str, default='hsv', help='visualization parameter')
    parser.add_argument(
        '--line_width', type=int, default=1, help='visualization parameter')
    parser.add_argument(
        '--axis', type=int, default=0, help='visualization parameter')
    parser.add_argument(
        '--downscale_factor', type=int, default=2, help='visualization parameter')
    parser.add_argument(
        '--enum_tiles', type=bool, default=False, help='whether to display the tile indices in the visualization')
    parser.add_argument(
        '--num_workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    process_tiling_list(Path(args.save_dir), args, args.num_workers)
