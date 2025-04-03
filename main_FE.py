from src.models.feature_extraction.builder import build_fe_from_name
from src.utils.feature_extraction import UnfoldCollator
from src.utils.transform import get_eval_transforms
from src.utils.slideloader import SlideLoader
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import threading
import argparse
import torch
import queue
import h5py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stop_event = threading.Event()

# -- loops over all slides in a process list csv and performs FE
def process_extraction_list(save_dir, configs):

    # -- make relevant directories
    (save_dir / 'pt_files').mkdir(parents=True, exist_ok=True)

    # -- load slide data from process list csv file
    slide_data = pd.read_csv(configs.process_list)

    slides = slide_data['slide_id'].tolist()
    slide_paths = slide_data['slide_location'].tolist()
    coord_paths = slide_data['coords_location'].tolist()

    slides, slide_paths, coord_paths = zip(*
                                           [(slide, Path(slide_path), Path(coord_path)) 
                                            for slide, slide_path, coord_path in zip(slides, slide_paths, coord_paths) 
                                            if isinstance(slide_path, str) and isinstance(coord_path, str) and Path(slide_path).is_file() and Path(coord_path).is_file()
                                            ])
    total = len(slides)
    print(f'Found {total} slides for feature extraction', flush=True)

    # -- init FE model
    print(f'Initializing model with name {configs.name}', flush=True)
    model, constants = build_fe_from_name(configs.name)
    mean, std, embed_dim = constants['mean'], constants['std'], constants['embed_dim']
    model.eval()
    model.to(device)
    print('Done!', flush=True)

    # -- define image transforms (when using HIPT, the transform must unfold the 4k region into a batch of smaller tiles)
    transform = get_eval_transforms(mean=mean, std=std, target_img_size=configs.target_img_size, pad=configs.tile_shape)
    collate_fn, batch_mp_factor = None, 1
    if configs.hipt:
        ratio = (configs.tile_shape // configs.unfold)
        transform = get_eval_transforms(mean=mean, std=std, target_img_size = int(ratio * configs.target_img_size))
        collate_fn = UnfoldCollator(shape=configs.target_img_size, stride=configs.target_img_size)
        batch_mp_factor = (ratio**2)
        print(f'Batch size has been increased to {configs.batch_size * batch_mp_factor} as a result of unfolding', flush=True)

    # -- initialize slide loader object and a queue for storing loaded batches of image tiles
    slide_loader = SlideLoader()
    tile_queue = queue.Queue(maxsize=3)
    load_lock = threading.Lock()

    # -- loop over all the slides
    for i, (slide, slide_path, coord_path) in enumerate(zip(slides, slide_paths, coord_paths)):
        print(f'Processing slide with id {slide} ({i + 1} / {total})', flush=True)

        dest = save_dir / 'pt_files' / f'{slide}.pt'
        if dest.is_file() and configs.auto_skip:
            print('Feature file already in destination, skipping slide', flush=True)
            continue

        # -- load tiling meta data
        file = h5py.File(coord_path, 'r')
        shape = file.attrs['shape']
        magnification = file.attrs['tiling_magnification']

        # -- load whole slide image
        slide_loader.load_slide(slide_path)
        slide_dims = slide_loader.get_dimensions(magnification)

        # -- retrieve the tile coordinates and batch them
        coordinates = file['0']['coords']
        coordinates, shapes = process_coords(coordinates, slide_dims, shape)
        batches = [coordinates[j:j+configs.batch_size] for j in range(0, len(coordinates), configs.batch_size)]
            
        # -- allocate space on the gpu for storing the extracted features
        n_tiles = len(coordinates) * batch_mp_factor if configs.hipt else len(coordinates)
        results_tensor = torch.zeros(n_tiles, embed_dim, device=device, dtype=torch.float32)
            
        # -- create a separate thread for extracting the features from the tile batches placed in the queue by the main thread
        inference_thread = threading.Thread(
            target=run_inference, 
            args=(model, tile_queue, results_tensor, 0, configs.batch_size * batch_mp_factor, device))
        inference_thread.start()

        # -- load batches of tiles and put them in the queue
        coordinates = None
        for j, batch in enumerate(batches):

            with load_lock:
                tiles = slide_loader.get_tiles(magnification, list(batch), shapes[j*configs.batch_size: j*configs.batch_size + len(batch)])
                tiles = torch.stack([transform(Image.fromarray(tile)) for tile in tiles]).float()

                # -- optionally unfold the tile (e.g. when loading tiles of 4096 for HIPT)
                if configs.hipt:
                    # Broadcast grid coordinates to the whole batch and add the base coordinates
                    tiles, batch = collate_fn(tiles, batch)
                        
                coordinates = np.vstack([coordinates, batch]) if coordinates is not None else batch
                tile_queue.put(tiles)

        # -- put None in the queue to signal the inference thread it must end
        tile_queue.put(None)

        # -- wait for the queue to be empty and the inference to have finished
        tile_queue.join()
        inference_thread.join()

        # -- move the gpu tensor with features back to the cpu, add the coordinates at the end of 
        # -- the features and combine with features from other cross sections
        cpu_features = torch.cat((results_tensor.cpu(), torch.from_numpy(coordinates)), dim=1)
        torch.save(cpu_features, dest)

        # -- close the h5 file and slide file
        file.close()
        slide_loader.close()

# -- performs a forward pass of the tiles through the feature extractor
def run_inference(model, tile_queue, results_tensor, batch_idx, batch_size, device):

    # -- start infinite loop
    while True:

        try:
            # -- get element from the queue, the thread will block if the queue is empty
            tiles = tile_queue.get()

            # -- check whether the infinite loop must be broken
            if tiles is None:
                tile_queue.task_done()
                break 
            
            # -- move the tiles to the gpu (if present, else to the cpu)
            tiles = tiles.to(device, non_blocking=True)

            # -- extract the features
            with torch.inference_mode():
                features = model(tiles)

            # -- in case of Virchow2
            if len(features.shape) == 3:
                class_token = features[:, 0]    # size: 1 x 1280
                patch_tokens = features[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

                # concatenate class token and average pool of patch tokens
                features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

            # -- assign the features to their designated part of the cuda tensor
            results_tensor[batch_idx * batch_size: batch_idx * batch_size + len(tiles), ...] = features

            # -- inform the queue that one element has been processed
            tile_queue.task_done()

            # -- increment the batch index
            batch_idx += 1

        except torch.cuda.OutOfMemoryError as _:

            stop_event.set()

# -- corrects coordinates if exceed image was set to true when tiling and therefore top-left coordinates can exceed the image
def process_coords(coords, slide_dims, shape):
    _coords = []
    _shapes = []
    
    for col, row in coords:
        # Append corrected coordinates (col, row) with non-negative values
        _coords.append((max(col, 0), max(row, 0)))
        
        # Width and height corrections based on boundary conditions
        width_correction = min(0, slide_dims[1] - (col + shape)) if col >= 0 else col
        height_correction = min(0, slide_dims[0] - (row + shape)) if row >= 0 else row
        
        # Calculate corrected shape and append to _shapes
        _shapes.append((shape + height_correction, shape + width_correction))
    
    return np.array(_coords), _shapes


parser = argparse.ArgumentParser()
parser.add_argument( 
    '--process_list', type=str, help='csv file with columns slide_id, slide_location, and coords_location')
parser.add_argument(
    '--save_dir', type=str, help='location of the root folder in which FE results will be stored')
parser.add_argument(
	'--name', type=str, help='name of the FE model (must correspond with a model in the modelzoo --> src/models/feature_extraction/modelzoo.py)')
parser.add_argument(
    '--hipt', action='store_true', help='whether FE is intended for HIPT in which case FE will be performed on unfolded image tile')
parser.add_argument(
    '--tile_shape', type=int, default=256, help='size in pixels of the tiles square sides')
parser.add_argument(
    '--unfold', type=int, default=256, help='in case FE is intended for HIPT, this argument specifies the tile size of the subtiles within a 4K region')
parser.add_argument(
    '--target_img_size', type=int, default=224, help='the expected image input size of the FE model')
parser.add_argument(
    '--auto_skip', type=bool, default=True, help='whether to skip files for which the destinatio folder already specifies a tile file')
parser.add_argument(
    '--batch_size', type=int, default=256, help='Number of tiles to include in one forward pass \
    (N.B.: In case of HIPT, this value will be multiplied by the number of tiles present in one 4K region)')

if __name__ == "__main__":

    args = parser.parse_args()

    process_extraction_list(
        save_dir = Path(args.save_dir),
        configs = args
    )
