import os
from PIL import Image
from pathlib import Path
import pandas as pd


def split_png_stack(input_folder):
    # Define the output subfolder names
    subfolders = ['thumbnails', 'segmenter_masks', 'otsu_masks']
    
    # Create the subfolders in the input folder if they don't exist
    for subfolder in subfolders:
        os.makedirs(os.path.join(input_folder, subfolder), exist_ok=True)
    
    # Process each PNG file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(input_folder, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # Get image dimensions
                width, height = img.size
                
                # Calculate the height of each third
                third_height = height // 3
                
                # Ensure the image height is divisible by 3
                if height % 3 != 0:
                    print(f"Warning: {filename} height ({height}) is not divisible by 3. Skipping.")
                    continue
                
                # Split the image into three parts
                for i, subfolder in enumerate(subfolders):
                    # Define the crop box (left, upper, right, lower)
                    box = (0, i * third_height, width, (i + 1) * third_height)
                    # Crop the image
                    cropped_img = img.crop(box)
                    # Save the cropped image to the corresponding subfolder
                    output_path = os.path.join(input_folder, subfolder, filename)
                    cropped_img.save(output_path)
                print(f"Processed: {filename}")

def create_tiling_process_list(text_file, mask_location, output_file):

    '''
    text_file: text file whose lines are formatted as: <SLIDE_ID>,<PATH_TO_WSI_FILE>,<PATH_TO_STACKED_SEGMENTATION_RESULTS>
    mask_location: the folder containing the masks that must be used for tiling (potentially subjected to manual refinements)

    output: csv file with columns slide_id, slide_location, and mask_location which serves as input to the tile script
    '''

    with open(text_file, 'r') as file:
        lines = [x.replace('\n', '') for x in file.readlines()]

    ids = [line.split(',')[0] for line in lines]
    slide_locs = [line.split(',')[1] for line in lines]
    mask_locs = []

    for id in ids:

        png = Path(mask_location) / f'{id}.png'
        if png.is_file():
            mask_locs.append(str(png))
        else:
            mask_locs.append(None)

    ids = [i for i, j in zip(ids, mask_locs) if j is not None]
    slide_locs = [i for i, j in zip(slide_locs, mask_locs) if j is not None]
    mask_locs = [i for i in mask_locs if i is not None]

    data = {
        'slide_id': ids,
        'slide_location': slide_locs,
        'mask_location': mask_locs
    }

    pd.DataFrame(data=data).to_csv(output_file)
        