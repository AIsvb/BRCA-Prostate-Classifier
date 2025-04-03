import cv2
import numpy as np
from pathlib import Path
import os


# Function to process a stacked PNG file
def process_stacked_png(input_path, output_path):
    # Read the stacked image
    stacked_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    h, _, _ = stacked_image.shape

    if stacked_image is None:
        print(f"Could not read {input_path}")
        return

    sh = h // 3
    # Split the stack into the RGB image and the two binary masks
    rgb_image = stacked_image[:sh, :, :]  # RGB image
    mask1 = stacked_image[sh:sh*2, :, 0]      # First binary mask
    mask2 = stacked_image[sh*2:sh*3, :, 0]      # Second binary mask

    # Ensure binary masks are truly binary (0 or 255)
    mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)[1]
    mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)[1]

    # Perform the intersection of the two masks
    intersection = cv2.bitwise_and(mask1, mask2)
    intersection = np.repeat(intersection[:, :, np.newaxis], 3, axis=2)
    # Stack the RGB image and the intersection mask
    image = np.vstack((rgb_image, intersection))

    # Save the resulting stack as a PNG file
    cv2.imwrite(output_path, image)
    print(f"Processed and saved: {output_path}")

def reduced_stacked_png(input_path, output_path):
    # Read the stacked image
    stacked_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    h, _, _ = stacked_image.shape

    if stacked_image is None:
        print(f"Could not read {input_path}")
        return

    sh = h // 3
    # Split the stack into the RGB image and the two binary masks
    rgb_image = stacked_image[:sh, :, :]  # RGB image
    mask1 = stacked_image[sh:sh*2, :, :]      # First binary mask

    # Stack the RGB image and the intersection mask
    image = np.vstack((rgb_image, mask1))

    # Save the resulting stack as a PNG file
    cv2.imwrite(output_path, image)
    print(f"Processed and saved: {output_path}")

def split_and_save_images(folder_path, rgb_output_folder, mask_output_folder):
    """
    Split images into RGB thumbnails and binary masks, and save them in separate folders.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        rgb_output_folder (str): Path to the folder where RGB thumbnails will be saved.
        mask_output_folder (str): Path to the folder where binary masks will be saved.
    """
    # Create output folders if they do not exist
    os.makedirs(rgb_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Failed to load {image_file}. Skipping.")
            continue

        # Split the image into RGB thumbnail and binary mask
        H = image.shape[0] // 2  # Half the height
        rgb_thumbnail = image[:H, :]
        binary_mask = image[H:, :]

        # Save the RGB thumbnail and binary mask
        rgb_save_path = os.path.join(rgb_output_folder, image_file)
        mask_save_path = os.path.join(mask_output_folder, image_file)

        cv2.imwrite(rgb_save_path, rgb_thumbnail)
        cv2.imwrite(mask_save_path, binary_mask)

        print(f"Saved RGB thumbnail to {rgb_save_path}")
        print(f"Saved binary mask to {mask_save_path}")

# Example usage
folder_path = 'data_processing/data/temp_'
rgb_output_folder = 'data_processing/data/temp_/RGB'
mask_output_folder = 'data_processing/data/temp_/masks'
split_and_save_images(folder_path, rgb_output_folder, mask_output_folder)


# files = list(Path('data_processing/data/segmentation_visuals').rglob('*.png'))

# for file in files:
#      output_path = f'data_processing/data/segmentation_visuals_reduced/{file.name}'
#      reduced_stacked_png(file, output_path)

