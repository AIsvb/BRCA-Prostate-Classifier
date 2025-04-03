import os
import cv2
import shutil
from pathlib import Path
import random


# Define the source and destination directories
source_dir = "data_processing/data/segmentation_visuals/"
accepted_dest_dir = "data_processing/data/segmentation_visuals_otsu"

# Ensure destination directories exist
os.makedirs(accepted_dest_dir, exist_ok=True)

# Get all PNG files in the source directory
png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

if not png_files:
    print("No PNG files found in the source directory.")
else:
    print(f"Found {len(png_files)} PNG files.")

# Loop over the PNG files
for file_name in png_files:
    file_path = os.path.join(source_dir, file_name)

    # Read the image
    image = cv2.imread(file_path)

    if image is None:
        print(f"Could not read {file_name}. Skipping...")
        continue

    # Resize the image window to fit the screen
    screen_width = 1000
    screen_height = 700
    image_height, image_width = image.shape[:2]
    scaling_factor = min(screen_width / image_width, screen_height / image_height, 1.0)
    resized_image = cv2.resize(image, (int(image_width * scaling_factor), int(image_height * scaling_factor)))

    # Display the image
    cv2.imshow("Image Viewer", resized_image)
    print(f"Displaying: {file_name}. Press Enter to move to accepted folder, 'd' to move to deletable folder, or any other key to move to skipped folder.")

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    if key == 13:  # Enter key
        # Move the file to the accepted destination folder
        shutil.move(file_path, os.path.join(accepted_dest_dir, file_name))
        print(f"Moved: {file_name} to {accepted_dest_dir}")
    elif key == ord('x'):  # 'd' key
        break
    else:
        continue

    # Close the image window
    cv2.destroyAllWindows()

print("Processing complete.")
