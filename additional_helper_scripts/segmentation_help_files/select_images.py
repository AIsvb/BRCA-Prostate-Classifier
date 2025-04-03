import os
import cv2
import shutil
from pathlib import Path
import random


INPUT_FOLDER = ...
OUTPUT_FOLDER = ...


Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
done = [x.name for x in list(Path(OUTPUT_FOLDER).rglob('*.png'))]

png_files = [x for x in list(Path(INPUT_FOLDER).rglob('*.png')) if not x.name in done]
random.shuffle(png_files)

# Loop over the PNG files
for file_name in png_files:

    # Read the image
    image = cv2.imread(file_name)

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
    print(f"Displaying: {file_name}. Press Enter to move to output folder, x to stop the program, or any other key to continue to the next image.")

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    if key == 13:  # Enter key
        shutil.move(file_name, Path(OUTPUT_FOLDER) / file_name.name)
    elif key == ord('x'):  # 'x' key
        break
    else:
        continue

    # Close the image window
    cv2.destroyAllWindows()

print("Processing complete.")
