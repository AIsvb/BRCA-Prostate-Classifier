import os
import cv2
import numpy as np

def process_images(folder_path, output_folder):
    """
    Process images by letting the user choose a portion of the binary mask to mask as black, either from the left, right, top, or bottom edges.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where modified images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_file in image_files:
        # Load image
        save_path = os.path.join(output_folder, image_file)

        if os.path.exists(save_path):
            print('File already processed')
            continue

        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Failed to load {image_file}. Skipping.")
            continue

        H = image.shape[0] // 2  # Height of each part (thumbnail and mask)
        W = image.shape[1]       # Width of the image

        # Split into thumbnail and mask
        thumbnail = image[:H, :]
        mask = image[H:, :]

        screen_width = 1000
        screen_height = 700
        image_height, image_width = image.shape[:2]
        scaling_factor = min(screen_width / image_width, screen_height / image_height, 1.0)

        mask_value_0, mask_value_1, mask_value_2, mask_value_3 = 10, 10, 10, 10
        while True:
            # Resize for display if necessary
            display_image = cv2.resize(np.vstack((thumbnail, mask)), (int(image_width * scaling_factor), int(image_height * scaling_factor)))
            #display_image = cv2.resize(np.vstack((thumbnail, mask)), (W // 2, H)) if W > 1000 or H > 1000 else np.vstack((thumbnail, mask))
            cv2.imshow(f"Processing: {image_file}", display_image)

            print("LEFT > RIGHT > TOP > BOTTOM | SMALL > LARGE, 12345678qwertyui")


            key = cv2.waitKey(0) & 0xFF

            if key == ord('1'):
                columns_to_mask = int(W * 1 / 16)
                #mask[:, :columns_to_mask] = 0
                mask[:, :mask_value_0] = 0
                mask_value_0 += 10
            elif key == ord('2'):
                columns_to_mask = int(W * 1 / 8)
                mask[:, :columns_to_mask] = 0
            elif key == ord('3'):
                columns_to_mask = int(W * 1 / 4)
                mask[:, :columns_to_mask] = 0
            elif key == ord('4'):
                columns_to_mask = int(W * 1 / 2)
                mask[:, :columns_to_mask] = 0
            elif key == ord('5'):
                columns_to_mask = int(W * 1 / 16)
                #mask[:, -columns_to_mask:] = 0
                mask[:, -mask_value_1:] = 0
                mask_value_1 += 10
            elif key == ord('6'):
                columns_to_mask = int(W * 1 / 8)
                mask[:, -columns_to_mask:] = 0
            elif key == ord('7'):
                columns_to_mask = int(W * 1 / 4)
                mask[:, -columns_to_mask:] = 0
            elif key == ord('8'):
                columns_to_mask = int(W * 1 / 2)
                mask[:, -columns_to_mask:] = 0
            elif key == ord('q'):
                rows_to_mask = int(H * 1 / 16)
                #mask[:rows_to_mask, :] = 0
                mask[:mask_value_2, :] = 0
                mask_value_2 += 10
            elif key == ord('w'):
                rows_to_mask = int(H * 1 / 8)
                mask[:rows_to_mask, :] = 0                
            elif key == ord('e'):
                rows_to_mask = int(H * 1 / 4)
                mask[:rows_to_mask, :] = 0
            elif key == ord('r'):
                rows_to_mask = int(H * 1 / 2)
                mask[:rows_to_mask, :] = 0                
            elif key == ord('t'):
                rows_to_mask = int(H * 1 / 16)
                #mask[-rows_to_mask:, :] = 0
                mask[-mask_value_3:, :] = 0
                mask_value_3 += 10
            elif key == ord('y'):
                rows_to_mask = int(H * 1 / 8)
                mask[-rows_to_mask:, :] = 0
            elif key == ord('u'):
                rows_to_mask = int(H * 1 / 4)
                mask[-rows_to_mask:, :] = 0
            elif key == ord('i'):
                rows_to_mask = int(H * 1 / 2)
                mask[-rows_to_mask:, :] = 0
            elif key == 13:  # Enter key
                # Save the modified image
                save_path = os.path.join(output_folder, image_file)
                modified_image = np.vstack((thumbnail, mask))
                cv2.imwrite(save_path, modified_image)
                print(f"Modified image saved to {save_path}")
                break
            elif key == ord('x'):
                print("Exiting program.")
                cv2.destroyAllWindows()
                return
            else:
                print("Invalid key. Try again.")
                break

        cv2.destroyAllWindows()

# Example usage
process_images("data_processing/data/", "data_processing/data/temp_")
