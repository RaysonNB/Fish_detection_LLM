import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from PIL import Image
import os
import matplotlib.pyplot as plt


# --- Functions from the first script ---
def average_brightness(a):
    """Calculates the average brightness of an image in the HSV color space."""
    arr = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    return np.mean(arr[:, :, 2])


def check(a):
    """Clips pixel values to the valid range [0, 1]."""
    a = np.clip(a, 0, 1)
    return a


# --- Functions from the second script ---
def plot_histograms(image, title, output_path):
    """
    Plots and saves the B, G, R channel histograms of an image.
    """
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    plt.plot(hist_b, color='blue', label='Blue Channel')
    plt.plot(hist_g, color='green', label='Green Channel')
    plt.plot(hist_r, color='red', label='Red Channel')

    plt.legend()
    plt.grid(True)
    plt.xlim([0, 256])

    plt.savefig(output_path)
    plt.close()


def enhance_image_clahe(input_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    """
    if input_image is None:
        return None

    lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l)
    merged_lab = cv2.merge((enhanced_l, a, b))
    enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return enhanced_img


# --- Main Execution Section ---
if __name__ == '__main__':
    # Load the MAXIM model once
    model = from_pretrained_keras("google/maxim-s2-enhancement-lol")

    # Define all paths
    base_path = "C:/Users/rayso/Desktop/python/"
    input_directory = os.path.join(base_path, "before_images/")
    temp_directory = os.path.join(base_path,"evertings_for_final/temp_enhanced/")  # Temporary folder for intermediate results
    output_base_dir = os.path.join(base_path, "evertings_for_final/final_output/")

    # Define output sub-directories for the final results
    output_dirs = {
        'original_histograms': os.path.join(output_base_dir, 'Original_Histograms'),
        'processed_histograms': os.path.join(output_base_dir, 'Processed_Histograms'),
        'image_comparisons': os.path.join(output_base_dir, 'Image_Comparisons'),
        'processed_images': os.path.join(output_base_dir, 'Processed_Images'),
    }

    # Create all necessary directories
    os.makedirs(temp_directory, exist_ok=True)
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    print("Starting image processing pipeline...")

    # --- Step 1: Low-Light Enhancement & Sharpening ---
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_directory, filename)
            temp_image_path = os.path.join(temp_directory, filename)

            print(f"\nProcessing '{filename}' for initial enhancement...")

            # Load and process the image with MAXIM
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            h, w, _ = image_array.shape

            t = image_array.copy()
            t = np.array(t, np.float32) / 255
            t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
            b = average_brightness(t)

            # Apply Real-ESRGAN for initial sharpening
            os.system(
                f"{base_path}realesrgan-ncnn-vulkan.exe -i \"{image_path}\" -o \"{temp_image_path}\" -n realesrgan-x4plus")

            # Use the newly upscaled image for the next step
            enhanced_image_for_clahe = cv2.imread(temp_image_path)

            if enhanced_image_for_clahe is None:
                print(f"Error: Could not read upscaled image from Real-ESRGAN for '{filename}'. Skipping CLAHE.")
                continue

            # --- Step 2: CLAHE Contrast Enhancement & Analysis ---
            base_name, _ = os.path.splitext(filename)

            # Load original image for comparison
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error: Could not read original image for '{filename}'. Skipping comparison.")
                continue

            # Plot original histogram
            original_hist_output_path = os.path.join(output_dirs['original_histograms'],
                                                     f'{base_name}_original_hist.png')
            plot_histograms(original_image, 'RGB Distribution of Original Image', original_hist_output_path)

            # Apply CLAHE
            final_processed_image = enhance_image_clahe(enhanced_image_for_clahe)

            if final_processed_image is not None:
                # Plot processed histogram
                processed_hist_output_path = os.path.join(output_dirs['processed_histograms'],
                                                          f'{base_name}_clahe_hist.png')
                plot_histograms(final_processed_image, 'RGB Distribution of Final CLAHE Image',
                                processed_hist_output_path)

                # Create and save a comparison image of ORIGINAL vs. FINAL PROCESSED
                # Resize original to match the final image size for proper stacking
                original_resized = cv2.resize(original_image, (final_processed_image.shape[1], final_processed_image.shape[0]))
                comparison_image = np.hstack([original_resized, final_processed_image])
                comparison_output_path = os.path.join(output_dirs['image_comparisons'], f'{base_name}_comparison.jpg')
                cv2.imwrite(comparison_output_path, comparison_image)

                # Save the final processed image
                processed_image_output_path = os.path.join(output_dirs['processed_images'],
                                                           f'{base_name}_clahe_enhanced.jpg')
                cv2.imwrite(processed_image_output_path, final_processed_image)

            else:
                print(f"CLAHE enhancement failed for '{filename}'.")

    print("\nAll image processing completed successfully.")
    cv2.destroyAllWindows()