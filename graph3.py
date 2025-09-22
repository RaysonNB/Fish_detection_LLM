import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_comparison_images(original_path, processed_path, output_dir):
    """
    Generates and saves three types of comparison images and plots.

    Args:
        original_path (str): The file path to the original image.
        processed_path (str): The file path to the processed image.
        output_dir (str): The directory to save the output files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if files exist
    if not os.path.exists(original_path) or not os.path.exists(processed_path):
        print("Error: One or both image files not found.")
        print(f"Original path: {original_path}")
        print(f"Processed path: {processed_path}")
        return

    # Load images
    original_img = cv2.imread(original_path)
    processed_img = cv2.imread(processed_path)

    if original_img is None or processed_img is None:
        print("Error: Could not load the images. Check file paths and formats.")
        return

    # --- 1. 並排比較圖 (Side-by-Side Comparison) ---
    # Ensure images have the same dimensions for stacking
    h, w, _ = processed_img.shape
    original_resized = cv2.resize(original_img, (w, h))

    comparison_img = np.hstack([original_resized, processed_img])
    comparison_path = os.path.join(output_dir, "side_by_side_comparison.jpg")
    cv2.imwrite(comparison_path, comparison_img)
    print(f"Side-by-side comparison saved to: {comparison_path}")

    # --- 2. 差值圖 (Difference Map) ---
    # Convert images to grayscale and calculate the absolute difference
    original_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    diff_img = cv2.absdiff(original_gray, processed_gray)
    diff_path = os.path.join(output_dir, "difference_map.jpg")
    cv2.imwrite(diff_path, diff_img)
    print(f"Difference map saved to: {diff_path}")

    # --- 3. 亮度直方圖對比 (Luminosity Histogram Comparison) ---
    plt.figure(figsize=(10, 6))
    plt.title("Luminosity Distribution Comparison")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Calculate histograms
    hist_original = cv2.calcHist([original_gray], [0], None, [256], [0, 256])
    hist_processed = cv2.calcHist([processed_gray], [0], None, [256], [0, 256])

    plt.plot(hist_original, color='blue', label='Original Image')
    plt.plot(hist_processed, color='red', linestyle='--', label='Processed Image')

    plt.legend()
    plt.grid(True)
    plt.xlim([0, 256])

    histogram_path = os.path.join(output_dir, "luminosity_histogram.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"Luminosity histogram saved to: {histogram_path}")


# --- Main Execution Section ---
if __name__ == '__main__':
    # Define the base paths
    base_path = "C:/Users/rayso/Desktop/python/"
    original_images_dir = os.path.join(base_path, "before_images/")
    processed_images_dir = os.path.join(base_path, "evertings_for_final/final_output/processed_images/")
    output_base_dir = os.path.join(base_path, "evertings_for_final/final_output/comparison_results/")
    # Create the base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Loop through all files in the original images directory
    for filename in os.listdir(original_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the paths for the original and processed files
            original_image_path = os.path.join(original_images_dir, filename)

            # Assuming the processed filename has the format '1_clahe_enhanced.jpg'
            base_name, _ = os.path.splitext(filename)
            processed_filename = f"{base_name}_clahe_enhanced.jpg"
            processed_image_path = os.path.join(processed_images_dir, processed_filename)

            # Create a dedicated output directory for the current image
            current_output_dir = os.path.join(output_base_dir, base_name)

            print(f"\n--- Processing '{filename}' ---")

            # Generate and save comparison images for the current pair
            generate_comparison_images(original_image_path, processed_image_path, current_output_dir)

    print("\nAll comparison tasks completed.")