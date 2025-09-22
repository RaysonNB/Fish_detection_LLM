import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# --- Functions for Image Comparisons ---
def create_split_screen_comparison(original_img, processed_img, window_name):
    """
    Creates an interactive split-screen comparison with a slider.
    """
    h, w, _ = processed_img.shape
    original_resized = cv2.resize(original_img, (w, h))

    def on_trackbar(x):
        split_point = x
        temp_img = processed_img.copy()
        temp_img[:, split_point:] = original_resized[:, split_point:]
        cv2.imshow(window_name, temp_img)

    cv2.namedWindow(window_name)
    cv2.createTrackbar('Split', window_name, w // 2, w, on_trackbar)
    on_trackbar(w // 2)

    print(f"Displaying '{window_name}'. Drag the slider or press 'q' to quit.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def plot_rgb_histograms(original_img, processed_img, output_path):
    """
    Plots a comparison of RGB histograms for both images.
    """
    processed_img_resized = cv2.resize(processed_img, (original_img.shape[1], original_img.shape[0]))

    plt.figure(figsize=(15, 5))
    colors = ('b', 'g', 'r')

    # Original Image Histograms
    plt.subplot(1, 2, 1)
    plt.title('Original Image RGB Histogram')
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([original_img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 256])

    # Processed Image Histograms
    plt.subplot(1, 2, 2)
    plt.title('Processed Image RGB Histogram')
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([processed_img_resized], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"RGB histogram comparison saved to: {output_path}")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Define your base paths
    base_path = "C:/Users/rayso/Desktop/python/"
    original_images_dir = os.path.join(base_path, "before_images/")
    processed_images_dir = os.path.join(base_path, "evertings_for_final/final_output/processed_images/")
    output_base_dir = os.path.join(base_path, "comparison_results/")
    os.makedirs(output_base_dir, exist_ok=True)

    # Loop through all files in the original images directory
    for filename in os.listdir(original_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\n--- Processing '{filename}' ---")

            # Construct the paths
            original_image_path = os.path.join(original_images_dir, filename)
            base_name, _ = os.path.splitext(filename)
            processed_filename = f"{base_name}_clahe_enhanced.jpg"
            processed_image_path = os.path.join(processed_images_dir, processed_filename)

            # Check if the corresponding processed file exists
            if not os.path.exists(processed_image_path):
                print(f"Skipping: Processed file '{processed_filename}' not found.")
                continue

            # Load images
            original_img = cv2.imread(original_image_path)
            processed_img = cv2.imread(processed_image_path)

            if original_img is None or processed_img is None:
                print("Error: Could not load one or both images. Skipping.")
                continue

            # Create and display the interactive split-screen comparison
            create_split_screen_comparison(original_img, processed_img, f"Comparison for {filename}")

            # Plot and save the RGB histogram comparison
            histogram_path = os.path.join(output_base_dir, f"{base_name}_rgb_hist_comp.png")
            plot_rgb_histograms(original_img, processed_img, histogram_path)

    print("\nAll comparison tasks completed.")