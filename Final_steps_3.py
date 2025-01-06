from PIL import Image
import numpy as np
import cv2
import os

# Directories for input and output images
input_directory = "C:/Users/rayso/Desktop/python/evertings_for_final/file_saving/"  # Folder with input images
output_directory = "C:/Users/rayso/Desktop/python/evertings_for_final/images_correction/"

files1 = os.listdir(input_directory)
files2 = os.listdir(output_directory)

# Iterate through all image files in the input directory
for i in range(len(files2)):
    # Load the first image using PIL
    image_path1 = input_directory + files1[i]
    image1 = Image.open(image_path1)
    image1 = np.array(image1)  # Convert to numpy array (RGB format by default)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    # Load the second image using PIL
    image_path2 = output_directory + files2[i]
    image2 = Image.open(image_path2)
    image2 = np.array(image2)  # Convert to numpy array (RGB format by default)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Adjust the font size
    thickness = 1  # Thickness of the text
    line_type = cv2.LINE_AA
    resized_image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # Add "Before" text on the left side (first image)
    before_text = "Before"
    before_position = (10, 50)  # Position on the left side
    before_color = (0, 0, 255)  # Blue text in BGR
    cv2.putText(resized_image1, before_text, before_position, font, font_scale, before_color, thickness, line_type)

    # Add "After" text on the right side (second image)
    after_text = "After"
    after_position = (10, 50)  # Position on the right side
    after_color = (0, 255, 0)  # Green text in BGR
    cv2.putText(image2, after_text, after_position, font, font_scale, after_color, thickness, line_type)
    # Resize image1 to match the dimensions of image2


    # Combine the images horizontally (side-by-side)
    combined_image = np.hstack((resized_image1, image2))

    # Text properties

    # Display the combined image
    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(0)  # Wait for a key press to move to the next image

    # Optionally, save the combined image
    save_path = f"C:/Users/rayso/Desktop/python/evertings_for_final/combined_image_{i}.png"
    cv2.imwrite(save_path, combined_image)

# Close all OpenCV windows
cv2.destroyAllWindows()
