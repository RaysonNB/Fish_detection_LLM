from huggingface_hub import from_pretrained_keras
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os

def average_brightness(a):
    arr = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    return np.mean(arr[:, :, 2])

def check(a):
    a = np.clip(a, 0, 1)
    return a

model = from_pretrained_keras("google/maxim-s2-enhancement-lol")

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
path = "C:/Users/rayso/Desktop/python/"
input_directory = "C:/Users/rayso/Desktop/python/evertings_for_final/file_saving/"  # Folder with input images
output_directory = "C:/Users/rayso/Desktop/python/evertings_for_final/images_correction/"
os.makedirs(output_directory, exist_ok=True)

# Iterate through all image files in the input directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        image_path = os.path.join(input_directory, filename)

        # Load and process the image
        image = Image.open(image_path)
        image = np.array(image)
        h, w, c = image.shape

        t = image.copy()
        t = np.array(t, np.float32) / 255
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        b = average_brightness(t)

        # Run Real-ESRGAN for initial enhancement
        os.system(f"{path}realesrgan-ncnn-vulkan.exe -i {image_path} -o {output_directory}{filename}_best_clear.png -n realesrgan-x4plus")

        t = t * 255
        #cv2.imwrite(f"{output_directory}{b}_original.jpg", t)

        while b < 0.4:
            image_tensor = tf.convert_to_tensor(image)
            image_tensor = tf.image.resize(image_tensor, (256, 256))

            predictions = model.predict(tf.expand_dims(image_tensor, 0))

            img = predictions[1][2]
            im = np.array(img[0], np.float32) / 255
            im = cv2.resize(im, (w, h))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im = check(im)
            b = average_brightness(im)
            print(f"Processing {filename}: Brightness = {b}")

            # Only save images where brightness is between 0.2 and 0.3
            if 0.2 <= b <= 0.4:
                image_output_path = f"{output_directory}{filename}_best_clear.png"
                image_to_save = im * 255
                cv2.imwrite(image_output_path, image_to_save)

                os.system(f"{path}realesrgan-ncnn-vulkan.exe -i {image_output_path} -o {output_directory}{filename}_best_clear.png -n realesrgan-x4plus")

            # Update the image for further processing in the loop
            image = im * 255
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the intermediate image
            cv2.imshow("Intermediate Image", im)  # Show the image
            key = cv2.waitKey(1)  # Wait for 1ms (use a larger value if needed to observe the image)
            if key == ord('q'):  # Allow the user to quit the loop by pressing 'q'
                break

        # Destroy OpenCV window after processing the image
        cv2.destroyAllWindows()