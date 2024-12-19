import google.generativeai as genai
import os
import time
import PIL.Image
import cv2
# Configuration and setup
genai.configure(api_key='AIzaSyAagkj3QC_Zt_uO7UYW_NASMCpHc8VnNxw') # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")
path_sample = r"C:\Users\rayso\Desktop\python\evertings_for_final\images_correction" # Use raw string to handle backslashes
files = os.listdir(path_sample)
# Prepare the prompt template
sample_txt = """
question1: What is this?
question1_format: (1): ***fish_name***
"""


# Process each image
for filename in files:
    path = os.path.join(path_sample, filename)

    try:
        img = PIL.Image.open(path)
    except PIL.UnidentifiedImageError:
        print(f"Skipping file {filename}: Could not open as an image.")
        continue

    response = model.generate_content([img, sample_txt])
    file_data_string = response.text
    print(file_data_string)
    base_path = "C:/Users/rayso/Desktop/python/evertings_for_final/video_detail_information"
    folder_name = file_data_string
    folder_path = os.path.join(base_path, folder_name)
    base_path+=str(folder_name)
    os.makedirs(folder_path, exist_ok=True)  # `exist_ok=True` avoids errors if the folder already exists
    print(f"Folder '{folder_name}' created successfully at: {folder_path}")

    cv2.imwrite(base_path, img)
    time.sleep(4)