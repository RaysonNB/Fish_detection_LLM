import google.generativeai as genai
import os
import time
import PIL.Image
import cv2
import numpy as np
# Configuration and setup
genai.configure(api_key='AIzaSyAvyiK4rsV3C_KQUJEXJnSEL2qhtOBhGmY') # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")
path_sample = "C:/Users/rayso/Desktop/python/evertings_for_final/images_correction" # Use raw string to handle backslashes
files = os.listdir(path_sample)
# Prepare the prompt template
sample_txt ="""
What is this?
answer_format(not too long limted 30 words): ***fish_name***
"""


# Process each image
for filename in files:
    path = os.path.join(path_sample, filename)
    print(path)
    img = PIL.Image.open(path)
    response = model.generate_content([img, sample_txt])
    file_data_string = response.text
    print(file_data_string)
    import re
    text = file_data_string
    num = 0
    for i in range(len(text)):
        if text[i] == "*":
            num = i
            break
    a = text[i:-1]
    a = a.replace("*", "")
    a = a.replace(" ", "_")
    print(a)
    base_path = "C:/Users/rayso/Desktop/python/evertings_for_final/video_detail_information"
    folder_name = a
    print(folder_name)
    folder_path = os.path.join(base_path, folder_name)
    base_path+="/"+str(folder_name)

    os.makedirs(folder_path, exist_ok=True)  # `exist_ok=True` avoids errors if the folder already exists
    print(f"Folder '{folder_name}' created successfully at: {folder_path}")
    files_1 = os.listdir(base_path)
    number1=len(files_1)+1
    file_name1=base_path+"/"+str(folder_name)+str(number1)+".png"
    img.save(file_name1)

    time.sleep(4)