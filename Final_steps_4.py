import google.generativeai as genai
import os
import time
import PIL.Image
import cv2
import numpy as np
from PIL import Image
# Configuration and setup
genai.configure(api_key='GOOGLE_API') # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")
path_sample = "C:/Users/rayso/Desktop/python/evertings_for_final/images_correction" # Use raw string to handle backslashes
files = os.listdir(path_sample)
# Prepare the prompt template
sample_txt ="""
What is this(just give me the name is enough, no extra description)?
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


    image = np.array(img).copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    text = a
    position = (10, 20)  # Position on the right side
    color = (0, 255, 0)  # Green text in BGR
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55  # Adjust the font size
    thickness = 2  # Thickness of the text
    line_type = cv2.LINE_AA
    cv2.putText(image, text, position, font, font_scale, color, thickness, line_type)
    cv2.imshow("Combined Image", image)
    cv2.waitKey(0)

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
