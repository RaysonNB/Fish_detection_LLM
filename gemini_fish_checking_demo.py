import google.generativeai as genai
import os
import PIL.Image
#version 1
genai.configure(api_key='AIzaSyAagkj3QC_Zt_uO7UYW_NASMCpHc8VnNxw')
model = genai.GenerativeModel("gemini-1.5-flash")
#response = model.generate_content("這是什麼魚")
img = PIL.Image.open("C:/Users/rayso/Downloads/images.jpg")
response = model.generate_content([img, "這是什麼魚(以這個格式回答:這是一隻**XX魚**。)"])
answer_text=response.text
question_text=answer_text+"這種魚在 [鯿魚、大頭、金山鰂、鯪魚、鰂魚、白條] 中嗎?"
response = model.generate_content(question_text)
print(response.text)
#cv2 version
