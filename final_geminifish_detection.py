import google.generativeai as genai
import os
import PIL.Image
#version 1
genai.configure(api_key='AIzaSyAagkj3QC_Zt_uO7UYW_NASMCpHc8VnNxw')
model = genai.GenerativeModel("gemini-1.5-flash")
#response = model.generate_content("這是什麼魚")
file_name=["鯿魚(Abramis brama)","金山鰂(Tilapia)","鯪魚(Cyprinidae)","鯽魚(Carassius auratus)","鱅魚(Hypophthalmichthys)"]
img_name=["Abramis_brama_","tilapia_","Cyprinidae_","Carassius_auratus_","Hypophthalmichthys_"]
ans=[]
for g in range(1,2):
    for j in range(1,51):
        list1=[]
        for i in range(3):
            list2=[]
            print(j,i)
            aaa="C:/Users/rayso/Downloads/S4A_胡_isef_dataset-20241002T074044Z-001/S4A_胡_isef_dataset/單魚/水塘內/"+str(file_name[g])+"/"+str(img_name[g])+str(j)+".png"
            print(aaa)
            img = PIL.Image.open(aaa)
            response = model.generate_content([img, "這是什麼魚(以這個格式回答:這是一隻**XX魚**。)"])
            answer_text=response.text
            print(answer_text)
            list2.append(answer_text)
            question_text=answer_text+"這種魚在 [鯿魚、鱅魚、金山鰂、鯪魚、鰂魚]中嗎? 回答  **是**  或  **不是** "
            response = model.generate_content(question_text)
            print(response.text)
            list2.append(response.text)
            list1.append(list2)
        ans.append(list1)
        print(ans)
