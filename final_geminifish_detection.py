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
list1=[]
list2=[]
#O(5*50*3)
for g in range(0,5):
    for j in range(1,51):
        list1=[]
        for i in range(3):
            list2=[str(file_name[g])]
            print(i,str(img_name[g])+str(j))
            aaa="C:/Users/rayso/Downloads/S4A_胡_isef_dataset-20241002T074044Z-001/S4A_胡_isef_dataset/單魚/水塘內/"+str(file_name[g])+"/"+str(img_name[g])+str(j)+".png"
            print(aaa)
            img = PIL.Image.open(aaa)
            response = model.generate_content([img, "這是什麼魚(以這個格式回答:這是一隻 *XX魚* 。)"])
            answer_text=response.text
            str1 = ""
            st, ed = -1, -1
            for i1 in range(0, len(answer_text)):
                if answer_text[i1] == '*' and st == -1:
                    st = i1
                if answer_text[i1] == '*' and st != -1:
                    ed = i1
                print(st, ed)
            str1=answer_text[st+1:ed]
            print("*"+str1+"*")
            list2.append(str1)
        list1.append(list2)
    ans.append(list1)
    print(ans)
