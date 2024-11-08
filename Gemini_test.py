import google.generativeai as genai
import os
import time
import PIL.Image
#version 1
list_a=["CA_ans"]
path_sample="C:/Users/rayso/Downloads/現實相片-20241101T052224Z-001/Data"
files = os.listdir(path_sample)
main={}
list_T=[]
list_AB=[]
list_HY=[]
list_CY=[]
list_CA=[]
hhjjh=0
for i in list_a:
    list_y = []
    gggg = path_sample + "/" + i
    files = os.listdir(gggg)
    print(files)
    for j in files:
        path=gggg+"/"+str(j)

        print(path)
        genai.configure(api_key='AIzaSyAagkj3QC_Zt_uO7UYW_NASMCpHc8VnNxw')
        model = genai.GenerativeModel("gemini-1.5-flash")
        #response = model.generate_content("這是什麼魚")
        img = PIL.Image.open(path)
        sample_txt='''
        question1: What fish is this?
        question2: How many percentage of this fish being a Abramis_brama
        question3: How many percentage of this fish being a Cyprinidae
        question4: How many percentage of this fish being a Hypophyhalmichthys
        question5: How many percentage of this fish being a Carassius_auratus
        question6: How many percentage of this fish being a Thilapa
        question1_format: (1): ***fish_name***
        question2_format: (2): XX%
        question3_format: (3): XX%
        question4_format: (4): XX%
        question5_format: (5): XX%
        question6_format: (6): XX%
        '''

        percentage_list=["fish_name","Abramis_brama","Cyprinidae","Hypophyhalmichthys","Carassius_auratus","Thilapa"]
        response = model.generate_content([img, sample_txt])

        file_data_string = response.text

        data_dict = {}

        lines = file_data_string.strip().split('\n')
        cntt=0
        check=0
        for line in lines:
            parts = line.split(': ')
            key = parts[0].strip('()')
            value = parts[1].strip()
            data_dict[percentage_list[cntt]] = value
            if value!="1%" and value!="5%" and value!="0%" and value!="10%": check+=1
            cntt+=1
        if check>1:
            list_y.append("ys")
        else:
            list_y.append("no")
        name=i+"_"+str(j)
        main[name] = data_dict
        data_dict={}
        print(file_data_string)
        print(main)
        print(list_y)
        time.sleep(4)
    if(hhjjh==0): list_T=list_y
    if (hhjjh == 1): list_AB = list_y
    if (hhjjh == 2): list_HY = list_y
    if (hhjjh == 3): list_CA = list_y
    hhjjh+=1
    print(i,list_y)

print("T",list_T)
print("AB",list_AB)
print("HY",list_HY)
print("CA ",list_CA)
