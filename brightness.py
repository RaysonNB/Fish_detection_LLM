from huggingface_hub import from_pretrained_keras
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os

def average_brightness(a):
    arr = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    l = []
    for i in arr:
        for j in i:
            l.append(j[2])
    s = np.sum(l)
    return s/(len(arr)*len(arr[0]))
def check(a):
    for i in a:
        for j in i:
            j[0] = min(1, j[0])
            j[1] = min(1, j[1])
            j[2] = min(1, j[2])
    return a

model = from_pretrained_keras("google/maxim-s2-enhancement-lol")

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
b = 0
path = "C:/Users/brian/OneDrive/Desktop/model_clear/"

s = "C:/Users/brian/OneDrive/Desktop/Carassius_auratus_49_.jpg"
image = Image.open(s)
image = np.array(image)
h, w, c = image.shape

t = image.copy()
t = np.array(t, np.float32) / 255
t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
b = average_brightness(t)
print(b)

os.system(path + "realesrgan-ncnn-vulkan.exe -i " + s + " -o " + path + "test/" + str(b) + "_clear.jpg -n realesrgan-x4plus")

t = t * 255
cv2.imwrite(path + "test/" + str(b) + "_original.jpg", t)

while b < 0.65:
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))

    predictions = model.predict(tf.expand_dims(image, 0))

    img = predictions[1][2]

    im = np.array(img[0], np.float32) / 255
    im = cv2.resize(im, (w, h))

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = check(im)
    b = average_brightness(im)
    print(b)

    image = im * 255

    name = str(b) + ".jpg"

    cv2.imwrite("C:/Users/brian/OneDrive/Desktop/model_clear/test/" + name, image)

    os.system(path + "realesrgan-ncnn-vulkan.exe -i " + path + "test/" + name + " -o " + path + "test/" + str(b) + "_clear.jpg -n realesrgan-x4plus")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)