import os
import cv2
import numpy as np
import google.generativeai as genai
import json
import csv
from PIL import Image
import random
import re
import time

# --- 1. 環境設定：配置您的 Google API 金鑰 ---
# 請將 'YOUR_API_KEY' 替換為您自己的 Gemini API 金鑰
# 建議使用環境變數來管理您的金鑰，以策安全
try:
    # 範例：從環境變數讀取
    # GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    # genai.configure(api_key=GOOGLE_API_KEY)

    # 為了方便直接運行，此處我們直接配置
    # *** 請在這裡貼上您的 API 金鑰 ***
    API_KEY = "YOUR_API_KEY"
    if API_KEY == "YOUR_API_KEY":
        raise ValueError("請替換 'YOUR_API_KEY' 為您真實的Google API金鑰。")
    genai.configure(api_key=API_KEY)

except ValueError as e:
    print(e)
    exit()  # 如果沒有金鑰，則終止程式


def create_loading_screen(width=800, height=600):
    """建立一個顯示“正在處理”訊息的黑色畫面"""
    loading_img = np.zeros((height, width, 3), dtype=np.uint8)
    text = "Please wait, drawing bounding boxes..."

    # 選擇一個合適的字體大小和粗細
    font_scale = 1.2
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 計算文字尺寸以將其置中
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (loading_img.shape[1] - text_size[0]) // 2
    text_y = (loading_img.shape[0] + text_size[1]) // 2

    cv2.putText(loading_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return loading_img


def parse_bounding_box(response):
    # Use a regex to find the JSON string between "******"
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        raise ValueError("No JSON list found in the response.")

    json_string = match.group(0)
    print("Cleaned JSON string:", json_string)
    try:
        dict_list = json.loads(json_string)
        numbers1 = len(dict_list)
        parsed_boxes = []
        for box in dict_list:
            numbers = box["box_2d"]
            label = box["label"]
            parsed_boxes.append((numbers, label))

        return parsed_boxes
    except Exception as e:
        print(f"錯誤：調用Gemini API時發生問題或解析JSON失敗。 {e}")
        return []



def call_gemini_vision_api(image_path):
    print("api")
    time.sleep(2.55)
    """
    調用Gemini 1.5 Pro API來偵測圖片中的魚。

    Args:
        image_path (str): 圖片的檔案路徑。

    Returns:
        list: 一個包含所有偵測到的魚的邊界框座標的列表。
              如果失敗或未偵測到，則返回空列表。
    """
    print(f"-> 正在向 Gemini API 傳送圖片： {os.path.basename(image_path)}")

    # 設定模型，Gemini 1.5 Pro 是目前最強大的多模態模型之一
    model = genai.GenerativeModel('gemini-2.0-flash')

    # 準備圖片
    try:
        img = Image.open(image_path)
    except IOError:
        print(f"錯誤：無法開啟圖片檔案 {image_path}")
        return []

    # 設計一個精確的、要求JSON輸出的Prompt
    prompt = """
    '''
    if there is no any fish in the image, please say "there is none fish"
    else:
    now you are going to help me identify the fishes position and type in the image
    please Return bounding boxes for all fishes in the image in the following format as a list. 
    when the fish possibility is bigger than 95% and the fish is not too small than make a directly bounding boxes
    the bounding box must have ymin, xmin, ymax, xmax,, 4 factors
    please make the bounding boxes carefully
    each fish must have a fish name please help me to identify 
    the maxium count of fish in 7

    don't give me 'fish' in 'label', fish must have a name
    output example:

    Here are the bounding box detections:
    ******

    [{"box_2d":[ymin, xmin, ymax, xmax], "label":"fish_name"},
    {"box_2d":[ymin, xmin, ymax, xmax], "label":"fish_name"},
    {"box_2d":[ymin, xmin, ymax, xmax], "label":"fish_name"},
    {"box_2d":[ymin, xmin, ymax, xmax], "label":"fish_name"},
    {"box_2d":[ymin, xmin, ymax, xmax], "label":"fish_name"}]
    ******
    """
    response = model.generate_content([prompt, img], request_options={"timeout": 120})
    # print(response)
    if "none" in response.text or "no "in response.text:
        return "none"
    return parse_bounding_box(response.text)


def draw_bounding_boxes(image, detections):
    """
    在圖片上繪製邊界框。

    Args:
        image (numpy.ndarray): OpenCV格式的圖片。
        detections (list): 從API獲取的偵測結果列表。

    Returns:
        tuple: (帶有邊界框的圖片, 偵測到的魚的數量)
    """
    h, w, _ = image.shape
    annotated_image = image.copy()
    fish_count = len(detections)
    label_colors = {}
    for bounding_box, label in detections:
        width, height = annotated_image.shape[1], annotated_image.shape[0]
        ymin, xmin, ymax, xmax = bounding_box
        x1 = int(xmin / 1000 * width)
        y1 = int(ymin / 1000 * height)
        x2 = int(xmax / 1000 * width)
        y2 = int(ymax / 1000 * height)

        if label not in label_colors:
            color = np.random.randint(0, 256, (3,)).tolist()
            label_colors[label] = color
        else:
            color = label_colors[label]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        box_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        text_bg_x1 = x1
        text_bg_y1 = y1 - text_size[1] - 5
        text_bg_x2 = x1 + text_size[0] + 8
        text_bg_y2 = y1

        cv2.rectangle(annotated_image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        cv2.putText(annotated_image, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)

    return annotated_image, fish_count


def save_to_csv(data, filename="gemini_detection_results.csv"):
    """將偵測結果儲存為CSV檔案。"""
    if not data:
        print("沒有偵測結果可供儲存。")
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_filename', 'detected_fish_count'])
            writer.writerows(data)
        print(f"\n[成功] 偵測結果已儲存至 {filename}")
    except IOError:
        print(f"[錯誤] 無法寫入檔案 {filename}。")


# --- 主執行程序 ---
def main():
    folder_path = "C:/Users/rayso/Desktop/python/before_images/"

    if not os.path.isdir(folder_path):
        print(f"錯誤：找不到資料夾 '{folder_path}'")
        return

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"在 '{folder_path}' 中找不到任何圖片檔案。")
        return

    detection_results = []
    window_name = "Gemini Fish Detection - Press SPACE for Next, ESC to Exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 可調整視窗大小

    print("\n--- 開始處理圖片 ---")
    print(f"總共找到 {len(image_files)} 張圖片。")
    print("在圖片視窗中，按下 [空格鍵] 觀看下一張，按下 [ESC] 鍵退出程序。")

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # 1. 顯示“請稍等”畫面
        loading_screen = create_loading_screen()
        cv2.imshow(window_name, loading_screen)
        cv2.waitKey(1)  # 必須有短暫等待，才能確保視窗刷新

        # 2. 調用API並處理圖片
        detections = call_gemini_vision_api(image_path)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"警告：無法讀取圖片 {image_file}，已跳過。")
            continue
        annotated_image=original_image
        if detections != "none":
            annotated_image, fish_count = draw_bounding_boxes(original_image, detections)
            detection_results.append([image_file, fish_count])
        else:
            detection_results.append([image_file, 0])

        # 3. 顯示繪製了bounding boxes的照片
        cv2.imshow(window_name, annotated_image)
        start_time=time.time()
        print(abs(time.time()-start_time))
        # 4. 等待使用者操作
        '''
        key = cv2.waitKey(0) & 0xFF
        if abs(time.time()-start_time)>=3:
            continue
        elif key == 27:  # ESC鍵
            print("\n使用者提前退出程序。")
            break
        elif key == ord(' '):  # 空格鍵
            continue
        '''
    # 5. 處理結束後儲存結果
    print("\n所有圖片處理完畢。")
    save_to_csv(detection_results)

    # 銷毀所有視窗
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
