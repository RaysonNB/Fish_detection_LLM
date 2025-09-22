import cv2
import numpy as np
import os


def enhance_image_clahe(input_image_path, output_image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    使用自適應直方圖均衡化 (CLAHE) 演算法優化圖片光度，使其更清晰。

    Args:
        input_image_path (str): 輸入圖片的路徑。
        output_image_path (str): 輸出圖片的儲存路徑。
        clip_limit (float): 裁剪對比度限制。數值越大，對比度增強越明顯。
        tile_grid_size (tuple): 圖片分割網格的大小 (寬, 高)。
    """
    # 檢查輸入檔案是否存在
    if not os.path.exists(input_image_path):
        print(f"錯誤：找不到輸入檔案 {input_image_path}")
        return

    # 讀取圖片，以彩色模式載入
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    # 如果讀取失敗，則終止
    if img is None:
        print(f"錯誤：無法讀取圖片 {input_image_path}")
        return

    # 將圖片從 BGR 色彩空間轉換為 LAB 色彩空間
    # LAB 空間將亮度 (L) 與色彩分開，以便單獨調整光度
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 分離 LAB 通道
    l, a, b = cv2.split(lab)

    # 創建 CLAHE 物件
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 應用 CLAHE 到亮度 (L) 通道
    enhanced_l = clahe.apply(l)

    # 將優化後的亮度通道與原始的 a, b 通道合併
    merged_lab = cv2.merge((enhanced_l, a, b))

    # 將圖片從 LAB 空間轉換回 BGR 空間
    enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # 儲存優化後的圖片
    cv2.imwrite(output_image_path, enhanced_img)

    print(f"圖片已成功優化並儲存至：{output_image_path}")


# --- 主要執行程式 ---
if __name__ == '__main__':
    # 設定輸入和輸出資料夾路徑
    input_folder = r'C:\Users\rayso\Desktop\python\evertings_for_final\images_correction'
    output_folder = r'C:\Users\rayso\Desktop\python\evertings_for_final\light'

    # 確保輸出資料夾存在，如果不存在則創建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已創建輸出資料夾：{output_folder}")

    # 迴圈處理輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        # 檢查檔案是否為圖片格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 執行圖片優化函數
            enhance_image_clahe(input_path, output_path)

    print("所有圖片處理完畢。")