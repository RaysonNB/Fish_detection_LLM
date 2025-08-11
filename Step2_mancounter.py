import os
import cv2
import csv


def get_user_input_for_count(image_filename):
    """
    在終端中提示使用者輸入，並確保輸入為有效的非負整數。

    Args:
        image_filename (str): 當前顯示的圖片檔案名稱。

    Returns:
        int: 使用者輸入的魚的數量。
    """
    while True:
        try:
            # 建立一個清晰、資訊豐富的提示
            prompt = f"-> 正在顯示 [{image_filename}]：請輸入圖片中魚的數量，然後按 Enter： "
            count_str = input(prompt)
            count = int(count_str)

            if count < 0:
                print("[警告] 數量不能為負數，請重新輸入。")
                continue

            return count

        except ValueError:
            # 如果使用者輸入的不是數字，給予友善的提示
            print("[錯誤] 無效輸入。請只輸入整數數字（例如 0, 1, 5 ...）。")
        except KeyboardInterrupt:
            # 允許使用者使用 Ctrl+C 中斷程式
            print("\n[資訊] 使用者中斷了輸入。")
            return None


def save_to_csv(data, filename="manual_count_ground_truth.csv"):
    """
    將手動計數的結果儲存為CSV檔案。

    Args:
        data (list): 包含 [檔名, 數量] 的列表。
        filename (str): 要儲存的CSV檔案名稱。
    """
    if not data:
        print("沒有手動計數的結果可供儲存。")
        return

    try:
        # 使用 'w' 模式，確保每次執行都會建立一個全新的檔案
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 寫入清晰的標頭
            writer.writerow(['image_filename', 'manual_fish_count'])

            # 寫入所有資料
            writer.writerows(data)

        print(f"\n[成功] 標準答案已儲存至檔案： {filename}")

    except IOError:
        print(f"[錯誤] 無法寫入檔案 {filename}。請檢查權限或路徑。")


# --- 主執行程序 ---
def main():
    """
    主函式，用於執行整個手動計數流程。
    """
    # 1. 獲取並驗證資料夾路徑
    folder_path = input("請輸入包含海底照片的資料夾路徑： ")

    if not os.path.isdir(folder_path):
        print(f"錯誤：找不到資料夾 '{folder_path}'")
        return

    # 2. 獲取並排序圖片列表，確保處理順序一致
    try:
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
    except Exception as e:
        print(f"讀取資料夾時發生錯誤：{e}")
        return

    if not image_files:
        print(f"在 '{folder_path}' 中找不到任何支援的圖片檔案。")
        return

    # 3. 初始化並開始主迴圈
    manual_counts = []
    window_name = "Manual Fish Counting - Press ANY KEY in this window to close it after input"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 可調整視窗大小

    print("\n--- 開始手動計數 ---")
    print(f"總共找到 {len(image_files)} 張圖片需要處理。")
    print("對於每一張顯示的圖片，請在下方的終端機中輸入魚的數量。")

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # 讀取並顯示圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：無法讀取圖片 {image_file}，已跳過。")
            continue

        cv2.imshow(window_name, image)

        # 這裡的waitKey(1)是為了確保視窗能即時刷新顯示，而不會卡住等待終端輸入
        cv2.waitKey(1)

        # 在終端中獲取使用者輸入
        count = get_user_input_for_count(image_file)

        # 如果使用者中斷輸入 (Ctrl+C), 則停止並詢問是否儲存
        if count is None:
            confirm_save = input("您已中斷操作。是否要儲存目前已完成的計數結果？(y/n): ").lower()
            if confirm_save == 'y':
                break  # 跳出迴圈去儲存
            else:
                print("操作已取消，不會儲存任何結果。")
                cv2.destroyAllWindows()
                return  # 直接退出

        manual_counts.append([image_file, count])

    # 4. 迴圈結束後，銷毀視窗並儲存結果
    cv2.destroyAllWindows()

    print("\n所有圖片計數完畢。")
    save_to_csv(manual_counts)


if __name__ == "__main__":
    main()
