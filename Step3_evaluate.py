import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_and_merge_data(gemini_csv_path, manual_csv_path):
    """
    載入Gemini和手動計數的CSV檔案，並根據檔名合併它們。

    Args:
        gemini_csv_path (str): Gemini偵測結果的CSV檔案路徑。
        manual_csv_path (str): 手動計數結果（標準答案）的CSV檔案路徑。

    Returns:
        pandas.DataFrame: 一個包含合併後數據的DataFrame，如果出錯則返回None。
    """
    try:
        # 讀取兩個CSV檔案
        gemini_df = pd.read_csv(gemini_csv_path)
        manual_df = pd.read_csv(manual_csv_path)
        print("gemini", gemini_df)
        print("man", manual_df)
        # 檢查必要的欄位是否存在
        required_gemini_cols = ['image_filename', 'detected_fish_count']
        required_manual_cols = ['image_filename', 'manual_fish_count']

        if not all(col in gemini_df.columns for col in required_gemini_cols):
            print(f"錯誤：Gemini結果檔案 '{gemini_csv_path}' 缺少必要欄位。需要 {required_gemini_cols}")
            return None
        if not all(col in manual_df.columns for col in required_manual_cols):
            print(f"錯誤：手動計數檔案 '{manual_csv_path}' 缺少必要欄位。需要 {required_manual_cols}")
            return None

        # 使用 'inner' 合併，只保留兩個檔案中都存在的圖片記錄
        merged_df = pd.merge(gemini_df, manual_df, on='image_filename', how='inner')

        print("-> 資料載入與合併成功。")
        return merged_df

    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案 {e.filename}。請確認路徑是否正確。")
        return None
    except Exception as e:
        print(f"讀取或合併檔案時發生錯誤：{e}")
        return None


def analyze_and_classify(df):
    """
    分析合併後的數據，根據定義的規則計算TP, FP, FN, TN。

    Args:
        df (pandas.DataFrame): 包含合併後數據的DataFrame。

    Returns:
        dict: 一個包含 'tp', 'fp', 'fn', 'tn' 計數的字典。
    """
    tp, fp, fn, tn = 0, 0, 0, 0

    # 使用 itertuples() 進行高效遍歷
    for row in df.itertuples():
        real_count = row.manual_fish_count
        gemini_count = row.detected_fish_count

        # 規則 5: True Negative (TN)
        # 照片中真的沒有魚，且Gemini也偵測為0
        if real_count == 0 and gemini_count == 0:
            tn += 1
            continue

        # 處理 real_count 為 0 但 gemini_count > 0 的情況
        # 這是一種特殊的 False Positive，因為成功率公式會除以0
        if real_count == 0 and gemini_count > 0:
            fp += 1
            continue

        # 規則 2: 計算檢測成功率
        # 使用 np.abs 確保結果為正
        success_rate = 1.0 - (np.abs(gemini_count - real_count) / real_count)

        # 規則 3: True Positive (TP)
        if success_rate >= 0.8:
            tp += 1
        # 規則 4: False Positive (FP) 和 False Negative (FN)
        else:
            if gemini_count > real_count:
                fp += 1
            else:  # 包含 gemini_count < real_count 的情況
                fn += 1

    print("-> 結果分析與分類完成。")
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def calculate_performance_metrics(counts):
    """
    根據TP, FP, FN, TN計算召回率、精確率和準確率。

    Args:
        counts (dict): 包含 'tp', 'fp', 'fn', 'tn' 計數的字典。

    Returns:
        dict: 一個包含 'Recall', 'Precision', 'Accuracy' 的字典。
    """
    tp, fp, fn, tn = counts['tp'], counts['fp'], counts['fn'], counts['tn']
    # 計算 Precision (精確率)
    # TP / (TP + FP) -> 在所有被模型預測為正樣本的結果中，有多少是真的正樣本
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 計算 Recall (召回率/真正率)
    # TP / (TP + FN) -> 在所有真實為正樣本的結果中，有多少被模型成功預測出來
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 計算 Accuracy (準確率)
    # (TP + TN) / (TP + FP + FN + TN) -> 在所有樣本中，模型預測正確的比例
    total_samples = tp + fp + fn + tn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

    print("-> 性能指標計算完成。")
    return {'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}


def plot_metrics_barchart(metrics):
    """
    使用matplotlib繪製並顯示性能指標的條形圖。

    Args:
        metrics (dict): 包含指標名稱和值的字典。
    """
    names = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]  # 將值轉換為百分比

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(names, values, color=['#4285F4', '#34A853', '#FBBC05'])

    # --- 設定圖表視覺效果 (全英文) ---
    ax.set_ylabel('Score (%)')
    ax.set_title('Gemini Fish Detection Performance Metrics', fontsize=16)
    ax.set_ylim(0, 110)  # 設定Y軸上限，留出頂部空間
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 在每個條形圖上顯示具體數值
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f'{yval:.2f}%',
                ha='center', va='bottom', fontsize=12)

    plt.tight_layout()  # 自動調整佈局
    print("-> 圖表已生成，即將顯示...")
    plt.show()


# --- 主執行程序 ---
def main():
    print("--- 模型性能評估程序 ---")

    # 預設檔案名稱
    default_gemini_file = "gemini_detection_results.csv"
    default_manual_file = "manual_count_ground_truth.csv"

    print(f"本程式將會讀取以下兩個檔案：")
    print(f"1. Gemini 的結果: {default_gemini_file}")
    print(f"2. 您的手動計數: {default_manual_file}")
    print("請確保這兩個檔案與本程式在同一個資料夾中，或提供完整路徑。")

    gemini_file = input(f"請輸入Gemini結果檔案路徑 [直接按Enter使用預設值]: ") or default_gemini_file
    manual_file = input(f"請輸入手動計數檔案路徑 [直接按Enter使用預設值]: ") or default_manual_file

    # 1. 載入並合併數據
    merged_data = load_and_merge_data(gemini_file, manual_file)
    if merged_data is None:
        print("程式因資料讀取錯誤而終止。")
        return
    if merged_data.empty:
        print("錯誤：合併後的資料為空，請檢查兩個CSV檔案中的 'image_filename' 是否有共同的項目。")
        return

    # 2. 分析結果
    counts = analyze_and_classify(merged_data)
    print(f"分類統計: TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}, TN={counts['tn']}")

    # 3. 計算最終指標
    metrics = calculate_performance_metrics(counts)
    print(f"最終評估指標:")
    print(f"  - Precision: {metrics['Precision']:.4f}")
    print(f"  - Recall:    {metrics['Recall']:.4f}")
    print(f"  - Accuracy:  {metrics['Accuracy']:.4f}")

    # 4. 繪製並顯示圖表
    plot_metrics_barchart(metrics)

    print("\n--- 程序執行完畢 ---")


if __name__ == "__main__":
    main()
