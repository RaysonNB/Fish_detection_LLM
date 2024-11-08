import os

def rename_files(directory, base_name, extension):
    # 確保輸入的資料夾存在
    if not os.path.exists(directory):
        print("指定的資料夾不存在！")
        return

    # 獲取資料夾中的所有檔案
    files = os.listdir(directory)
    print(files)
    # 只保留前 50 個檔案
    files = files[:70]
    '''
    for index, filename in enumerate(files, start=1):
        # 確定新的檔案名稱
        new_name = f"{base_name}_{index}.{extension}"
        new_file = os.path.join(directory, new_name)

        # 检查新文件名是否已存在
        if not os.path.exists(new_file):
            # 形成完整的檔案路徑
            old_file = os.path.join(directory, filename)

            # 確保不會重命名為相同名稱
            if old_file != new_file:
                # 重命名檔案
                os.rename(old_file, new_file)
                print(f"已將 {old_file} 重命名為 {new_file}")
        else:
            print(f"文件 {new_name} 已存在，跳過重命名。")'''

# 使用範例
directory_path = r"C:\Users\rayso\Downloads\現實相片-20241101T052224Z-001\現實相片\brightness_range"

# 執行改名
rename_files(directory_path, base_name='brightness_range', extension='png')