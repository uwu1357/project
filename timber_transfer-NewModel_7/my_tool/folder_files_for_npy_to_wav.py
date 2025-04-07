import os
import numpy as np
import soundfile as sf

def npy_to_wav(input_file: str, output_file: str, sr: int = 16000):
    """
    將 .npy 檔案轉換為 .wav 檔案。
    
    參數:
    - input_file (str): 輸入 .npy 檔案的路徑
    - output_file (str): 輸出 .wav 檔案的路徑
    - sr (int): 取樣率，默認為 16000
    """
    # 讀取 .npy 檔案
    y = np.load(input_file)
    
    # 保存為 .wav 檔案
    sf.write(output_file, y, sr)
    print(f"音訊檔案已成功轉換為 {output_file}！")

def convert_folder_to_wav(input_folder: str, output_folder: str, sr: int = 16000):
    """
    將資料夾中的所有 .npy 檔案轉換為 .wav 格式的數據。
    
    參數:
    - input_folder (str): 包含 .npy 檔案的資料夾路徑
    - output_folder (str): 輸出 .wav 檔案的資料夾路徑
    - sr (int): 取樣率，默認為 16000
    """
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
            npy_to_wav(input_file, output_file, sr)

# 範例使用
input_folder = "C:\keyboard_npy"  # 輸入 .npy 檔案的資料夾路徑
output_folder = "C:\\timber_transfer-NewModel_3\\keyboard_wav"  # 輸出 .wav 檔案的資料夾路徑

convert_folder_to_wav(input_folder, output_folder)