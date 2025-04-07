import librosa
import numpy as np
import os

def audio_to_npy(audio_file: str, output_file: str, sr: int = 16000):
    """
    將音頻檔案轉換為 .npy 格式的數據。
    
    參數:
    - audio_file (str): 要轉換的音頻檔案的路徑（.wav 等格式）
    - output_file (str): 輸出的 .npy 檔案路徑
    - sr (int): 取樣率，默認為 16000
    """
    # 使用 librosa 讀取音頻檔案
    y, sr = librosa.load(audio_file, sr=sr)

    # 保存為 .npy 檔案
    np.save(output_file, y)
    print(f"音訊檔案已成功轉換為 {output_file}！")

def convert_folder_to_npy(input_folder: str, output_folder: str, sr: int = 16000):
    """
    將資料夾中的所有音頻檔案轉換為 .npy 格式的數據。
    
    參數:
    - input_folder (str): 包含音頻檔案的資料夾路徑
    - output_folder (str): 輸出 .npy 檔案的資料夾路徑
    - sr (int): 取樣率，默認為 16000
    """
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            audio_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".npy")
            audio_to_npy(audio_file, output_file, sr)

# 範例使用
input_folder = "C:\\timber_transfer-NewModel_3\\test"  # 輸入音頻檔案的資料夾路徑
output_folder = "C:\\timber_transfer-NewModel_3\\test\\npy"  # 輸出 .npy 檔案的資料夾路徑

convert_folder_to_npy(input_folder, output_folder)