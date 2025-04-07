import numpy as np
import librosa

class FrequencyToMIDI:
    def __init__(self, npy_file_path):
        """
        初始化類別，載入頻率特徵數據
        """
        self.npy_file_path = npy_file_path
        self.frequency_data = self.load_npy_file()

    def load_npy_file(self):
        """
        讀取 .npy 文件
        """
        try:
            frequency_data = np.load(self.npy_file_path)
            print(f"成功載入檔案: {self.npy_file_path}")
            print(f"數據形狀: {frequency_data.shape}")
            print(f"數據範例: {frequency_data[:10]}")  # 顯示前 10 筆頻率
            return frequency_data
        except Exception as e:
            print(f"讀取檔案失敗: {e}")
            raise

    def calculate_midi(self):
        """
        計算 MIDI 音符數值
        """
        if len(self.frequency_data) == 0:
            raise ValueError("頻率數據為空，無法計算 MIDI")
        
        # 計算主頻率，這裡取平均值（也可以取最大值或其他方法）
        main_frequency = np.mean(self.frequency_data)
        print(f"計算出的主頻率: {main_frequency:.2f} Hz")

        # 將主頻率轉換為 MIDI 數值
        try:
            midi_value = int(round(librosa.hz_to_midi(main_frequency)))
            print(f"對應的 MIDI 音符: {midi_value}")
            return midi_value
        except ValueError:
            print("主頻率無法對應至有效的 MIDI 音符範圍")
            return None

# 範例使用
if __name__ == "__main__":
    # 輸入 .npy 文件的路徑
    npy_file_path = "C:\\timber_transfer-NewModel_3\\timber_transfer-NewModel\\test\\test\\piano_save_20250107_030246_frequency_c.npy"

    # 初始化處理類別
    freq_to_midi = FrequencyToMIDI(npy_file_path)

    # 計算 MIDI 音符
    midi_value = freq_to_midi.calculate_midi()

    # 輸出結果
    if midi_value is not None:
        print(f"檔案的 MIDI 音符數值為: {midi_value}")
    else:
        print("計算失敗，請確認數據正確性")
