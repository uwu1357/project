import os
import pyaudio
import wave
from datetime import datetime

class AudioRecorder:
    def __init__(self):
        # 音訊參數
        #self.base_dir = "recordings"  # 預設錄音檔案儲存目錄
        self.base_dir = "nsynth-subset5/audio/wav"  # 預設錄音檔案儲存目錄
        self.rate = 16000             # 預設取樣率
        self.channels = 1             # 預設單聲道
        self.chunk = 1024             # 音訊塊大小
        self.format = pyaudio.paInt16 # 音訊格式

        # 初始化 PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

        # 確保資料夾存在
        os.makedirs(self.base_dir, exist_ok=True)

        # 動態生成檔案名稱
        self.output_filename = self.generate_filename()

    def generate_filename(self):
        """根據時間戳生成檔案名稱"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #return os.path.join(self.base_dir, f"{timestamp}.wav")
        return os.path.join(self.base_dir, f"recording_{timestamp}_100-XXX-100.wav")

    def start_recording(self):
        """開始錄音"""
        if not self.is_recording:
            self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer=self.chunk)
            self.frames = []
            self.is_recording = True
            print("錄音開始...")

    def stop_recording(self):
        """停止錄音並儲存檔案"""
        if self.is_recording:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            #self.save_audio()
            print("錄音結束。")

    def save_audio(self):
        """將錄製的音訊儲存為 wav 檔案"""
        with wave.open(self.output_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        print(f"音訊檔案已儲存為 {self.output_filename}")

    def record_chunk(self):
        """錄製一個音訊塊"""
        if self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)


