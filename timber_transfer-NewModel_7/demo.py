from audio_recorder import AudioRecorder  # 引入錄音類別
from components.timbre_transformer.utils import extract_loudness, get_A_weight
from components.timbre_transformer.utils import extract_pitch, get_extract_pitch_needs
from components.timbre_transformer.TimberTransformer import TimbreTransformer
from tools.utils import cal_loudness_norm
from data.dataset import NSynthDataset
import sys
import os

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QMessageBox, QStackedWidget, QLabel, QFileDialog, QProgressBar
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import shutil
from PyQt6.QtCore import QRect
from PyQt6 import QtCore
from PyQt6.QtCore import QDir
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QSpacerItem, QSizePolicy
from PyQt6.QtCore import QTimer
import torch
import random
import numpy as np
from numpy import ndarray
import soundfile as sf
import time
import pygame
import threading
import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torchaudio
from pydub import AudioSegment
from datetime import datetime
sys.path.append(".")
# from frequency_to_midi import FrequencyToMIDI
# from feature_extractor import FeatureExtractor


def transform_frequency(frequency, semitone_shift):
    transformed_frequency = frequency * (2 ** (semitone_shift / 12))
    return transformed_frequency


class InstrumentConverterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.paths = {
            "Bass": ".\\nsynth-subset5\\Bass\\test\\signal",
            "Brass": ".\\nsynth-subset5\\Brass\\test\\signal",
            "Flute": ".\\nsynth-subset5\\Flute\\test\\signal",
            "Guitar": ".\\nsynth-subset5\\Guitar\\test\\signal",
            "Piano": ".\\nsynth-subset5\\Piano\\signal",
            "Mallet": ".\\nsynth-subset5\\Mallet\\test\\signal",
            "Audio": ".\\nsynth-subset5\\audio\\signal",
        }

        self.image_paths = {
            "Bass": "MI/Bass.png",
            "Brass": "MI/Brass.png",
            "Flute": "MI/Flute.png",
            "Guitar": "MI/Guitar.png",
            "Piano": "MI/Piano.png",
            "Mallet": "MI/Mallet.png",
        }
        self.image_paths_1 = {
            "Bass": "your_image/bass.png",
            "Brass": "your_image/brass.png",
            "Flute": "your_image/flute.png",
            "Guitar": "your_image/guitar.png",
            "Piano": "your_image/piano.png",
            "Mallet": "your_image/mallet.png",
            "Audio": "your_image/musical.png",
        }
        self.right_buttons = {
            "Play Music": self.play_music,
            "Pause": self.pause_music,
            "Visual Keyboard": self.switch_piano_page,
            "Microphone": self.switch_recording_page,
            "NEXT": self.switch_page,
        }
        self.list_widget = QListWidget(self)
        self.stacked_widget = QStackedWidget(self)
        self.page1 = QWidget()
        self.page2 = QWidget()
        self.page3 = QWidget()
        self.page4 = QWidget()
        self.setup_ui()
        self.setup_page2_ui()
        self.setup_page3_ui()
        self.setup_page4_ui()
        pygame.mixer.init()
        self.wav_file = None
        self.output_dir = "nsynth-subset5/audio/test"
        self.clicked_notes = []

        pt_dir = ".\\pt_file"
        run_name = "decoder_v21_6_addmfftx3_energy_ftimbreE"
        self.current_pt_file_name = f"{run_name}_generator_best_12.pt"
        self.pt_file = f"{pt_dir}/{self.current_pt_file_name}"
        self.pt_file_list = [self.pt_file]
        self.model = TimbreTransformer(
            is_train=False, is_smooth=True, timbre_emb_dim=256)
        self.dataset = NSynthDataset(
            data_mode="test", sr=16000, frequency_with_confidence=True)
        self.source_audio_file_name = None
        self.target_audio_file_name = None
        self.model_input_selection = ("source", "source")
        self.model.eval()
        self.model.load_state_dict(torch.load(
            self.pt_file, map_location=torch.device('cpu')))

    def sample_data(self, t: str = "source"):
        # 動態更新檔案列表
        self.dataset.update_audio_list()

        fn_with_path = random.choice(self.dataset.audio_list)
        fn = fn_with_path.split("/")[-1][:-4]
        fn, s, _, _ = self.dataset.getitem_by_filename(fn)
        if t == "source":
            self.source_audio_file_name = fn
        else:
            self.target_audio_file_name = fn
        return fn, (16000, s)

    def sampel_source_audio_data(self):
        return self.sample_data("source")

    def sampel_target_audio_data(self):
        return self.sample_data("target")

    def generate_model_input(self):
        # 動態更新檔案列表
        self.dataset.update_audio_list()
        source_fn = self.source_audio_file_name
        target_fn = self.target_audio_file_name
        _, source_s, source_l, source_f = self.dataset.getitem_by_filename(
            source_fn)
        _, ref_s, ref_l, ref_f = self.dataset.getitem_by_filename(target_fn)
        if self.model_input_selection[0] == "source":
            s, l, f = source_s, source_l, source_f
        else:
            s, l, f = ref_s, ref_l, ref_f
        if self.model_input_selection[1] == "source":
            ref = source_s
        else:
            ref = ref_s
        return s, l, f, ref

    def generate_output(self):
        def get_midi(x): return int(
            x.split("_")[-1].split(".")[0].split("-")[1])
        s, l, f, ref = self.generate_model_input()
        source_midi = get_midi(self.source_audio_file_name)
        target_midi = get_midi(self.target_audio_file_name)
        midi_table = {
            "source": source_midi,
            "ref": target_midi,
            "custom": 60
        }
        semitone_shift = midi_table[self.model_input_selection[1]
                                    ] - midi_table[self.model_input_selection[0]]
        new_f = transform_frequency(f, semitone_shift)
        rec_s = self.model_gen(s, cal_loudness_norm(
            l), new_f, ref).squeeze().detach().numpy()
        return (16000, rec_s)

    def model_gen(self, s: ndarray, l_norm: ndarray, f: ndarray, timbre_s: ndarray):
        def transfrom(x_array): return torch.from_numpy(x_array).unsqueeze(0)
        s, l_norm, f, timbre_s = transfrom(s), transfrom(
            l_norm), transfrom(f), transfrom(timbre_s)
        f = f[:, :-1, 0]
        _, _, rec_s, _, _, _ = self.model(s, l_norm, f, timbre_s)
        return rec_s

    def change_model_input(self, source: str, ref: str):
        selection = [source, ref]
        for i, item in enumerate(selection):
            if item == None:
                selection[i] = "source"
        self.model_input_selection = selection
        return f"Source: {selection[0]}, Ref: {selection[1]}"

    def save_audio(self, audio_data, filename="generated_audio.wav", samplerate=16000):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{self.source_audio_file_name}_to_{self.target_audio_file_name}_{timestamp}.wav"
        save_path = os.path.join("generated_audio", save_filename)

        if not os.path.exists("generated_audio"):
            os.makedirs("generated_audio")

        sf.write(save_path, audio_data, samplerate)
        return save_path

    def update_audio_list(self):

        self.audio_list = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.base_path)
            for file in files if file.endswith('.npy')
        ]
        print(f"Audio list updated: {len(self.audio_list)} files found.")
    from PyQt6.QtCore import Qt

    def setup_ui(self):
        """建立原始 UI 介面"""

        # 創建主佈局
        main_layout = QVBoxLayout()  # 將主佈局改為 QVBoxLayout 來支援文字區放在上方

        # 創建顯示提示文字區域
        text_label = QLabel("Select the instrument to input", self)
        # text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 使用 AlignmentFlag 代替 AlignCenter
        text_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # 設定字體
        font = QFont("Arial", 16, QFont.Weight.Bold)  # 這裡可以設置字體名稱、大小和加粗等樣式
        text_label.setFont(font)  # 設置字體

        main_layout.addWidget(text_label)  # 把文字區域加入主佈局

        # 原本的主佈局內容
        content_layout = QHBoxLayout()

        # 左側按鈕佈局
        left_button_layout = QVBoxLayout()
        for instrument_name in self.paths.keys():
            instrument_layout = QHBoxLayout()
            image_label = QLabel(self)

            # 確保圖片路徑有效
            # image_path = self.image_paths.get(instrument_name, "")
            image_path = self.image_paths_1.get(instrument_name, "")
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                image_label.setPixmap(pixmap.scaled(
                    50, 50, Qt.AspectRatioMode.KeepAspectRatio))
            else:
                image_label.setText("圖片缺失")

            # 創建按鈕並連接到相應的處理函數
            btn = QPushButton(instrument_name, self)
            # 設定按鈕的寬度和高度
            btn.setFixedSize(150, 50)  # 設定固定寬度為 150，高度為 50

            # 設定按鈕文字的字體和大小

            # 設定按鈕文字的字體和大小
            font = QFont()  # 使用默認字體
            font.setPointSize(12)  # 設定字體大小為 14
            font.setBold(False)  # 設定字體為加粗
            btn.setFont(font)  # 設置按鈕字體
            # 防止 lambda 捕獲變數問題，顯式傳遞 instrument_name
            btn.clicked.connect(
                lambda _, button=btn, name=instrument_name: self.on_button_click(button, name))

            # 防止 lambda 捕獲變數問題，顯式傳遞 instrument_name
            # btn.clicked.connect(lambda _, name=instrument_name: self.load_files(name))

            instrument_layout.addWidget(image_label)
            instrument_layout.addWidget(btn)
            left_button_layout.addLayout(instrument_layout)

        content_layout.addLayout(left_button_layout)

        # list_widget 佈局
        self.list_widget.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection)
        self.list_widget.itemClicked.connect(self.on_item_clicked)

        # 設定字體為系統默認字體，但調整字體大小
        font = QFont()  # 使用系統的默認字體
        font.setPointSize(14)  # 調整字體大小
        self.list_widget.setFont(font)  # 設定 list_widget 的字體

        content_layout.addWidget(self.list_widget)
        self.list_widget.setSpacing(5)  # 設定 list_widget 的間距

        # 插入空白區域來調整右側按鈕間距
        spacer = QSpacerItem(40, 0, QSizePolicy.Policy.Fixed,
                             QSizePolicy.Policy.Expanding)
        content_layout.addItem(spacer)

        # 右側按鈕佈局
        right_button_layout = QVBoxLayout()
        for name, func in self.right_buttons.items():
            btn = QPushButton(name, self)

            # 設定按鈕的寬度和高度
            btn.setFixedSize(150, 50)  # 設定固定寬度為 150，高度為 50
            # 設定按鈕文字的字體和大小
            font = QFont()  # 使用默認字體
            font.setPointSize(12)  # 設定字體大小為 14
            font.setBold(False)  # 設定字體為加粗
            btn.setFont(font)  # 設置按鈕字體

            btn.clicked.connect(func)
            right_button_layout.addWidget(btn)

        if self.right_buttons:
            content_layout.addLayout(right_button_layout)

        main_layout.addLayout(content_layout)

        # 設定頁面和佈局
        self.page1.setLayout(main_layout)
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)
        self.stacked_widget.addWidget(self.page4)  # 加入第三頁

        layout = QVBoxLayout(self)
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

        self.setWindowTitle("樂器音色轉換 - 介面")
        self.setGeometry(100, 100, 1000, 600)

    def on_button_click(self, button, instrument_name):
        """處理按鈕點擊事件，先改變按鈕顏色，再載入列表"""
        # 1. 改變按鈕顏色
        # self.change_button_style(button)
        # 2. 強制 UI 立即更新，確保顏色變化不會延遲
        # QApplication.processEvents()
        # 3. 更新列表內容
        if self.stacked_widget.currentIndex() == 0:
            # 檢查上一個選中的按鈕
            if hasattr(self, 'last_selected_button') and self.last_selected_button != button:
                # 恢復上次選中按鈕的顏色
                self.last_selected_button.setStyleSheet(
                    "background-color: none; color: black;")

            # 設定當前點擊按鈕的顏色
            button.setStyleSheet("background-color: lightblue; color: black;")

            # 儲存當前選中的按鈕
            self.last_selected_button = button
        self.load_files(instrument_name)  # 呼叫載入資料的函數

    def reset_buttons_style(self):
        """重置所有按鈕顏色為預設狀態"""
        for btn in self.findChildren(QPushButton):
            # 恢復按鈕顏色為預設狀態
            btn.setStyleSheet("background-color: none; color: black;")

    def setup_page2_ui(self):
        """設定第二頁介面"""
        page2_layout = QVBoxLayout()

        # 創建顯示提示文字區域
        text_label = QLabel("Select the target instrument", self)
        text_label.setAlignment(Qt.AlignmentFlag.AlignLeft)  # 讓文字靠左對齊
        # 設定字體
        font = QFont("Arial", 16, QFont.Weight.Bold)  # 設定字體為 Arial、大小 16、加粗
        text_label.setFont(font)
        # 限制 QLabel 的高度，避免影響下面的按鈕佈局
        text_label.setFixedHeight(40)
        # 設定 QLabel 不會自動擴展
        text_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # 創建一個水平布局來將標題置於左側，並確保後面有空間
        title_layout = QHBoxLayout()
        title_layout.addWidget(text_label)
        title_layout.addStretch()  # 留下空間讓標題貼左
        # 將標題布局加入到主頁面的布局中
        page2_layout.addLayout(title_layout)

        button_layout = QHBoxLayout()
        button_names = ["Bass", "Brass", "Flute", "Guitar", "Piano", "Mallet"]
        image_paths = ["MI/bass.png", "MI/Brass.png", "MI/flute.png",
                       "MI/guitar.png", "MI/piano.png", "MI/Mallet.png"]  # 替換為實際的圖片路徑
        image_paths_1 = ["your_image/bass.png", "your_image/brass.png", "your_image/flute.png",
                         "your_image/guitar.png", "your_image/piano.png", "your_image/mallet.png"]  # 替換為實際的圖片路徑
        target_files = [
            "bass_acoustic_000-039-127", "brass_acoustic_059-048-075", "flute_acoustic_023-067-050", "guitar_acoustic_022-058-100",
            "piano20250109_051732_000-052-050", "mallet_acoustic_032-037-075"
        ]
        # 原本的六個按鈕
        button_layout = QHBoxLayout()
        for i in range(6):
            # 創建一個垂直布局來放置圖片和按鈕
            item_layout = QVBoxLayout()
            item_layout.setAlignment(
                Qt.AlignmentFlag.AlignCenter)  # 確保圖片和按鈕居中對齊

            # 設定圖片
            image_label = QLabel(self)
            # pixmap = QPixmap(image_paths[i])
            pixmap = QPixmap(image_paths_1[i])
            # 調整圖片大小
            pixmap = pixmap.scaled(
                80, 80, Qt.AspectRatioMode.KeepAspectRatio)  # 這裡調整圖片大小
            image_label.setPixmap(pixmap)
            image_label.setAlignment(
                Qt.AlignmentFlag.AlignCenter)  # 修改這行為 PyQt6 的寫法
            # 設定 QLabel 的最大尺寸
            image_label.setFixedSize(80, 80)  # 限制圖片區域的大小
            # 設定按鈕
            btn = QPushButton(button_names[i], self)
            btn.setContentsMargins(0, 0, 0, 0)  # 確保按鈕沒有外邊距
            # 設定按鈕的大小策略
            btn.setFixedWidth(100)  # 可根據需求設置按鈕寬度
            btn.setFixedHeight(50)  # 可根據需求設置按鈕高度
            btn.setStyleSheet("font-size: 16px;")  # 設定文字大小為 14px

            # 為每個按鈕設置對應的檔案
            btn.clicked.connect(
                lambda _, i=i: self.select_instrument_file(target_files[i]))
            # 將圖片和按鈕加到垂直佈局
            item_layout.addWidget(image_label)
            item_layout.addWidget(btn)
            # 將這個垂直佈局加入到水平佈局中
            button_layout.addLayout(item_layout)
        page2_layout.addLayout(button_layout)

        # 顯示生成成功的 QLabel，初始顯示 "尚未生成"
        self.generation_success_text = QLabel("Not Generated Yet", self)
        self.generation_success_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.generation_success_text.setFixedHeight(20)  # 設定固定高度
        self.generation_success_text.setContentsMargins(10, 0, 10, 0)  # 設置內邊距
        page2_layout.addWidget(self.generation_success_text)

        # 顯示路徑的 QLabel
        self.save_path_text = QLabel("Save Path: Not Generated Yet", self)
        self.save_path_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.save_path_text.setFixedHeight(20)  # 調整固定高度，減少空間
        self.save_path_text.setContentsMargins(0, 0, 0, 0)  # 設置內邊距
        page2_layout.addWidget(self.save_path_text)

        # 顯示時間的 QLabel
        self.time_text = QLabel("Time：", self)
        self.time_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_text.setFixedHeight(20)  # 調整固定高度，減少空間
        self.time_text.setContentsMargins(10, 0, 10, 0)  # 設置內邊距
        page2_layout.addWidget(self.time_text)

        # 新增 '開始生成'、'撥放生成' 和 '返回上一頁' 按鈕，並放在同一排
        navigation_button_layout = QHBoxLayout()
        navigation_button_layout.setSpacing(5)  # 調整按鈕之間的間距，減少空間

        # 開始生成按鈕
        start_generation_button = QPushButton("Start Generation", self)
        start_generation_button.setFixedWidth(120)  # 可根據需求調整按鈕大小
        start_generation_button.setFixedHeight(40)
        start_generation_button.clicked.connect(self.start_generation)
        navigation_button_layout.addWidget(start_generation_button)

        # 播放生成按鈕
        play_generation_button = QPushButton("Play Generated", self)
        play_generation_button.setFixedWidth(120)  # 可根據需求調整按鈕大小
        play_generation_button.setFixedHeight(40)
        play_generation_button.clicked.connect(self.play_generation)
        navigation_button_layout.addWidget(play_generation_button)

        # 返回上一頁按鈕
        back_button = QPushButton("back", self)
        back_button.setFixedWidth(120)  # 可根據需求調整按鈕大小
        back_button.setFixedHeight(40)
        back_button.clicked.connect(self.switch_back_page)
        navigation_button_layout.addWidget(back_button)

        # 將三個按鈕加入頁面
        page2_layout.addLayout(navigation_button_layout)

        # 設定第二頁的 layout
        self.page2.setLayout(page2_layout)

    def setup_page3_ui(self):
        """建立鋼琴介面，並保留底部空白放置返回按鈕"""
        layout = QVBoxLayout()

        # 新增標題 QLabel
        title_label = QLabel("Tap to start recording", self.page3)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)  # 靠左對齊
        title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold;")  # 設定字體大小與粗體
        layout.addWidget(title_label)  # 把標題加到版面配置

        # 設定頁面背景顏色為粉紅色
        self.page3.setStyleSheet("background-color: #D3D3D3;")  # 灰色背景

        self.sounds = {
            "do0": "piano_sounds/261.63Hz.wav",  # C4
            "re0": "piano_sounds/293.66Hz.wav",  # D4
            "mi0": "piano_sounds/329.63Hz.wav",  # E4
            "fa0": "piano_sounds/349.23Hz.wav",  # F4
            "sol0": "piano_sounds/392.00Hz.wav",  # G4
            "la0": "piano_sounds/440.00Hz.wav",  # A4
            "si0": "piano_sounds/493.88Hz.wav",  # B4
            "do1": "piano_sounds/523.25Hz.wav",  # C5
            "doS0": "piano_sounds/277.18Hz.wav",  # C#4
            "reS0": "piano_sounds/311.13Hz.wav",  # D#4
            "faS0": "piano_sounds/369.99Hz.wav",  # F#4
            "solS0": "piano_sounds/415.30Hz.wav",  # G#4
            "laS0": "piano_sounds/466.16Hz.wav",  # A#4
        }
        self.last_saved_path = None
        self.last_timestamp = None  # 儲存上一次音符被按下的時間

        # 設定初始的 x 和 y 坐標偏移量
        offset_x = 100
        offset_y = 20
        a = 28
        # 鋼琴白鍵
        self.do0 = QPushButton(self.page3)
        self.do0.setGeometry(QRect(50 + offset_x, 70 + offset_y, 88, 378))
        self.do0.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.do0.setText("")
        self.do0.setObjectName("do0")
        self.do0.clicked.connect(lambda: self.play_piano_sound("do0"))

        self.re0 = QPushButton(self.page3)
        self.re0.setGeometry(
            QRect(110 + offset_x + 28, 70 + offset_y, 88, 378))
        self.re0.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.re0.setText("")
        self.re0.setObjectName("re0")
        self.re0.clicked.connect(lambda: self.play_piano_sound("re0"))

        self.mi0 = QPushButton(self.page3)
        self.mi0.setGeometry(
            QRect(170 + offset_x + 56, 70 + offset_y, 88, 378))
        self.mi0.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.mi0.setText("")
        self.mi0.setObjectName("mi0")
        self.mi0.clicked.connect(lambda: self.play_piano_sound("mi0"))

        self.fa0 = QPushButton(self.page3)
        self.fa0.setGeometry(
            QRect(230 + offset_x + 84, 70 + offset_y, 88, 378))
        self.fa0.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.fa0.setText("")
        self.fa0.setObjectName("fa0")
        self.fa0.clicked.connect(lambda: self.play_piano_sound("fa0"))

        self.sol0 = QPushButton(self.page3)
        self.sol0.setGeometry(
            QRect(290 + offset_x + 112, 70 + offset_y, 88, 378))
        self.sol0.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.sol0.setText("")
        self.sol0.setObjectName("sol0")
        self.sol0.clicked.connect(lambda: self.play_piano_sound("sol0"))

        self.la0 = QPushButton(self.page3)
        self.la0.setGeometry(
            QRect(350 + offset_x + 140, 70 + offset_y, 88, 378))
        self.la0.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.la0.setText("")
        self.la0.setObjectName("la0")
        self.la0.clicked.connect(lambda: self.play_piano_sound("la0"))

        self.si0 = QPushButton(self.page3)
        self.si0.setGeometry(
            QRect(410 + offset_x + 168, 70 + offset_y, 88, 378))
        self.si0.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.si0.setText("")
        self.si0.setObjectName("si0")
        self.si0.clicked.connect(lambda: self.play_piano_sound("si0"))

        self.do1 = QPushButton(self.page3)
        self.do1.setGeometry(
            QRect(470 + offset_x + 196, 70 + offset_y, 88, 378))
        self.do1.setStyleSheet("background-color: #FFFFFF")  # 白色白鍵
        self.do1.setText("")
        self.do1.setObjectName("do1")
        self.do1.clicked.connect(lambda: self.play_piano_sound("do1"))

        # 鋼琴黑鍵
        self.doS0 = QPushButton(self.page3)
        self.doS0.setGeometry(QRect(110 + offset_x, 70 + offset_y, 58, 244))
        self.doS0.setStyleSheet("background-color: #000000")  # 黑色黑鍵
        self.doS0.setText("")
        self.doS0.clicked.connect(lambda: self.play_piano_sound("doS0"))

        self.reS0 = QPushButton(self.page3)
        self.reS0.setGeometry(QRect(195 + offset_x, 70 + offset_y, 58, 244))
        self.reS0.setStyleSheet("background-color: #000000")  # 黑色黑鍵
        self.reS0.setText("")
        self.reS0.clicked.connect(lambda: self.play_piano_sound("reS0"))

        self.faS0 = QPushButton(self.page3)
        self.faS0.setGeometry(QRect(375 + offset_x, 70 + offset_y, 58, 244))
        self.faS0.setStyleSheet("background-color: #000000")  # 黑色黑鍵
        self.faS0.setText("")
        self.faS0.clicked.connect(lambda: self.play_piano_sound("faS0"))

        self.solS0 = QPushButton(self.page3)
        self.solS0.setGeometry(QRect(460 + offset_x, 70 + offset_y, 58, 244))
        self.solS0.setStyleSheet("background-color: #000000")  # 黑色黑鍵
        self.solS0.setText("")
        self.solS0.clicked.connect(lambda: self.play_piano_sound("solS0"))

        self.laS0 = QPushButton(self.page3)
        self.laS0.setGeometry(QRect(550 + offset_x, 70 + offset_y, 58, 244))
        self.laS0.setStyleSheet("background-color: #000000")  # 黑色黑鍵
        self.laS0.setText("")
        self.laS0.clicked.connect(lambda: self.play_piano_sound("laS0"))

        # 設定清除按鈕
        clear_button = QPushButton("Clear", self.page3)
        clear_button.setStyleSheet(
            "font-size: 18px; padding: 10px 20px;")  # 設定字型大小與內距
        clear_button.clicked.connect(self.clear_action)

        # 設定儲存按鈕
        save_button = QPushButton("Save", self.page3)
        save_button.setStyleSheet(
            "font-size: 18px; padding: 10px 20px;")  # 設定字型大小與內距
        save_button.clicked.connect(self.save_action)

        # 設定撥放按鈕
        play_piano_button = QPushButton("Play_piano", self.page3)
        play_piano_button.setStyleSheet(
            "font-size: 18px; padding: 10px 20px;")  # 設定字型大小與內距
        play_piano_button.clicked.connect(self.play_save_piano_music)

        # 鋼琴彈奏處理按鈕
        piano_audio_button = QPushButton("Piano_prosses", self.page3)
        piano_audio_button.setStyleSheet(
            "font-size: 18px; padding: 10px 20px;")  # 設定字型大小與內距
        piano_audio_button.clicked.connect(self.process_audio_piano)

        # 設定返回按鈕
        back_button = QPushButton("Back", self.page3)
        back_button.setStyleSheet(
            "font-size: 18px; padding: 10px 20px;")  # 設定字型大小與內距
        back_button.clicked.connect(self.switch_back_page)

        # 設定next按鈕
        next_button = QPushButton("Next", self.page3)
        next_button.setStyleSheet(
            "font-size: 18px; padding: 10px 20px;")  # 設定字型大小與內距
        next_button.clicked.connect(self.switch_page)

        # 創建水平布局來放置按鈕
        button_layout = QHBoxLayout()
        button_layout.addWidget(clear_button)  # 清除按鈕
        button_layout.addWidget(save_button)  # 儲存按鈕
        button_layout.addWidget(play_piano_button)  # 播放按鈕
        button_layout.addWidget(piano_audio_button)  # 彈奏處理按鈕
        button_layout.addWidget(back_button)  # 返回按鈕
        button_layout.addWidget(next_button)  # next按鈕

        # 在垂直布局中添加空間和水平按鈕布局
        layout.addStretch(1)  # 空間填充，使其他按鈕保持在底部
        layout.addLayout(button_layout)  # 把水平按鈕布局放到底部

        # 設定頁面佈局
        self.page3.setLayout(layout)

    # def play_piano_sound(self, key):
        # """播放對應的鋼琴鍵音檔"""
        # sound_path = self.sounds.get(key, None)

        # if sound_path and os.path.exists(sound_path):
        # pygame.mixer.init()
        # pygame.mixer.music.load(sound_path)
        # pygame.mixer.music.play()
        # else:
        # print(f"[DEBUG] 音檔未找到: {sound_path}")
    def clear_action(self):

        self.audio_sequence = AudioSegment.empty()

    def play_piano_sound(self, key):
        # """播放對應的鋼琴鍵音檔並記錄音符，合成新的音檔"""
        sound_path = self.sounds.get(key, None)

        if sound_path and os.path.exists(sound_path):
            # 播放音檔
            pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()

            # 記錄按下的音符
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 取得當前時間
            print(f"[DEBUG] 按下音符：{key}，時間：{timestamp}")

            # 初始化音頻合成序列
            if not hasattr(self, 'audio_sequence'):
                self.audio_sequence = AudioSegment.empty()

            # 把音符音檔加入到合成音檔中
            sound = AudioSegment.from_wav(sound_path)
            self.audio_sequence += sound  # 合成音檔

        else:
            print(f"[DEBUG] 音檔未找到: {sound_path}")

    def save_action(self):
        """儲存合成的音檔並使用當前時間戳確保檔案名稱唯一"""
        if hasattr(self, 'audio_sequence') and len(self.audio_sequence) > 0:
            # 指定儲存的資料夾和檔案名稱
            folder_path = 'nsynth-subset5/Piano/wav'  # 這是您可以修改的資料夾名稱
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.last_saved_path = os.path.join(
                folder_path, f"recorded_piano_song_{timestamp}_050-XXX-050.wav")

            # 確保資料夾存在
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 儲存音檔
            self.audio_sequence.export(self.last_saved_path, format="wav")
            print(f"[DEBUG] 合成音檔已儲存至: {self.last_saved_path}")
        else:
            print("[DEBUG] 沒有錄製音符，無法儲存音檔")

    def play_save_piano_music(self):
        """播放儲存的音檔"""
        if self.last_saved_path and os.path.exists(self.last_saved_path):
            pygame.mixer.init()
            pygame.mixer.music.load(self.last_saved_path)
            pygame.mixer.music.play()
            print(f"正在播放錄音：{self.last_saved_path}")

            # 播放結束後釋放資源
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # 等待音樂播放結束
            pygame.mixer.quit()  # 播放結束後釋放資源
        else:
            print("沒有錄音文件")

    def process_audio_piano(self):
        # 開始計時
        start_time = time.time()

        # 儲存錄音並獲取檔案路徑
        audio_file_path = self.last_saved_path
        # 設定輸入 .wav 檔案路徑
        wav_file = audio_file_path

        # 設定輸出資料夾
        output_dir = "nsynth-subset5/Piano"  # 請更換為你想儲存特徵的資料夾名稱
        os.makedirs(output_dir, exist_ok=True)

        # 建立特徵提取器
        extractor = FeatureExtractor(wav_file, output_dir)

        # 執行特徵提取
        extractor.extract_features()

        # 保存音訊信號為 .npy 檔案
        extractor.save_signal_as_npy()

        base_dir = "nsynth-subset5/Piano"
        rename_files_1(base_dir, midi_value)

        # 計算並顯示處理時間
        elapsed_time = time.time() - start_time
        print(f"音頻處理已完成！總共花費時間: {elapsed_time:.2f} 秒")

        self.audio_sequence = AudioSegment.empty()  # 清空音檔序列

        self.source_audio_file_name = rename_files_1(
            base_dir, midi_value)  # 取得音檔名稱

        # 顯示完成訊息
        QMessageBox.information(None, "完成", "音頻轉換為 .npy 格式已完成！")

    def setup_page4_ui(self):
        """第四頁面：錄音功能 UI"""
        layout = QVBoxLayout()

        # 新增標題 QLabel，並設定居中對齊
        title_label = QLabel("Record your voice", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 中心對齊
        title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold;")  # 設定字體大小與粗體
        layout.addWidget(title_label)  # 把標題加到版面配置

        # 插入空白區域來調整上下間距
        spacer = QSpacerItem(
            20, 100, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addItem(spacer)

        # 顯示錄音時間進度條
        self.recording_progress = QProgressBar(self)
        self.recording_progress.setRange(0, 30)  # 設定範圍為 0 到 30（秒）
        self.recording_progress.setValue(0)  # 初始為 0
        self.recording_progress.setFormat("%v s")  # 讓進度條顯示秒數
        self.recording_progress.setTextVisible(True)  # 顯示文字
        self.recording_progress.setStyleSheet(
            "height: 20px; background-color: #f0f0f0;")
        layout.addWidget(self.recording_progress,
                         alignment=Qt.AlignmentFlag.AlignCenter)

        # 設定水平排列的按鈕佈局
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 設定按鈕區域居中

        # 設定開始/暫停錄音按鈕
        self.record_button = QPushButton("Start Recording", self)
        self.record_button.setStyleSheet("font-size: 14px;")  # 設定按鈕文字大小
        self.record_button.setFixedSize(120, 40)  # 設定按鈕大小
        self.record_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_button)

        # 設定儲存錄音按鈕
        self.save_button = QPushButton("Save Recording", self)
        self.save_button.setStyleSheet("font-size: 14px;")  # 設定按鈕文字大小
        self.save_button.setFixedSize(120, 40)  # 設定按鈕大小
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_recording)
        button_layout.addWidget(self.save_button)

        # 設定播放錄音按鈕
        self.play_button = QPushButton("Play Recording", self)
        self.play_button.setStyleSheet("font-size: 14px;")  # 設定按鈕文字大小
        self.play_button.setFixedSize(120, 40)  # 設定按鈕大小
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_recording)
        button_layout.addWidget(self.play_button)

        # 設定音訊處理按鈕
        self.audio_process_button = QPushButton("Process Audio", self)
        self.audio_process_button.setStyleSheet("font-size: 14px;")  # 設定按鈕文字大小
        self.audio_process_button.setFixedSize(120, 40)  # 設定按鈕大小
        self.audio_process_button.setEnabled(True)
        # 音訊處理按鈕的點擊事件
        # self.audio_process_button.clicked.connect(self.on_audio_process_click)  # 連結到處理函式
        self.audio_process_button.clicked.connect(
            self.process_audio)  # 連結到處理函式
        button_layout.addWidget(self.audio_process_button)

        # 設定返回按鈕
        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("font-size: 14px;")  # 設定按鈕文字大小
        self.back_button.setFixedSize(120, 40)  # 設定按鈕大小
        self.back_button.setEnabled(True)
        self.back_button.clicked.connect(self.switch_back_page)
        button_layout.addWidget(self.back_button)

        # 設定next按鈕
        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("font-size: 14px;")  # 設定按鈕文字大小
        self.next_button.setFixedSize(120, 40)  # 設定按鈕大小
        self.next_button.setEnabled(True)
        self.next_button.clicked.connect(self.switch_page)
        button_layout.addWidget(self.next_button)

        # 把按鈕佈局加入到主布局中
        layout.addLayout(button_layout)

        # 設定頁面佈局
        self.page4.setLayout(layout)

        # 插入空白區域來調整上下間距
        spacer = QSpacerItem(
            20, 100, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addItem(spacer)

        # 初始化錄音器
        self.recorder = AudioRecorder()

        # 錄音狀態初始化
        self.is_recording = False  # 初始狀態為不錄音

        # 定時器來更新進度條
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.elapsed_time = 0
        self.max_time = 50000

    def toggle_recording(self):
        """開始或停止錄音"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def toggle_recording(self):
        """開始或停止錄音"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """開始錄音"""
        self.is_recording = True
        self.elapsed_time = 0
        self.recording_progress.setValue(0)
        self.record_button.setText("Pause Recording")

        # **開始錄音**
        self.recorder.start_recording()

        # 啟動計時器，每秒觸發一次
        self.timer.start(10)
        print("錄音開始")
        # 立即開始錄製音訊塊
        self.recorder.record_chunk()

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        self.timer.stop()  # 停止計時器
        self.recorder.stop_recording()  # **停止錄音器**
        self.record_button.setText("Start Recording")
        self.save_button.setEnabled(True)
        self.play_button.setEnabled(True)
        print("錄音結束")

    def update_progress(self):
        """更新錄音時間條並錄製音訊塊"""
        if self.is_recording:
            # 每次更新進度時，elapsed_time 增加 0.01 秒
            self.elapsed_time += 0.0665  # 每 10 毫秒增加 0.01 秒

            # 更新進度條顯示為以秒為單位
            self.recording_progress.setValue(
                int(self.elapsed_time))  # 進度條顯示以秒為單位
            self.recording_progress.setFormat(
                f"{int(self.elapsed_time)} s / {self.max_time} s")

            # **錄製當前的音訊區塊**
            self.recorder.record_chunk()  # 每次更新進度時調用錄音類別的 record_chunk()

            # 超過最大時間時自動停止錄音
            if self.elapsed_time >= self.max_time:
                self.stop_recording()

    def save_recording(self):
        """儲存錄音檔案"""
        if self.is_recording:
            # 如果正在錄音，先停止錄音再保存
            self.stop_recording()
        self.recorder.save_audio()
        print(f"錄音已儲存到 {self.recorder.output_filename}")
        # 儲存後清空 frames 資料
        self.recorder.frames = []

        return self.recorder.output_filename  # 返回儲存的檔案路徑  不需要可註解掉

    def play_recording(self):
        """播放錄音"""
        file_path = self.recorder.output_filename

        if os.path.exists(file_path):
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            print(f"正在播放錄音：{file_path}")

            # 播放結束後釋放資源
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # 等待音樂播放結束
            pygame.mixer.quit()  # 播放結束後釋放資源

        else:
            print("沒有錄音文件")

    def on_audio_process_click(self):
        try:
            # 儲存錄音並獲取檔案路徑
            audio_file_path = self.save_recording()

            # 打印調試訊息來檢查返回的路徑
            print(f"儲存的錄音檔案路徑: {audio_file_path}")

            if audio_file_path:  # 檢查是否成功儲存
                # 如果返回了有效的檔案路徑，則進行音訊處理
                print("開始音訊處理...")
                self.process_audio(audio_file_path)
            else:
                # 如果檔案路徑無效，提示錯誤
                print("錄音檔案儲存失敗，無效的檔案路徑！")
                QMessageBox.warning(None, "錯誤", "錄音檔案儲存失敗，請重試！")
        except Exception as e:
            # 捕捉並顯示異常錯誤
            print(f"錯誤: {e}")
            QMessageBox.warning(None, "錯誤", f"處理過程中發生錯誤: {e}")

    def process_audio(self):
        # 開始計時
        start_time = time.time()

        # 儲存錄音並獲取檔案路徑
        audio_file_path = self.recorder.output_filename
        # 設定輸入 .wav 檔案路徑
        wav_file = audio_file_path

        # 設定輸出資料夾
        output_dir = "nsynth-subset5/audio"  # 請更換為你想儲存特徵的資料夾名稱
        os.makedirs(output_dir, exist_ok=True)

        # 建立特徵提取器
        extractor = FeatureExtractor(wav_file, output_dir)

        # 執行特徵提取
        extractor.extract_features()

        # 保存音訊信號為 .npy 檔案
        extractor.save_signal_as_npy()

        base_dir = "nsynth-subset5/audio"
        rename_files(base_dir, midi_value)

        self.target_audio_file_name = rename_files(
            base_dir, midi_value)  # 取得音檔名稱

        # 計算並顯示處理時間
        elapsed_time = time.time() - start_time
        print(f"音頻處理已完成！總共花費時間: {elapsed_time:.2f} 秒")

        self.source_audio_file_name = rename_files(
            base_dir, midi_value)  # 取得音檔名稱

        # 顯示完成訊息
        QMessageBox.information(None, "完成", "音頻轉換為 .npy 格式已完成！")

    def select_instrument_file(self, file_name):
        """根據選擇的檔案名稱執行相應操作，並將檔案名稱儲存在 target_audio_file_name"""
        self.target_audio_file_name = file_name  # 將選中的檔案名稱賦值給 self.target_audio_file_name

        QTimer.singleShot(
            3000, lambda: self.generation_success_text.setText("Not Generated Yet"))

        audio_folder = "./nsynth-subset2/test/signal"
        file_path = os.path.join(
            audio_folder, self.target_audio_file_name + ".wav")
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "錯誤", "音檔不存在！")
            return
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        # QMessageBox.information(self, "選擇的樂器檔案", f"您選擇的樂器檔案是：{self.target_audio_file_name}")

    def play_audio(self, file_name):
        # 自定義資料夾路徑
        custom_folder_path = "./nsynth-subset2/test/signal"
        file_path = os.path.join(custom_folder_path, file_name + ".wav")
        # 初始化 pygame
        # pygame.mixer.init()
        # 加載音頻文件
        pygame.mixer.music.load(file_path)
        # 播放音頻
        pygame.mixer.music.play()

    def piano_input(self):
        """開始生成處理邏輯"""
        print("[DEBUG] 鋼琴輸入")
        QMessageBox.information(self, "功能", "鋼琴輸入")

    def start_generation(self):
        """開始生成處理邏輯"""
        global saved_path
        # 更新參數
        self.change_model_input("source", "ref")
        # 開始計時
        start_time = time.time()
        # 生成音頻
        rec_audio = self.generate_output()
        saved_path = self.save_audio(rec_audio[1])
        # 結束計時
        end_time = time.time()
        elapsed_time = end_time - start_time

        self.generation_success_text.setText("Generation Successful")
        self.generation_success_text.setVisible(True)
        # QTimer.singleShot(3000, lambda: self.generation_success_text.setText("尚未生成"))
        # 更新顯示的路徑
        self.save_path_text.setText(saved_path)  # 使用 setText 更新顯示的路徑
        # 更新顯示的生成時間
        # 使用 setText 更新顯示的時間
        self.time_text.setText(f"Time Taken: {elapsed_time:.2f} seconds")

        print("[DEBUG] 開始生成")
        # 彈出提示框
        # QMessageBox.information(self, "功能", "開始生成")

    def play_generation(self):
        """撥放生成處理邏輯"""
        audio_folder = "generated_audio"  # 音檔資料夾
        # 取得 QLabel 顯示的檔案名稱（無副檔名）
        file_name = self.save_path_text.text()  # 假設這是顯示的檔案名稱
        if not file_name:  # 檢查檔案名稱是否有效
            QMessageBox.warning(self, "錯誤", "檔案名稱無效！")
            return
        # 使用 os.path.basename 取得檔案名稱，並去除副檔名
        file_name_without_ext = os.path.basename(file_name)
        file_name_without_ext, _ = os.path.splitext(file_name_without_ext)
        # 組合檔案路徑
        file_path = os.path.join(audio_folder, file_name_without_ext + ".wav")
        # 播放音檔
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print("[DEBUG] 播放生成音檔")
        # QMessageBox.information(self, "訊息", "已播放生成音檔")

    def switch_recording_page(self):
        """切換到錄音介面（第四頁）"""
        self.stacked_widget.setCurrentIndex(3)

    def switch_piano_page(self):
        """切換到鋼琴介面（第三頁）"""
        self.stacked_widget.setCurrentIndex(2)

    def switch_page(self):
        """切換到第二頁"""
        self.stacked_widget.setCurrentIndex(1)

    def switch_back_page(self):
        """返回到第一頁"""
        self.stacked_widget.setCurrentIndex(0)

    def load_files(self, instrument_name):
        """載入選定樂器的檔案"""
        if instrument_name in self.paths:
            path = self.paths[instrument_name]
            if os.path.exists(path):
                # 清空舊的列表項目
                self.list_widget.clear()

                # 獲取資料夾中的檔案
                files = os.listdir(path)
                if files:
                    # 過濾檔案，只顯示 .wav 檔案，若有其他檔案類型需求，可以自行調整條件
                    # wav_files = [f for f in files if f.endswith(('.wav', '.npy'))]
                    wav_files = [f for f in files if f.endswith('.npy')]
                    if wav_files:
                        # 若有符合條件的檔案，將它們加入列表中
                        self.list_widget.addItems(wav_files)
                    else:
                        self.list_widget.addItem("資料夾內沒有 .wav 檔案")
                else:
                    # 若資料夾為空，顯示提示
                    self.list_widget.addItem("（此資料夾為空）")
            else:
                # 如果資料夾不存在，顯示錯誤提示
                QMessageBox.warning(self, "錯誤", f"找不到路徑: {path}")
        else:
            # 如果按鈕對應的樂器名稱沒有設定路徑，顯示錯誤提示
            QMessageBox.warning(self, "錯誤", "此按鈕未設定路徑")
        # 強制更新顯示
        # 強制更新顯示
        self.list_widget.update()  # 更新 widget 顯示

    def on_item_clicked(self, item):
        """當點擊選單中的某一項時，將去除副檔名的檔案名稱存入 self.source_audio_file_name 並在終端機輸出"""
        file_name = item.text()
        global file_name_without_ext
        if file_name and "（此資料夾為空）" not in file_name:
            file_name_without_ext, _ = os.path.splitext(file_name)
            self.source_audio_file_name = file_name_without_ext
            # 終端機輸出
            print(f"[DEBUG] 已選擇的音檔名稱（無副檔名）: {self.source_audio_file_name}")
        else:
            print("[DEBUG] 無效的檔案選擇")  # 終端機輸出錯誤資訊
            QMessageBox.warning(self, "錯誤", "請選擇有效的音檔")

    def play_music(self):
        """播放音樂"""
        audio_folder = "./nsynth-subset2/test/signal"
        if not self.source_audio_file_name:
            QMessageBox.warning(self, "錯誤", "請先選擇音檔！")
            return
        file_path = os.path.join(
            audio_folder, self.source_audio_file_name + ".wav")
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "錯誤", "音檔不存在！")
            return
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def pause_music(self):
        """停止撥放"""
        QMessageBox.information(self, "功能", "音樂已停止")


class FeatureExtractor:
    def __init__(self, wav_file_path: str, output_dir: str, sr: int = 16000, n_fft: int = 1024):
        self.wav_file_path = wav_file_path
        self.output_dir = output_dir
        self.sr = sr  # 目標取樣率 16kHz
        self.n_fft = n_fft
        self.hop_length = n_fft // 4

        # 讀取音訊
        self.signal = self.load_signal()
        self.midi_value = None

        # 建立特徵保存資料夾
        self.mfcc_dir = os.path.join(output_dir, "mfcc")
        self.frequency_dir = os.path.join(output_dir, "frequency")
        self.frequency_c_dir = os.path.join(output_dir, "frequency_c")
        self.loudness_dir = os.path.join(output_dir, "loudness")
        self.loudness_old_dir = os.path.join(
            output_dir, "loudness_old")  # 新增 loudness_old 資料夾
        self.signal_dir = os.path.join(output_dir, "signal")  # 新增 signal 資料夾
        os.makedirs(self.mfcc_dir, exist_ok=True)
        os.makedirs(self.frequency_dir, exist_ok=True)
        os.makedirs(self.frequency_c_dir, exist_ok=True)
        os.makedirs(self.loudness_dir, exist_ok=True)
        # 確保 loudness_old 資料夾存在
        os.makedirs(self.loudness_old_dir, exist_ok=True)
        os.makedirs(self.signal_dir, exist_ok=True)  # 確保 signal 資料夾存在

    def load_signal(self):
        """載入 .wav 音訊，確保為 16kHz 單聲道"""
        signal, sample_rate = torchaudio.load(self.wav_file_path)

        # 轉為單聲道（如果是立體聲）
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # 重新取樣到 16kHz（如果必要）
        if sample_rate != self.sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sr)
            signal = resampler(signal)

        return signal

    def save_signal_as_npy(self):
        """將音訊信號儲存為 .npy 檔案"""
        # 保存音訊信號到 signal 資料夾
        signal_output_path = os.path.join(
            self.output_dir, "signal", f"{os.path.splitext(os.path.basename(self.wav_file_path))[0]}.npy")

        # 確保 signal 資料夾存在
        os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)

        # 儲存信號
        np.save(signal_output_path, self.signal.squeeze(
            0).cpu().numpy())  # 移除額外維度並保存為 .npy 檔案
        print(f"音訊信號已保存至 {signal_output_path}")

    def extract_features(self):
        """提取所有特徵並保存"""
        # 提取 MFCC 特徵
        extract_mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=80,
            melkwargs=dict(n_fft=self.n_fft, hop_length=self.hop_length,
                           n_mels=128, f_min=20.0, f_max=8000.0)
        )
        mfcc = extract_mfcc(self.signal)
        mfcc_output_path = os.path.join(
            self.mfcc_dir, f"{os.path.splitext(os.path.basename(self.wav_file_path))[0]}.npy")
        np.save(mfcc_output_path, mfcc.squeeze(0).cpu().numpy())
        print(f"MFCC 特徵已保存至 {mfcc_output_path}")

        # 提取頻率特徵並計算 MIDI 音符
        device, cr, m_sec = get_extract_pitch_needs(device=torch.device("cpu"))
        frequency_c = extract_pitch(
            signal=self.signal,
            device=device,
            cr=cr,
            m_sec=m_sec,
            sampling_rate=self.sr,
            with_confidence=True,
        )

        if torch.any(torch.isnan(frequency_c)):
            print("錯誤：頻率特徵中包含 NaN，無法保存或處理。")
            return

        frequency_output_path = os.path.join(
            self.frequency_dir, f"{os.path.splitext(os.path.basename(self.wav_file_path))[0]}.npy")
        np.save(frequency_output_path, frequency_c.squeeze(0).cpu().numpy())
        print(f"頻率特徵已保存至 {frequency_output_path}")

        frequency_c_output_path = os.path.join(
            self.frequency_c_dir, f"{os.path.splitext(os.path.basename(self.wav_file_path))[0]}.npy")
        np.save(frequency_c_output_path, frequency_c.squeeze(0).cpu().numpy())
        print(f"頻率特徵已保存至 {frequency_c_output_path}")

        global midi_value
        freq_to_midi = FrequencyToMIDI(frequency_output_path)
        midi_value = freq_to_midi.calculate_midi()
        if midi_value is not None:
            print(f"計算出的 MIDI 音符是: {midi_value}")

        # 提取響度特徵
        a_weighting = get_A_weight().to(torch.device("cpu"))
        loudness = extract_loudness(self.signal, a_weighting)

        if torch.any(torch.isnan(loudness)):
            print("錯誤：響度特徵中包含 NaN，無法保存或處理。")
            return

        loudness_output_path = os.path.join(
            self.loudness_dir, f"{os.path.splitext(os.path.basename(self.wav_file_path))[0]}.npy")
        np.save(loudness_output_path, loudness.squeeze(0).cpu().numpy())
        print(f"響度特徵已保存至 {loudness_output_path}")

        loudness_old_output_path = os.path.join(
            self.output_dir, "loudness_old", f"{os.path.splitext(os.path.basename(self.wav_file_path))[0]}.npy")
        np.save(loudness_old_output_path, loudness.squeeze(0).cpu().numpy())
        print(f"響度特徵已保存至 {loudness_old_output_path}")


class FrequencyToMIDI:
    def __init__(self, npy_file_path):
        """初始化類別，載入頻率數據"""
        self.npy_file_path = npy_file_path
        self.frequency_data = self._load_frequency_data()

    def _load_frequency_data(self):
        """安全載入頻率數據"""
        try:
            frequency_data = np.load(self.npy_file_path)
            if len(frequency_data) == 0:
                print("警告：頻率數據為空")
            return frequency_data
        except Exception as e:
            print(f"錯誤：無法載入頻率數據 - {e}")
            return np.array([])

    def calculate_midi(self):
        """計算並顯示 MIDI 音符"""
        if len(self.frequency_data) == 0:
            print("頻率數據為空")
            return None

        # 計算主頻率（取平均值）
        main_frequency = np.mean(self.frequency_data)

        # 檢查是否為 NaN
        if np.isnan(main_frequency):
            print("錯誤：計算出的主頻率為 NaN")
            return None

        midi_value = int(round(librosa.hz_to_midi(main_frequency)))
        print(f"計算出的主頻率: {main_frequency:.2f} Hz")
        print(f"對應的 MIDI 音符: {midi_value}")
        return midi_value

    def calculate_midi_threaded(self):
        """線程安全的 MIDI 計算方法"""
        # 使用線程執行計算，避免阻塞主線程
        thread = threading.Thread(target=self.calculate_midi)
        thread.start()
        thread.join()  # 等待線程結束


def rename_files(base_dir, midi_value):
    """
    遍歷資料夾內的所有檔案，並根據給定的 MIDI 值進行檔案改名。
    :param base_dir: 要遍歷的資料夾路徑
    :param midi_value: 用來替換檔案名稱中的 XXX 部分的 MIDI 值
    """
    print("已執行")
    global new_file_name, new_file_name_no_ext
    # 遍歷資料夾內的所有子資料夾
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)

        # 確保是資料夾
        if os.path.isdir(subfolder_path):
            # 遍歷該子資料夾中的所有檔案
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.npy') or file_name.endswith('.wav'):
                    old_file_path = os.path.join(subfolder_path, file_name)

                    # 檢查檔案名稱格式，假設檔案名稱為 'piano20250108_055613_000-XXX-050.npy'
                    if '-' in file_name:
                        parts = file_name.split('-')

                        # 如果檔案名稱正確分為三個部分
                        if len(parts) == 3 and parts[1] == "XXX":

                            file_ext = os.path.splitext(
                                file_name)[1]  # 取得原始副檔名 (.npy 或 .wav)
                            new_file_name = f"{parts[0]}-{str(midi_value).zfill(3)}-050{file_ext}"
                            # 根據 MIDI 值進行改名，XXX 部分會被替換成 midi_value
                            # new_file_name = f"{parts[0]}-{str(midi_value).zfill(3)}-050.npy"

                            # 定義新檔案的路徑
                            new_file_path = os.path.join(
                                subfolder_path, new_file_name)
                            new_file_name_no_ext = os.path.splitext(new_file_name)[
                                0]  # 取得無副檔名的名稱
                            # 進行檔案改名
                            os.rename(old_file_path, new_file_path)
                            print(
                                f'Renamed: {old_file_path} -> {new_file_path}')

                            # 設定檔案路徑與目標資料夾
                            new_file_path = new_file_path  # 你的檔案路徑
                            target_dir = "nsynth-subset2/test/signal"  # 目標資料夾
                            # 確保目標資料夾存在
                            if not os.path.exists(target_dir):
                                os.makedirs(target_dir)
                            # 複製檔案
                            final_destination = os.path.join(
                                target_dir, os.path.basename(new_file_path))
                            shutil.copy(new_file_path, final_destination)
                            print(
                                f'已複製: {new_file_path} -> {final_destination}')

                            source_dir_1 = "nsynth-subset5/audio"  # 原始資料夾
                            target_dir_1 = "nsynth-subset4/test"  # 目標資料夾

                            for root, dirs, files in os.walk(source_dir_1):
                                # 設定目標資料夾路徑
                                target_root = root.replace(
                                    source_dir_1, target_dir_1)

                                # 確保目標資料夾存在
                                if not os.path.exists(target_root):
                                    os.makedirs(target_root)

                                # 複製檔案
                                for file_name in files:
                                    if file_name.endswith('.npy') or file_name.endswith('.wav'):
                                        source_file_path = os.path.join(
                                            root, file_name)
                                        target_file_path = os.path.join(
                                            target_root, file_name)
                                        shutil.copy(
                                            source_file_path, target_file_path)
                                        print(
                                            f'已複製: {source_file_path} -> {target_file_path}')
    return new_file_name_no_ext


def rename_files_1(base_dir, midi_value):
    """
    遍歷資料夾內的所有檔案，並根據給定的 MIDI 值進行檔案改名。
    :param base_dir: 要遍歷的資料夾路徑
    :param midi_value: 用來替換檔案名稱中的 XXX 部分的 MIDI 值
    """
    print("已執行")
    global new_file_name, new_file_name_no_ext
    # 遍歷資料夾內的所有子資料夾
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)

        # 確保是資料夾
        if os.path.isdir(subfolder_path):
            # 遍歷該子資料夾中的所有檔案
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.npy') or file_name.endswith('.wav'):
                    old_file_path = os.path.join(subfolder_path, file_name)

                    # 檢查檔案名稱格式，假設檔案名稱為 'piano20250108_055613_000-XXX-050.npy'
                    if '-' in file_name:
                        parts = file_name.split('-')

                        # 如果檔案名稱正確分為三個部分
                        if len(parts) == 3 and parts[1] == "XXX":

                            file_ext = os.path.splitext(
                                file_name)[1]  # 取得原始副檔名 (.npy 或 .wav)
                            new_file_name = f"{parts[0]}-{str(midi_value).zfill(3)}-050{file_ext}"
                            # 根據 MIDI 值進行改名，XXX 部分會被替換成 midi_value
                            # new_file_name = f"{parts[0]}-{str(midi_value).zfill(3)}-050.npy"

                            # 定義新檔案的路徑
                            new_file_path = os.path.join(
                                subfolder_path, new_file_name)
                            new_file_name_no_ext = os.path.splitext(new_file_name)[
                                0]  # 取得無副檔名的名稱
                            # 進行檔案改名
                            os.rename(old_file_path, new_file_path)
                            print(
                                f'Renamed: {old_file_path} -> {new_file_path}')

                            # 設定檔案路徑與目標資料夾
                            new_file_path = new_file_path  # 你的檔案路徑
                            target_dir = "nsynth-subset2/test/signal"  # 目標資料夾
                            # 確保目標資料夾存在
                            if not os.path.exists(target_dir):
                                os.makedirs(target_dir)
                            # 複製檔案
                            final_destination = os.path.join(
                                target_dir, os.path.basename(new_file_path))
                            shutil.copy(new_file_path, final_destination)
                            print(
                                f'已複製: {new_file_path} -> {final_destination}')

                            source_dir_1 = "nsynth-subset5/Piano"  # 原始資料夾
                            target_dir_1 = "nsynth-subset4/test"  # 目標資料夾

                            for root, dirs, files in os.walk(source_dir_1):
                                # 設定目標資料夾路徑
                                target_root = root.replace(
                                    source_dir_1, target_dir_1)

                                # 確保目標資料夾存在
                                if not os.path.exists(target_root):
                                    os.makedirs(target_root)

                                # 複製檔案
                                for file_name in files:
                                    if file_name.endswith('.npy') or file_name.endswith('.wav'):
                                        source_file_path = os.path.join(
                                            root, file_name)
                                        target_file_path = os.path.join(
                                            target_root, file_name)
                                        shutil.copy(
                                            source_file_path, target_file_path)
                                        print(
                                            f'已複製: {source_file_path} -> {target_file_path}')

    return new_file_name_no_ext


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InstrumentConverterApp()
    window.show()
    sys.exit(app.exec())
