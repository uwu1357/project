# %%
import time
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import pyqtSlot, QUrl, QTimer, Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap   # 正確匯入 QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QRadioButton, QComboBox, QLineEdit, QGroupBox, QGridLayout, QGraphicsView, QGraphicsScene, QSlider, QListWidget, QListWidgetItem, QSizePolicy
from components.timbre_transformer.TimberTransformer import TimbreTransformer
from tools.utils import cal_loudness_norm
from data.dataset2 import NSynthDataset
import os
import torch
import random
from glob import glob
from numpy import ndarray
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
import tempfile
import sys
sys.path.append("..")


def create_fig(data: ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))  # 設定固定大小
    ax.plot(data)
    ax.set_xlim(0, len(data))  # 確保 X 軸顯示完整
    ax.set_ylim(np.min(data), np.max(data))  # 確保 Y 軸顯示完整
    plt.tight_layout()  # 自動調整佈局
    return fig


def update_chart_view(chart_view, image, width=450, height=400):
    canvas = FigureCanvas(image)
    scene = QGraphicsScene()
    scene.addWidget(canvas)
    chart_view.setScene(scene)
    # 讓圖表填滿整個 View
    chart_view.fitInView(scene.itemsBoundingRect(),
                         Qt.AspectRatioMode.KeepAspectRatio)


def transform_frequency(frequency, semitone_shift):
    """
    Transform a frequency by a given number of semitones.

    Parameters:
    frequency (float): The original frequency in Hz
    semitone_shift (int): Number of semitones to shift (positive for higher pitch, negative for lower pitch)

    Returns:
    float: The transformed frequency in Hz
    """
    transformed_frequency = frequency * (2 ** (semitone_shift / 12))
    return transformed_frequency


class GlobalInfo:
    def __init__(self):
        pt_dir = ".\pt_file"
        run_name = "decoder_v21_6_addmfftx3_energy_ftimbreE"
        self.current_pt_file_name = f"{run_name}_generator_best_12.pt"
        self.pt_file = f"{pt_dir}\\{self.current_pt_file_name}"
        self.pt_file_list = sorted(glob(f"{pt_dir}\\{run_name}*.pt"))
        self.model = TimbreTransformer(
            is_train=False, is_smooth=True, timbre_emb_dim=256)
        self.dataset = NSynthDataset(
            data_mode="train", sr=16000, frequency_with_confidence=True)
        self.source_audio_file_name = None
        self.target_audio_file_name = None
        self.model_input_selection = ("source", "source")
        self.model.eval()
        self.model.load_state_dict(torch.load(
            self.pt_file, map_location=torch.device('cpu'), weights_only=True))

    def sample_data(self, file_name: str, t: str = "source"):
        """
        根據檔案名稱取得對應的數據。

        Parameters:
        file_name (str): 檔案名稱（不含路徑與副檔名）
        t (str): 資料類型，"source" 或 "target"

        Returns:
        tuple: 檔案名稱、音頻數據 (取樣率, 音頻陣列)、圖像數據
        """
        fn, s, _, _ = self.dataset.getitem_by_filename(file_name)
        if t == "source":
            self.source_audio_file_name = fn
        else:
            self.target_audio_file_name = fn
        fig_s = create_fig(s)
        return fn, (16000, s), fig_s

    def sampel_source_audio_data(self, file_name: str):
        return self.sample_data(file_name, "source")

    def sampel_target_audio_data(self, file_name: str):
        return self.sample_data(file_name, "target")

    def generate_model_input(self):
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
        traget_midi = get_midi(self.target_audio_file_name)
        midi_table = {
            "source": source_midi,
            "ref": traget_midi,
            "custom": 60
        }
        semitone_shift = midi_table[self.model_input_selection[1]
                                    ] - midi_table[self.model_input_selection[0]]
        new_f = transform_frequency(f, semitone_shift)
        rec_s = self.model_gen(s, cal_loudness_norm(
            l), new_f, ref).squeeze().detach().numpy()
        fig_rec_s = create_fig(rec_s)

        return (16000, rec_s), fig_rec_s

    def model_gen(self, s: ndarray, l_norm: ndarray, f: ndarray, timbre_s: ndarray):
        def transfrom(x_array): return torch.from_numpy(x_array).unsqueeze(0)
        s, l_norm, f, timbre_s = transfrom(s), transfrom(
            l_norm), transfrom(f), transfrom(timbre_s)
        f = f[:, :-1, 0]
        _, _, rec_s, _, _, _ = self.model(s, l_norm, f, timbre_s)
        return rec_s

    def change_dataset(self, data_mode: str) -> str:
        self.dataset.set_data_mode(data_mode)
        return self.dataset.data_mode

    def change_pt_file(self, pt_file: str):
        self.current_pt_file_name = pt_file.split("\\")[-1]
        try:
            self.model.load_state_dict(torch.load(
                pt_file, map_location=torch.device('cpu')))
        except:
            raise gr.Error("load model failed")
        return self.current_pt_file_name

    def change_model_input(self, source: str, ref: str):
        selection = [source, ref]
        for i, item in enumerate(selection):
            if item == None:
                selection[i] = "source"
        self.model_input_selection = selection
        return f"Source: {selection[0]}, Ref: {selection[1]}"

    def change_source_audio_file_name(self, source_audio_file_name: str):
        fn_with_path = source_audio_file_name
        fn = fn_with_path.split("\\")[-1][:-4]
        fn, s, _, _ = self.dataset.getitem_by_filename(fn)
        self.source_audio_file_name = fn
        fig_s = create_fig(s)
        return fn, (16000, s), fig_s


class AudioPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 初始化播放器
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)  # 設置音量為 100%

    def set_audio(self, audio_data, sample_rate):
        """設置音頻數據並準備播放"""
        try:
            # 正規化音頻數據
            audio_data = audio_data.astype(np.float32)
            normalized_audio = audio_data / np.max(np.abs(audio_data))
            normalized_audio = np.clip(normalized_audio, -1, 1)

            # 保存為臨時 WAV 文件
            temp_audio_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav")
            sf.write(temp_audio_file.name, normalized_audio,
                     sample_rate, 'FLOAT', format='WAV')
            temp_audio_file.close()

            # 設置音頻源
            audio_url = QUrl.fromLocalFile(temp_audio_file.name)
            self.player.setSource(audio_url)

            # 啟用播放按鈕
            # self.play_button.setEnabled(True)
            print(f"音頻已載入: {temp_audio_file.name}")
        except Exception as e:
            print(f"音頻載入錯誤: {str(e)}")

    def play(self):
        """播放音頻"""
        if self.player.source().isValid():
            self.player.setPosition(0)
            self.player.play()
            print("開始播放音頻")
        else:
            print("無效的音頻源")

    def pause(self):
        """暫停音頻"""
        self.player.pause()
        print("音頻已暫停")


G = GlobalInfo()


class DemoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.paths = {
            "Bass":     "./nsynth-subset5/Bass/test/signal",
            "Brass":    "./nsynth-subset5/Brass/test/signal",
            "Flute":    "./nsynth-subset5/Flute/test/signal",
            "Guitar":   "./nsynth-subset5/Guitar/test/signal",
            "Piano":    "./nsynth-subset5/Piano/signal",
            "Mallet":   "./nsynth-subset5/Mallet/test/signal",
            "Audio":    "./nsynth-subset5/audio/signal",
        }

        self.image_paths_1 = {
            "Bass": "instrument_image/bass.png",
            "Brass": "instrument_image/brass.png",
            "Flute": "instrument_image/flute.png",
            "Guitar": "instrument_image/guitar.png",
            "Piano": "instrument_image/piano.png",
            "Mallet": "instrument_image/mallet.png",
            # "Audio" : "instrument_image/musical.png",
        }

        self.image_paths_2 = {
            "Bass": "instrument_image/bass.png",
            "Brass": "instrument_image/brass.png",
            "Flute": "instrument_image/flute.png",
            "Guitar": "instrument_image/guitar.png",
            "Piano": "instrument_image/piano.png",
            "Mallet": "instrument_image/mallet.png"
        }

        self.right_buttons = {
            "Play Music":       self.play_music,
            "Pause":            self.pause_music,
            # "Visual Keyboard":  self.switch_piano_page,
            # "Microphone":       self.switch_recording_page,
            "NEXT":             self.switch_page,
        }

        self.target_files = [
            "bass_acoustic_000-039-127", "brass_acoustic_059-048-075",
            "flute_acoustic_023-067-050", "guitar_acoustic_022-058-100",
            "piano20250109_051732_000-052-050", "mallet_acoustic_032-037-075"
        ]

        self.initUI()

    def init_source_chart(self):
        if G.source_audio_file_name is not None:
            source_text, source_audio, source_image = G.sampel_source_audio_data(G.source_audio_file_name)
            # 改用 update_chart_view 統一處理，使用 fitInView
            update_chart_view(self.source_chart_view, source_image)

    def initUI(self):
        self.setWindowTitle('Demo App')
        self.showMaximized()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(150, 0, 0, 0)  # 設定左邊距離

        # 左邊圖片按鈕區域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.image_buttons = {}
        for key, image_path in self.image_paths_1.items():
            btn = QPushButton()
            btn.setIcon(QIcon(image_path))
            # icon 固定方形
            btn.setIconSize(QSize(80, 80))
            # 按鈕本體長方形 (寬 = 高 * 2)
            init_h = 100
            init_w = init_h * 2
            btn.setFixedSize(init_w, init_h)
            btn.clicked.connect(lambda _, k=key: self.on_image_button_clicked(k))

            self.image_buttons[key] = btn
            left_layout.addWidget(btn)
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget)

        # 中間選單區域
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        self.path_selector = QComboBox()
        self.file_list_widget = QListWidget()
        self.path_selector.setStyleSheet("font-size: 24px;")
        self.file_list_widget.setStyleSheet("font-size: 24px;")
        self.path_selector.addItems(self.paths.keys())
        self.path_selector.currentIndexChanged.connect(self.on_path_selected)
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        middle_layout.addWidget(self.file_list_widget)
        middle_widget.setLayout(middle_layout)
        middle_external_layout = QHBoxLayout()
        middle_external_layout.setContentsMargins(0, 0, 100, 0)
        middle_external_layout.addWidget(middle_widget)

        main_layout.addLayout(middle_external_layout)
        # 初始化檔案列表
        self.update_file_list_widget(self.path_selector.currentText())

        # 右邊按鈕區域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        for button_text, button_action in self.right_buttons.items():
            button = QPushButton(button_text)
            button.setFixedSize(300, 120)  # 固定按鈕大小
            button.setStyleSheet("font-size: 24px;")
            button.clicked.connect(button_action)
            right_layout.addWidget(button)
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget)
        self.audio_player = AudioPlayer(self)

        main_layout.setStretch(0, 2)  # 左邊區域
        main_layout.setStretch(1, 6)  # 中間區域
        main_layout.setStretch(2, 2)  # 右邊區域

    def initUI_2(self):
        """第二頁介面：左側圖片區、右側三個圖表區，下方為導覽按鈕"""
        self.setWindowTitle('Demo App - Page 2')
        self.showMaximized()

        self.target_audio_player = AudioPlayer(self)
        self.rec_audio_player = AudioPlayer(self)
        # 主佈局（垂直排列）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 0, 0, 0)  # 保留左邊邊距

        # 標題
        title_label = QLabel("Generation Results", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # 上半部：內容分左右兩區
        upper_layout = QHBoxLayout()
        upper_layout.setSpacing(0)                 # 消除左右區塊之間的間隙
        upper_layout.setContentsMargins(0, 0, 0, 0)  # 消除邊距

        # 左側：垂直排列圖片按鈕 (例如樂器圖示)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.image_buttons = {}
        for key, image_path in self.image_paths_2.items():
            btn = QPushButton()
            btn.setIcon(QIcon(image_path))
            # icon 固定方形
            btn.setIconSize(QSize(80, 80))
            # 按鈕本體固定正方形
            btn.setFixedSize(100, 100)
            btn.clicked.connect(lambda _, fn=self.target_files[list(self.image_paths_2).index(key)]: 
                                self.select_instrument(fn))

            self.image_buttons[key] = btn
            left_layout.addWidget(btn)
        left_widget.setLayout(left_layout)
        upper_layout.addWidget(left_widget,0.5)

        # 右側：水平排列三個圖表區 (Source、Target、Rec)
        charts_widget = QWidget()
        charts_layout = QHBoxLayout(charts_widget)
        charts_layout.setSpacing(5)
        charts_layout.setContentsMargins(0, 0, 0, 0)

        # Source 圖組
        source_group = QGroupBox("Source")
        source_group.setStyleSheet("""
            QGroupBox { 
                font-size: 18px; 
                font-weight: bold; 
                border: none;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 70px 10px 0 10px;
                background-color: transparent;
            }
        """) 
        source_layout = QVBoxLayout()
        self.source_chart_view = QGraphicsView(self)
        self.source_chart_view.setFixedSize(450, 400)
        self.source_chart_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        source_layout.addWidget(self.source_chart_view)
        source_group.setLayout(source_layout)
        charts_layout.addWidget(source_group)

        # Target 圖組
        target_group = QGroupBox("Target")
        target_group.setStyleSheet("""
            QGroupBox { 
                font-size: 18px; 
                font-weight: bold; 
                border: none;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 70px 10px 0 10px;
                background-color: transparent;
            }
        """) 
        target_layout = QVBoxLayout()
        self.target_chart_view = QGraphicsView(self)
        self.target_chart_view.setFixedSize(450, 400)
        self.target_chart_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        target_layout.addWidget(self.target_chart_view)
        target_group.setLayout(target_layout)
        charts_layout.addWidget(target_group)

        # Rec 圖組
        rec_group = QGroupBox("Rec")
        rec_group.setStyleSheet("""
            QGroupBox { 
                font-size: 18px; 
                font-weight: bold; 
                border: none;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 70px 10px 0 10px;
                background-color: transparent;
            }
        """)

        rec_layout = QVBoxLayout()
        self.rec_chart_view = QGraphicsView(self)
        self.rec_chart_view.setFixedSize(450, 400)
        self.rec_chart_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rec_layout.addWidget(self.rec_chart_view)
        rec_group.setLayout(rec_layout)
        charts_layout.addWidget(rec_group)

        charts_widget.setLayout(charts_layout)
        upper_layout.addWidget(charts_widget, 3)  # 右側區域佔比例 2

        main_layout.addLayout(upper_layout)

        # 下半部：導覽按鈕區域

        navigation_button_layout = QHBoxLayout()
        navigation_button_layout.setContentsMargins(0, 0, 0, -20)
        navigation_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        navigation_button_layout.setSpacing(100)

        start_generation_button = QPushButton("Start Generation", self)
        start_generation_button.setFixedSize(150, 50)
        start_generation_button.clicked.connect(self.start_generation)
        navigation_button_layout.addWidget(start_generation_button)

        play_generation_button = QPushButton("Play Generated", self)
        play_generation_button.setFixedSize(150, 50)
        play_generation_button.clicked.connect(self.play_generated)
        navigation_button_layout.addWidget(play_generation_button)

        back_button = QPushButton("Back", self)
        back_button.setFixedSize(150, 50)
        back_button.clicked.connect(self.switch_back_page)
        navigation_button_layout.addWidget(back_button)

        # next_button = QPushButton("Next", self)
        # next_button.setFixedSize(100, 40)
        # next_button.clicked.connect(self.switch_next_page)
        # navigation_button_layout.addWidget(next_button)

        # 你也可以新增其他按鈕
        main_layout.addLayout(navigation_button_layout)

        if G.source_audio_file_name is not None:
            QTimer.singleShot(100, self.init_source_chart)

    def resizeEvent(self, event):
        if hasattr(self, 'image_buttons'):
            if self.windowTitle() == "Demo App - Page 2":
                # 第二頁不動
                return super().resizeEvent(event)

            # 第一頁：寬 = 高 * 2，隨視窗高度改變
            new_h = max(min(self.height() // 5, 100), 100)
            new_w = new_h * 2
            for btn in self.image_buttons.values():
                btn.setFixedSize(new_w, new_h)
                btn.setIconSize(QSize(int(new_w * 0.8), int(new_h * 0.8)))

        super().resizeEvent(event)

    @pyqtSlot()
    def play_music(self):
        self.audio_player.play()

    @pyqtSlot()
    def pause_music(self):
        self.audio_player.pause()
        print("Pause button clicked")

    @pyqtSlot()
    def switch_piano_page(self):
        print("Switch to Visual Keyboard page")

    @pyqtSlot()
    def switch_page(self):
        print("NEXT button clicked")
        self.initUI_2()

    @pyqtSlot()
    def on_image_button_clicked(self, key):
        self.path_selector.setCurrentText(key)

    @pyqtSlot()
    def on_path_selected(self):
        selected_path = self.path_selector.currentText()
        print(f"Path selected: {self.paths[selected_path]}")
        self.update_file_list_widget(selected_path)

    @pyqtSlot(QListWidgetItem)
    def on_file_selected(self, item):
        """當選擇檔案時，利用 G.sampel_source_audio_data() 更新資料"""
        file_name = item.text()  # 取得檔案名稱
        file_name = os.path.splitext(file_name)[0]
        print(f"File selected: {file_name}")
        source_text, source_audio, source_image = G.sampel_source_audio_data(
            file_name)
        sample_rate, audio_data = source_audio
        print(f"Source Text: {source_text}")
        # 傳遞音頻數據到播放器
        self.audio_player.set_audio(audio_data, sample_rate)
        # 在這裡可以更新其他 UI 元件，例如顯示音頻波形或播放音頻

    def update_file_list_widget(self, path_key):
        """更新檔案列表內容"""
        path = self.paths[path_key]
        if os.path.exists(path):
            files = os.listdir(path)
            display_files = [os.path.splitext(file)[0] for file in files]
            self.file_list_widget.clear()
            self.file_list_widget.addItems(display_files)
        else:
            print(f"Path does not exist: {path}")
            self.file_list_widget.clear()

    @pyqtSlot()
    def select_instrument(self, file_name):
        """處理第二頁圖片點擊事件"""
        print(f"Selected instrument: {file_name}")

        # 獲取目標數據
        target_text, target_audio, target_image = G.sampel_target_audio_data(
            file_name)
        G.change_model_input("source", "ref")
        update_chart_view(self.target_chart_view, target_image)

        # 播放音頻
        sample_rate, audio_data = target_audio
        self.target_audio_player.set_audio(audio_data, sample_rate)
        self.target_audio_player.play()

    @pyqtSlot()
    def start_generation(self):
        print("Start Generation clicked")
        try:
            start = time.time()
            rec_audio, rec_image = G.generate_output()

            # 使用統一的圖表更新方法更新所有圖表
            if G.source_audio_file_name is not None:
                source_text, source_audio, source_image = G.sampel_source_audio_data(
                    G.source_audio_file_name)
                update_chart_view(self.source_chart_view, source_image)

            if G.target_audio_file_name is not None:
                target_text, target_audio, target_image = G.sampel_target_audio_data(
                    G.target_audio_file_name)
                update_chart_view(self.target_chart_view, target_image)

            # 更新重建信號圖表
            update_chart_view(self.rec_chart_view, rec_image)

            # 更新重建信號音頻
            sample_rate, audio_data = rec_audio
            self.rec_audio_player.set_audio(audio_data, sample_rate)
            end = time.time()
            print(f"generation time: {end - start}")
        except Exception as e:
            print(f"生成錯誤: {str(e)}")

    @pyqtSlot()
    def play_generated(self):
        print("Play Generated clicked")
        # 確認已經設定了 rec_audio_player
        try:
            self.rec_audio_player.play()
        except Exception as e:
            print(f"播放 generated 音檔失敗: {str(e)}")

    @pyqtSlot()
    def switch_back_page(self):
        print("Back button clicked")
        current_widget = self.centralWidget()
        if current_widget is not None:
            current_widget.deleteLater()
        self.initUI()

    @pyqtSlot()
    def switch_next_page(self):
        print("Next button clicked")
        # 如果有第三頁，可以在這裡切換到第三頁

    @pyqtSlot()
    def source_sample_button_clicked(self):
        source_text, source_audio, source_image = G.sampel_source_audio_data()
        self.source_text.setText(source_text)
        self.source_audio.setText(source_text)  # Display the file name

        # Remove previous canvas if exists
        if hasattr(self, 'source_image_scene'):
            self.source_image_scene.setParent(None)

        # Create a FigureCanvas from the Figure
        canvas = FigureCanvas(source_image)
        self.source_image_scene = QGraphicsScene()
        self.source_image_scene.setSceneRect(0, 0, 350, 300)
        self.source_image_scene.addWidget(canvas)
        self.source_image_view.setScene(self.source_image_scene)

        sample_rate, audio_data = source_audio
        self.source_audio_player.set_audio(audio_data, sample_rate)

    @pyqtSlot()
    def target_sample_button_clicked(self):
        target_text, target_audio, target_image = G.sampel_target_audio_data()
        self.target_text.setText(target_text)
        self.target_audio.setText(target_text)  # Display the file name

        # Remove previous canvas if exists
        if hasattr(self, 'target_image_scene'):
            self.target_image_scene.setParent(None)

        # Create a FigureCanvas from the Figure
        canvas = FigureCanvas(target_image)
        self.target_image_scene = QGraphicsScene()
        self.target_image_scene.setSceneRect(0, 0, 300, 210)
        self.target_image_scene.addWidget(canvas)
        self.target_image_view.setScene(self.target_image_scene)

        sample_rate, audio_data = target_audio
        self.target_audio_player.set_audio(audio_data, sample_rate)

    @pyqtSlot()
    def on_model_input_changed(self):
        source = "source"
        if self.source_radio_ref.isChecked():
            source = "ref"
        elif self.source_radio_custom.isChecked():
            source = "custom"

        ref = "source" if self.ref_radio_source.isChecked() else "ref"

        result = G.change_model_input(source, ref)
        self.generate_selection_text.setText(result)

    @pyqtSlot()
    def on_generate_clicked(self):
        try:
            start = time.time()
            rec_audio, rec_image = G.generate_output()

            # 更新重建信號圖像
            if hasattr(self, 'rec_image_scene'):
                self.rec_image_scene.setParent(None)
            canvas = FigureCanvas(rec_image)
            self.rec_image_scene = QGraphicsScene()
            self.rec_image_scene.setSceneRect(0, 0, 300, 210)
            self.rec_image_scene.addWidget(canvas)
            self.rec_image_view.setScene(self.rec_image_scene)

            # 更新重建信號音頻
            sample_rate, audio_data = rec_audio
            self.rec_audio_player.set_audio(audio_data, sample_rate)
            end = time.time()
            print(f"generation time: {end - start}")
        except Exception as e:
            print(f"生成錯誤: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = DemoApp()
    demo.show()
    sys.exit(app.exec())
# %%
