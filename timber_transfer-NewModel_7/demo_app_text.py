# %%
import time
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import pyqtSlot, QUrl, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QRadioButton, QComboBox, QLineEdit, QGroupBox, QGridLayout, QGraphicsView, QGraphicsScene, QSlider
from components.timbre_transformer.TimberTransformer import TimbreTransformer
from tools.utils import cal_loudness_norm
from data.dataset import NSynthDataset
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
# from QFP8 import dequantize_float8_to_float32  # 確保導入反量化函數


def create_fig(data: ndarray) -> plt.Figure:
    fig = plt.figure()
    plt.plot(data)
    plt.close()
    return fig


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

    def sample_data(self, t: str = "source"):
        fn_with_path = random.choice(self.dataset.audio_list)
        fn = fn_with_path.split("\\")[-1][:-4]
        fn, s, _, _ = self.dataset.getitem_by_filename(fn)
        if t == "source":
            self.source_audio_file_name = fn
        else:
            self.target_audio_file_name = fn
        fig_s = create_fig(s)
        return fn, (16000, s), fig_s

    def sampel_source_audio_data(self):
        return self.sample_data("source")

    def sampel_target_audio_data(self):
        return self.sample_data("target")

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
            quantized_state_dict = torch.load(
                pt_file, map_location=torch.device('cpu'), weights_only=True)
            state_dict = {}
            for name, param in quantized_state_dict.items():
                if isinstance(param, tuple) and len(param) == 3 and isinstance(param[0], torch.Tensor):
                    quantized_tensor, scale, min_val = param
                    dequantized_tensor = dequantize_float8_to_float32(
                        quantized_tensor, scale, min_val)
                    state_dict[name] = dequantized_tensor
                elif isinstance(param, tuple):
                    dequantized_tuple = []
                    for item in param:
                        if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], torch.Tensor):
                            quantized_tensor, scale, min_val = item
                            dequantized_tensor = dequantize_float8_to_float32(
                                quantized_tensor, scale, min_val)
                            dequantized_tuple.append(dequantized_tensor)
                        else:
                            dequantized_tuple.append(item)
                    state_dict[name] = tuple(dequantized_tuple)
                else:
                    state_dict[name] = param
            self.model.load_state_dict(state_dict)
            self.current_pt_file_name = pt_file
            print("模型已成功加載！")

        except:
            raise RuntimeError("load model failed")
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
    def __init__(self, parent=None, label="Audio Player"):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 初始化播放器
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(100)  # 設置最大音量

        # 添加調試輸出
        self.player.errorOccurred.connect(self.handle_error)
        self.player.mediaStatusChanged.connect(self.handle_media_status)

        # UI 元件
        self.title_label = QLabel(label)
        self.time_label = QLabel("0.0 / 0.0")
        layout.addWidget(self.title_label)
        layout.addWidget(self.time_label)

        # 按鈕
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.pause_button = QPushButton("暫停")
        self.stop_button = QPushButton("停止")

        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.stop_button.clicked.connect(self.stop)

        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # 進度條
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.sliderMoved.connect(self.seek)
        layout.addWidget(self.slider)

        # 更新計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(100)

    def set_audio(self, audio_data, sample_rate):
        try:
            # 正規化音頻數據
            audio_data = audio_data.astype(np.float32)
            normalized_audio = audio_data / np.max(np.abs(audio_data))
            # 確保數據在 -1 到 1 之間
            normalized_audio = np.clip(normalized_audio, -1, 1)

            # 創建臨時文件
            temp_audio_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav")
            print(f"保存音頻到: {temp_audio_file.name}")

            # 保存為 WAV 文件
            sf.write(temp_audio_file.name, normalized_audio,
                     sample_rate, 'FLOAT', format='WAV')
            temp_audio_file.close()

            # 重置播放器
            self.player.stop()
            self.player = QMediaPlayer()  # 創建新的播放器實例
            self.audio_output = QAudioOutput()  # 創建新的音頻輸出
            self.player.setAudioOutput(self.audio_output)

            # 設置音量（使用百分比）
            self.audio_output.setVolume(1.0)  # 1.0 = 100%

            # 設置音頻源
            audio_url = QUrl.fromLocalFile(temp_audio_file.name)
            print(f"設置音頻源: {audio_url.toString()}")
            self.player.setSource(audio_url)

            # 啟用播放按鈕
            self.play_button.setEnabled(True)
            print(f"音頻載入完成")

        except Exception as e:
            print(f"音頻載入錯誤: {str(e)}")
            import traceback
            traceback.print_exc()

    def play(self):
        try:
            print("\n開始播放...")
            print(f"播放源: {self.player.source().toString()}")

            if not self.player.source().isValid():
                print("無效的音頻源")
                return

            # 設置最大音量並播放
            self.audio_output.setVolume(1.0)
            self.player.play()

            # 更新按鈕狀態
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)

        except Exception as e:
            print(f"播放錯誤: {str(e)}")
            import traceback
            traceback.print_exc()

    def pause(self):
        self.player.pause()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop(self):
        self.player.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def seek(self, position):
        self.player.setPosition(position)

    def update_position(self):
        position = self.player.position()
        duration = self.player.duration()
        self.time_label.setText(
            f'{str(round(position/1000, 1))} / {str(round(duration/1000, 1))}')

    def handle_error(self, error, error_string):
        print(f"播放器錯誤: {error} - {error_string}")

    def handle_media_status(self, status):
        print(f"媒體狀態: {status}")


G = GlobalInfo()


class DemoApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Demo App')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Data Mode Selector
        data_mode_group = QGroupBox("Data Mode")
        data_mode_layout = QHBoxLayout()
        self.data_mode_selector = QComboBox()
        self.data_mode_selector.addItems(["train", "valid", "test"])
        self.data_mode_selector.currentIndexChanged.connect(
            self.on_data_mode_changed)
        data_mode_layout.addWidget(self.data_mode_selector)
        data_mode_group.setLayout(data_mode_layout)
        main_layout.addWidget(data_mode_group)

        # Model pt file Selector
        pt_file_layout = QHBoxLayout()
        # pt_file_label = QLabel("Model pt file")
        # self.pt_file_selector = QComboBox()
        # self.pt_file_selector.addItems(G.pt_file_list)
        # self.pt_file_selector.currentIndexChanged.connect(
        #     self.on_pt_file_changed)
        # pt_file_layout.addWidget(pt_file_label)
        # pt_file_layout.addWidget(self.pt_file_selector)
        # main_layout.addLayout(pt_file_layout)

        # Dataset and pt file Textboxes
        # dataset_group = QGroupBox("Dataset and pt file")
        # dataset_layout = QGridLayout()
        # self.signal_name = QLineEdit()
        # self.signal_name.setPlaceholderText("Dataset")
        # self.pt_name = QLineEdit(G.current_pt_file_name)
        # self.pt_name.setPlaceholderText("pt file")
        # dataset_layout.addWidget(QLabel("Dataset"), 0, 0)
        # dataset_layout.addWidget(self.signal_name, 0, 1)
        # dataset_layout.addWidget(QLabel("pt file"), 1, 0)
        # dataset_layout.addWidget(self.pt_name, 1, 1)
        # dataset_group.setLayout(dataset_layout)
        # main_layout.addWidget(dataset_group)

        # Input Selection Section
        selection_group = QGroupBox("Input Selection")
        selection_layout = QHBoxLayout()

        # Source Selector
        source_selector_group = QGroupBox("Source")
        source_selector_layout = QVBoxLayout()
        self.source_radio_source = QRadioButton("source")
        self.source_radio_ref = QRadioButton("ref")
        self.source_radio_custom = QRadioButton("custom")
        self.source_radio_source.setChecked(True)
        source_selector_layout.addWidget(self.source_radio_source)
        source_selector_layout.addWidget(self.source_radio_ref)
        source_selector_layout.addWidget(self.source_radio_custom)
        source_selector_group.setLayout(source_selector_layout)

        # Source and Target Sections
        audio_section = QHBoxLayout()
        source_group = QGroupBox("Source")
        self.source_layout = QVBoxLayout()
        self.source_text = QLineEdit(G.source_audio_file_name)
        self.source_text.setPlaceholderText("source file")
        self.source_sample_button = QPushButton("source sample")
        # Create QGraphicsView for displaying the image
        self.source_image_view = QGraphicsView(self)
        self.source_image_view.setGeometry(0, 0, 360, 240)
        self.source_image_view.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)  # Align to top-left
        self.source_audio = QLabel("Source signal audio")
        self.source_layout.addWidget(self.source_text)
        self.source_layout.addWidget(self.source_sample_button)
        self.source_layout.addWidget(self.source_image_view)
        self.source_layout.addWidget(self.source_audio)
        source_group.setLayout(self.source_layout)

        self.source_audio_player = AudioPlayer(label="Source Audio")
        self.source_layout.addWidget(self.source_audio_player)

        target_group = QGroupBox("Target")
        self.target_layout = QVBoxLayout()
        self.target_text = QLineEdit(G.target_audio_file_name)
        self.target_text.setPlaceholderText("target file")
        self.target_sample_button = QPushButton("target sample")
        # Create QGraphicsView for displaying the image
        self.target_image_view = QGraphicsView(self)
        self.target_image_view.setGeometry(0, 0, 360, 240)
        self.target_image_view.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)  # Align to top-left
        self.target_audio = QLabel("Target signal audio")
        self.target_layout.addWidget(self.target_text)
        self.target_layout.addWidget(self.target_sample_button)
        self.target_layout.addWidget(self.target_image_view)
        self.target_layout.addWidget(self.target_audio)
        target_group.setLayout(self.target_layout)

        self.target_audio_player = AudioPlayer(label="Target Audio")
        self.target_layout.addWidget(self.target_audio_player)

        # Add source and target to horizontal layout
        audio_section.addWidget(source_group)
        audio_section.addWidget(target_group)
        # Add horizontal layout to main layout
        main_layout.addLayout(audio_section)

        # Input Selection Section
        selection_group = QGroupBox("Input Selection")
        selection_layout = QHBoxLayout()

        # Reference Selector
        ref_selector_group = QGroupBox("Reference")
        ref_selector_layout = QVBoxLayout()
        self.ref_radio_source = QRadioButton("source")
        self.ref_radio_ref = QRadioButton("ref")
        self.ref_radio_source.setChecked(True)
        ref_selector_layout.addWidget(self.ref_radio_source)
        ref_selector_layout.addWidget(self.ref_radio_ref)
        ref_selector_group.setLayout(ref_selector_layout)

        selection_layout.addWidget(source_selector_group)
        selection_layout.addWidget(ref_selector_group)
        selection_group.setLayout(selection_layout)
        main_layout.addWidget(selection_group)

        # Generation Section
        generation_group = QGroupBox("Generation")
        generation_layout = QVBoxLayout()

        self.generate_selection_text = QLineEdit()
        self.generate_selection_text.setPlaceholderText("Generate Selection")
        self.generate_selection_text.setReadOnly(True)

        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.on_generate_clicked)

        # Results Section
        results_group = QGroupBox("Results")
        results_layout = QHBoxLayout()

        # Rec Signal Image
        self.rec_image_view = QGraphicsView(self)
        self.rec_image_view.setGeometry(0, 0, 360, 240)

        # Rec Signal Audio
        self.rec_audio_player = AudioPlayer(label="Rec Signal")

        results_layout.addWidget(self.rec_image_view)
        results_layout.addWidget(self.rec_audio_player)
        results_group.setLayout(results_layout)

        generation_layout.addWidget(self.generate_selection_text)
        generation_layout.addWidget(self.generate_button)
        generation_layout.addWidget(results_group)
        generation_group.setLayout(generation_layout)
        main_layout.addWidget(generation_group)

        # 連接信號
        self.source_radio_source.toggled.connect(self.on_model_input_changed)
        self.source_radio_ref.toggled.connect(self.on_model_input_changed)
        self.source_radio_custom.toggled.connect(self.on_model_input_changed)
        self.ref_radio_source.toggled.connect(self.on_model_input_changed)
        self.ref_radio_ref.toggled.connect(self.on_model_input_changed)

        self.source_sample_button.clicked.connect(
            self.source_sample_button_clicked)
        self.target_sample_button.clicked.connect(
            self.target_sample_button_clicked)

    @pyqtSlot()
    def on_data_mode_changed(self):
        selected_mode = self.data_mode_selector.currentText()
        new_mode = G.change_dataset(selected_mode)
        self.signal_name.setText(f"Selected mode: {new_mode}")

    @pyqtSlot()
    def on_pt_file_changed(self):
        selected_pt = self.pt_file_selector.currentText()
        new_pt = G.change_pt_file(selected_pt)
        self.pt_name.setText(f"Selected mode: {new_pt}")

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
        self.source_image_scene.setSceneRect(0, 0, 340, 220)
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
        self.target_image_scene.setSceneRect(0, 0, 340, 220)
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
            self.rec_image_scene.setSceneRect(0, 0, 340, 220)
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
