import torch
import os
import numpy as np
from glob import glob


class NSynthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_mode: str,
        dir_path: str = "./nsynth-subset5",  # 預設為 nsynth-subset5
        sr: int = 16000,
        frequency_with_confidence: bool = False,
        with_f0_distanglement: bool = False,
        category: str = 'Bass',  # 新增參數來選擇資料夾的分類
    ):
        super().__init__()
        self.sr = sr
        self.dir_path = dir_path
        self.category = category  # 設置分類
        self.set_data_mode(data_mode)
        self.with_f0_distanglement = with_f0_distanglement
        self.info_type = (
            {
                "signal": "signal",
                "loudness": "loudness",
                "frequency": "frequency_c",
            }
            if frequency_with_confidence
            else
            {
                "signal": "signal",
                "loudness": "loudness",
                "frequency": "frequency",
            }
        )

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        signal_path = self.audio_list[idx]
        file_name = signal_path.split("\\")[-1][:-4]
        signal = np.load(
            os.path.join(self.data_mode_dir_path,
                         f"{self.info_type['signal']}\\{file_name}.npy")
        ).astype("float32")
        loudness = np.load(
            os.path.join(self.data_mode_dir_path,
                         f"{self.info_type['loudness']}\\{file_name}.npy")
        ).astype("float32")[..., :-1]
        frequency = np.load(
            os.path.join(self.data_mode_dir_path,
                         f"{self.info_type['frequency']}\\{file_name}.npy")
        ).astype("float32")

        if self.with_f0_distanglement:
            frequency = self.f0_distanglement_enhance(frequency)

        return (file_name, signal, loudness, frequency)

    def set_data_mode(self, data_mode: str):
        self.data_mode = data_mode
        # 根據分類修改資料路徑
        self.data_mode_dir_path = os.path.join(self.dir_path, self.category, data_mode)
        signal_path = os.path.join(self.data_mode_dir_path, "signal\\*")
        self.audio_list = glob(signal_path)

    def getitem_by_filename(self, fn: str):
        idx = self.audio_list.index(os.path.join(
            self.data_mode_dir_path, f"{self.info_type['signal']}\\{fn}.npy"))
        return self.__getitem__(idx)

    def update_audio_list(self):
        """動態更新檔案列表"""
        signal_path = os.path.join(self.data_mode_dir_path, "signal\\*")
        self.audio_list = glob(signal_path)
        print(f"Audio list updated: {len(self.audio_list)} files found.")

    def f0_distanglement_enhance(self, f0: np.ndarray):
        tmp_f0 = f0.copy()
        non_zero_f0 = tmp_f0[tmp_f0 != 0]
        f0_mean = np.mean(non_zero_f0)
        f0_std = np.std(non_zero_f0)
        scale_mean = np.random.uniform(0.6, 1.5)
        scale_std = np.random.uniform(0.9, 1.2)

        # Adjust mean
        adjusted_f0 = non_zero_f0 * scale_mean
        # Adjust std
        adjusted_std = f0_std * scale_std

        adjusted_f0 = ((adjusted_f0 - f0_mean) / f0_std) * \
            adjusted_std + f0_mean
        tmp_f0[tmp_f0 != 0] = adjusted_f0
        return tmp_f0


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # 設定 `category` 為你需要的分類 (例如：'Bass')
    train_dataset = NSynthDataset(data_mode="train", sr=16000, category='Bass')
    train_loader = DataLoader(
        train_dataset, batch_size=4, num_workers=1, shuffle=True)
    for fn, s, l, f in train_loader:
        print(fn)
