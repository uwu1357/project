import torch
import json
from librosa.filters import mel as librosa_mel_fn
import numpy as np
from numpy import ndarray
from torch import Tensor


from data.dataset import NSynthDataset
from components.timbre_transformer.utils import mean_std_loudness

def get_mean_std_dict(data_mode: str, batch: int=32):
    mean_std_dict = {}
    dataset = NSynthDataset(data_mode=data_mode, sr=16000)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=8)
    mean_std_dict["mean_loudness"], mean_std_dict["std_loudness"]= mean_std_loudness(valid_loader)
    return mean_std_dict

def cal_mean_std_loudness(loudness, mean_std_dict):
    return (loudness - mean_std_dict["mean_loudness"]) / mean_std_dict["std_loudness"]

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis
    hann_window = torch.hann_window(win_size).to(y.device)
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)

    y = torch.nn.functional.pad(y.squeeze(-1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def safe_log(x):
    return torch.log(x + 1e-7)

def multiscale_fft(
    signal: Tensor,
    scales: list= [4096, 2048, 1024, 512, 256, 128],
    overlap: float=0.75,
    ) -> Tensor:
    stfts = []
    for scale in scales:
        S = torch.stft(
            input = signal,
            n_fft = scale,
            hop_length = int(scale * (1 - overlap)),
            win_length = scale,
            window = torch.hann_window(scale).to(signal),
            center = True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def get_hyparam():
    """Get hyperparameters from config.json"""
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    with open("./config.json") as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)

    return h

def cal_loudness_norm(l: ndarray):
    mean_loudness = -20.235496416593854
    std_loudness = 36.28924789428713
    return (l - mean_loudness) / std_loudness

# write a funcion make frequency transofrom to MIDI
def get_midi_from_frequency(frequency: Tensor):
    midi_value = 69 + 12 * torch.log2(frequency / 440) 
    # fix midi_value, if midi_value < 0, set it to 0
    if midi_value.min() < 0:
        midi_value[midi_value < 0] = 0
    return  midi_value

def seperate_f0_confidence(f0_with_confidence: ndarray):
    f0, f0_confidence = f0_with_confidence[..., 0][...,: -1], f0_with_confidence[..., 1][...,: -1]
    return f0, f0_confidence

def mask_f0_with_confidence(f0_with_confidence: ndarray, threshold: float=0.85, return_midi: bool=True):
    f0, f0_confidence = f0_with_confidence[..., 0][...,: -1], f0_with_confidence[..., 1][...,: -1]
    if return_midi:
        f0 = get_midi_from_frequency(f0)
    f0[f0_confidence < threshold] = torch.nan
    return f0

def replace_zero_with_nan(arr):
    return np.where(arr == 0, np.nan, arr)

def cal_mean_for_loudness_after_mask(arr, window_size=1024, hop_size=256):
    if len(arr.shape) == 1:
        arr = arr.reshape(1, -1)
    mean = []
    for i in range(0, arr.shape[1], hop_size):
        mean.append(arr[:, i:i+window_size].mean(axis=1))
    return np.array(mean)

def get_loudness_mask(signal):
    mask = ~np.isnan(replace_zero_with_nan(signal))
    return cal_mean_for_loudness_after_mask(mask)

    

    
