# Adapted from https://github.com/NVIDIA/BigVGAN
# Build MPD(Multi-Period waveform Discriminator) and MRD(Multi-Resolution spectrogram Discriminator)
"""
MRD & MPD setting from BigVGAN paper
n_fft_i       |  [1024, 2048, 512]
hot_length_i  |  [120, 240, 50]
win-length_i  |  [600, 1200, 240]
Reshaped2d_i  |  [2, 3, 5, 7, 11]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm

from .utils import get_padding


# from BigVGAN paper
class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 256, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(256, 512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, kernel_size=(kernel_size, 1), stride=1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        
    def forward(self, x):
        fmap = []
        
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, mpd_reshapes=[2, 3, 5, 7, 11]):
        super().__init__()
        self.mpd_reshapes = mpd_reshapes
        print(f"mpd_reshapes: {mpd_reshapes}")
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period=rs) for rs in mpd_reshapes]
            )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
            
# from BigVGAN paper
class DiscriminatorR(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        norm_f = weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1 ))),
        ])
        self.conv_post = norm_f(Conv2d(32 , 1, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectromgram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x ,1 , -1)

        return x, fmap

    def spectromgram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode="reflect")
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x) # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1) # [B, F, TT]

        return mag

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, mrd_resolutions=[[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]):
        super().__init__()
        self.mrd_resolutions = mrd_resolutions
        assert len(self.mrd_resolutions) == 3,\
            f"MRD requires list of list with len=3, each element having a list with len=3, got{mrd_resolutions}"
        print(f"mrd_resolutions: {mrd_resolutions}")
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution=rs) for rs in mrd_resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

