import torch
import torch.nn as nn
import torchaudio
from functools import partial


# from nvc-net paper
class TimbreEncoder(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        n_mfcc=80,
        timbre_emb_dim=256,
        ):

        super().__init__()

        self.extract_mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs=dict(
                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=20.0, f_max=8000.0
            )
        )
            
        self.conv = nn.Sequential(
            nn.Conv1d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.downblocks_list = nn.ModuleList(
            [self.build_downblock(32 * 2 ** i) for i in range(4) ]
        )

        self.downblocks = nn.Sequential(*self.downblocks_list)

        self.conv_mean = nn.Conv1d(512, timbre_emb_dim, kernel_size=1, stride=1, padding=0)

        self.conv_covariance = nn.Conv1d(512, timbre_emb_dim, kernel_size=1, stride=1, padding=0)

    def build_downblock(self, in_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool1d(2),
        )
    
    def forward(self, x):
        x = self.extract_mfcc(x)
        x = self.conv(x)
        x = self.downblocks(x)
        x = nn.AvgPool1d(x.size(-1))(x)
        mu = self.conv_mean(x)
        logvar = self.conv_covariance(x)

        return mu, logvar


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in != self.dim_out:
            self.s_conv = nn.utils.weight_norm(nn.Conv1d(self.dim_in, self.dim_out, (1, ), bias=False))
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(self.dim_in, self.dim_in, (3, ), padding=(1, )))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(self.dim_in, self.dim_out, (1, )))
        self.avgpool = nn.AvgPool1d(kernel_size=(2,))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        sc = x.clone()
        if self.dim_in != self.dim_out:
            sc = self.s_conv(sc)
            sc = self.avgpool(sc)
        else:
            sc = self.avgpool(sc)

        x = self.lrelu(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.lrelu(x)
        x = self.conv2(x)

        return x + sc


class TimbreEncoderX(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        timbre_emb_dim=256,
        ):

        super().__init__()
        self.dim = timbre_emb_dim
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=8000.0
            )
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(80, 32, kernel_size=(3,), padding=(1, )))
        self.res_block1 = ResBlock(32, 64)
        self.res_block2 = ResBlock(64, 128)
        self.res_block3 = ResBlock(128, 256)
        self.res_block4 = ResBlock(256, 512)
        self.res_block5 = ResBlock(512, 512)
        self.avgpool = nn.AdaptiveAvgPool1d((1, ))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.m_conv = nn.utils.weight_norm(nn.Conv1d(512, self.dim, kernel_size=(1,)))
        self.v_conv = nn.utils.weight_norm(nn.Conv1d(512, self.dim, kernel_size=(1,)))

    
    def forward(self, x):
        out = self.mel_spec(x)
        out = torch.squeeze(out, 1)
        # out = out * 1e4 + 1
        out = torch.log(out + 1e-6)
        out = self.conv1(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.res_block5(out)
        out = self.avgpool(out)
        # out = torch.squeeze(out, -1)
        out = self.lrelu(out)
        mu = self.m_conv(out)
        logvar = self.v_conv(out)
        return mu, logvar
 
class ZEncoder(nn.Module):
    def __init__(self, nfft=1024, hop_lenght=256, z_units=16, hidden_size=256):
        super().__init__()
        self.nfft = nfft
        self.hop_lenght = hop_lenght
        input_size = nfft // 2 + 1
        self.norm = nn.InstanceNorm1d(input_size, affine=True)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, z_units)
    
    def forward(self, x):
        x = torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.hop_lenght,
            win_length=self.nfft,
            center=True,
            return_complex=True,
        )
        x = x[..., :-1]
        x = torch.abs(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous() # (batch, nfft, frame) -> (batch, frame, nfft)
        x = self.gru(x)[0]
        x = self.dense(x)
        return x



class ZMFCCEncoder(nn.Module):
    def __init__(
        self,
        nfft=1024,
        hop_lenght=256,
        z_units=16,
        gru_units=256,
        n_mfcc=30,
        n_mels=128,
        ):
        super().__init__()
        self.extract_mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=nfft, hop_length=hop_lenght, n_mels=n_mels, f_min=20.0, f_max=8000.0
            )
        )
            
        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            batch_first=True,
        )
        self.dense = nn.Linear(gru_units, z_units)
    
    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.extract_mfcc(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous() # (batch, nfft, frame) -> (batch, frame, nfft)
        x, _ = self.gru(x)
        x = self.dense(x)
        return x[..., :-1, :]

class EngryEncoder(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.extract_fn = partial(self.extract_frames, n_fft=n_fft, hop_length=hop_length)

    def forward(self, signal):
        frames = self.extract_fn(signal)
        if frames.dim() == 2:
            frames = frames.unsqueeze(dim=0)
        frames = frames[:, :-1, :].contiguous()
        energy = torch.sqrt((frames**2).mean(dim=-1, keepdim=True))
        return energy

    @staticmethod
    def extract_frames(signal, n_fft, hop_length, center=True):
        if center:
            signal = torch.nn.functional.pad(signal, (n_fft // 2, n_fft // 2), mode='reflect')
        
        num_frames = (signal.shape[-1] - n_fft) // hop_length + 1
        indices = torch.arange(0, num_frames * hop_length, hop_length).unsqueeze(1) + torch.arange(n_fft).unsqueeze(0)
        frames = signal[:, indices].squeeze(0)
        return frames


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.zencoder = ZMFCCEncoder()
        self.engry_encoder = EngryEncoder()
            
    def forward(self, signal, loundness, f0):
        f0 = f0.unsqueeze(dim=-1)
        l = loundness.unsqueeze(dim=-1)
        engry = self.engry_encoder(signal)
        # z = self.zencoder(signal)
        return  f0, l, engry