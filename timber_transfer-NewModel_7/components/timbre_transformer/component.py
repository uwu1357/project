import torch
import torch.nn as nn
import torch.fft as fft
import math


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]
    
    return output


class HarmonicOscillator(nn.Module):
    def __init__(
        self,
        sample_rate = 16000,
        hop_length = 256,
        n_harms = 101,
        is_smooth = False,
        ):
        super().__init__()
        self.sr = sample_rate
        self.hop_length = hop_length
        self.n_harms = n_harms
        self.is_smooth = is_smooth
            
    # FIXME: check smooth_envelop function is placed correctly or not
    def forward(self, harmonic_head_output, f0):
        # harmonic_head_output -> batch, frame, harm_head_output
        n_harm_dis, global_amp = harmonic_head_output[0], harmonic_head_output[1]

        n_harm_dis = self.remove_above_nyquist(n_harm_dis, f0, self.sr)
        n_harm_dis = self.upsample(n_harm_dis, self.hop_length)
        global_amp = self.upsample(global_amp, self.hop_length)
        if self.is_smooth:
            global_amp = self.smooth_envelop(global_amp, self.hop_length, self.hop_length * 2)
        harm_amp_dis = n_harm_dis * global_amp
        f = self.upsample(f0, self.hop_length)
        harmonic = self.harmonic_synth(f, harm_amp_dis, self.sr)
        return harmonic
        
    @staticmethod
    def remove_above_nyquist(n_harm_dis, pitch, sample_rate):
        n_harm = n_harm_dis.shape[-1]
        pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
        n_harm_dis = torch.where(
            torch.ge(pitches, sample_rate / 2.0),
            torch.zeros_like(pitches), n_harm_dis
            )
        return n_harm_dis
    
    def upsample(self, x, factor, mode="linear"):
        if mode == "linear":
            return self.upsample_linear(x, factor)
        elif mode == "hann":
            return self.upsample_with_hann_window(x, factor)

    # paper says bilinear ?
    @staticmethod
    def upsample_linear(signal, factor):
        # signal -> batch, frame, channel 
        signal = signal.permute(0, 2, 1)
        signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor, mode="linear")
        return signal.permute(0, 2, 1)

    @staticmethod
    def upsample_with_hann_window(x, factor: int):
        batch, frame, channel = x.shape
        x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)
        
        window = torch.hann_window(
            factor * 2,
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, 1, -1)
        y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
        y[..., ::factor] = x
        y[..., -1:] = x[..., -1:]
        y = torch.nn.functional.pad(y, [factor, factor])
        y = torch.nn.functional.conv1d(y, window)[..., :-1]

        y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)
        return y

        
    @staticmethod
    def harmonic_synth(pitch, amplitudes, sampling_rate):
        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
        signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
        return signal

    # smooth envelops from paper
    @staticmethod
    def smooth_envelop(x, hop_size=64, win_size=128):
        # input harm_amp [batch, frame, 1]
        win = torch.hamming_window(win_size).to(x.device)
        x = nn.functional.pad(x.squeeze(dim=-1), (win_size // 2, win_size // 2))
        size_with_padding = x.size(-1)
        x = x.unfold(-1, win_size, hop_size)
        x = x * win
        x = x.permute(0, 2, 1).contiguous()
        fold = nn.Fold(output_size=(1, size_with_padding), kernel_size=(1, win_size), stride=(1, hop_size))
        x = fold(x)
        x = x.view(x.size(0), -1, 1).contiguous()
        x = x[:, hop_size+1: -hop_size+1, :] # remove the padding
        return x


class NoiseFilter(nn.Module):
    def __init__(
        self,
        hop_length=256,
        ):
        super().__init__()
        self.hop_length = hop_length
        self.fft_convolve = fft_convolve
    
    def forward(self, filter_bank):

        impulse = self.amp_to_impulse_response(filter_bank, self.hop_length)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.hop_length,
        ).to(impulse) * 2 - 1
        noise = self.fft_convolve(noise, impulse).contiguous()

        noise = noise.reshape(noise.shape[0], -1 , 1)

        return noise

    @staticmethod
    def amp_to_impulse_response(amp, target_size):
        amp = torch.stack([amp, torch.zeros_like(amp)], -1)
        amp = torch.view_as_complex(amp)
        amp = fft.irfft(amp)
        
        filter_size = amp.shape[-1]

        amp = torch.roll(amp, filter_size // 2, -1)
        win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

        amp = amp * win
        
        amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
        amp = torch.roll(amp, -filter_size // 2, -1)

        return amp
    

# uncheck     
class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)
        self.fft_convolve = fft_convolve

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = self.fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class EnhanceHarmonicOscillator(nn.Module):
    def __init__(
        self,
        sample_rate = 16000,
        hop_length = 256,
        is_smooth = False,
        ):
        super().__init__()
        self.sr = sample_rate
        self.hop_length = hop_length
        self.is_smooth = is_smooth
            
    # FIXME: check smooth_envelop function is placed correctly or not
    def forward(self, harmonic_head_output, f0):
        # harmonic_head_output -> batch, frame, harm_head_output
        n_harm_dis, global_amp = harmonic_head_output[0], harmonic_head_output[1]
        enhance_harmonic_coef = harmonic_head_output[2]

        n_harm_dis = self.remove_above_nyquist(n_harm_dis, f0, enhance_harmonic_coef, self.sr)
        n_harm_dis = self.upsample(n_harm_dis, self.hop_length)
        global_amp = self.upsample(global_amp, self.hop_length)
        if self.is_smooth:
            global_amp = self.smooth_envelop(global_amp, self.hop_length, self.hop_length * 2)
        harm_amp_dis = n_harm_dis * global_amp
        f = self.upsample(f0, self.hop_length)
        harmonic = self.harmonic_synth(f, harm_amp_dis, self.sr)
        return harmonic
        
    @staticmethod
    def remove_above_nyquist(n_harm_dis, pitch, enhance_harmonic_coef, sample_rate):
        n_harm = n_harm_dis.shape[-1]
        pitches = pitch  * enhance_harmonic_coef
        n_harm_dis = torch.where(
            torch.ge(pitches, sample_rate / 2.0),
            torch.zeros_like(pitches), n_harm_dis
            )
        return n_harm_dis
    
    def upsample(self, x, factor, mode="linear"):
        if mode == "linear":
            return self.upsample_linear(x, factor)
        elif mode == "hann":
            return self.upsample_with_hann_window(x, factor)

    # paper says bilinear ?
    @staticmethod
    def upsample_linear(signal, factor):
        # signal -> batch, frame, channel 
        signal = signal.permute(0, 2, 1)
        signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor, mode="linear")
        return signal.permute(0, 2, 1)

    @staticmethod
    def upsample_with_hann_window(x, factor: int):
        batch, frame, channel = x.shape
        x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)
        
        window = torch.hann_window(
            factor * 2,
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, 1, -1)
        y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
        y[..., ::factor] = x
        y[..., -1:] = x[..., -1:]
        y = torch.nn.functional.pad(y, [factor, factor])
        y = torch.nn.functional.conv1d(y, window)[..., :-1]

        y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)
        return y

        
    @staticmethod
    def harmonic_synth(pitch, amplitudes, sampling_rate):
        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
        signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
        return signal

    # smooth envelops from paper
    @staticmethod
    def smooth_envelop(x, hop_size=64, win_size=128):
        # input harm_amp [batch, frame, 1]
        win = torch.hamming_window(win_size).to(x.device)
        x = nn.functional.pad(x.squeeze(dim=-1), (win_size // 2, win_size // 2))
        size_with_padding = x.size(-1)
        x = x.unfold(-1, win_size, hop_size)
        x = x * win
        x = x.permute(0, 2, 1).contiguous()
        fold = nn.Fold(output_size=(1, size_with_padding), kernel_size=(1, win_size), stride=(1, hop_size))
        x = fold(x)
        x = x.view(x.size(0), -1, 1).contiguous()
        x = x[:, hop_size+1: -hop_size+1, :] # remove the padding
        return x