#%%
import torch
import math
import torch.nn as nn
from ..utils_blocks import DFBlock, AttSubBlock

from ..utils import safe_divide

# force the amplitudes, harmonic distributions, and filtered noise magnitudes 
# to be non-negative by applying a sigmoid nonlinearity to network outputs.
def modified_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7): 
    return max_value * torch.sigmoid(x)**math.log(exponent) + threshold

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            self.linear_stack(in_size, hidden_size),
            self.linear_stack(hidden_size, hidden_size),
            self.linear_stack(hidden_size, hidden_size),
        )
    
    def forward(self, x):
        return self.net(x)

    def linear_stack(self, in_size, hidden_size):
        block = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU()
            )
        return block

class ConditionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv_1x1_input = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.conv_1x1_condition = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.conv_1x1_output = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
    
    def forward(self, x, timbre_emb):
        x_input = self.conv_1x1_input(x.transpose(1, 2))
        x_condition = self.conv_1x1_condition(timbre_emb.transpose(1, 2))
        
        mix = x_input + x_condition
        mix_tanh = torch.tanh(mix)
        mix_sigmoid = torch.sigmoid(mix)
        mix_output = mix_tanh * mix_sigmoid
        mix_output = self.conv_1x1_output(mix_output)
        
        output = nn.LeakyReLU(0.2)(mix_output + x.transpose(1, 2))
        return output.transpose(1, 2)

class CondMLP(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.linear_1 = self.linear_stack(in_size, hidden_size)
        self.linear_2 = self.linear_stack(hidden_size, hidden_size)
        self.linear_3 = self.linear_stack(hidden_size, hidden_size)
        self.cond_block_1 = ConditionBlock(hidden_size)
        self.cond_block_2 = ConditionBlock(hidden_size)
        self.cond_block_3 = ConditionBlock(hidden_size)
    
    def forward(self, x, timbre_emb):
        x = self.linear_1(x)
        x = self.cond_block_1(x, timbre_emb)
        x = self.linear_2(x)
        x = self.cond_block_2(x, timbre_emb)
        x = self.linear_3(x)
        x = self.cond_block_3(x, timbre_emb)
        return x

    def linear_stack(self, in_size, hidden_size):
        block = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU()
            )
        return block


class HarmonicHead(nn.Module):
    def __init__(self, in_size, n_harms):
        super().__init__()
        self.dense_harm = nn.Linear(in_size, n_harms+1)

    def forward(self, out_mlp_final):
        n_harm_amps = self.dense_harm(out_mlp_final)

        # out_dense_harmonic output -> global_amplitude(1) + n_harmonics(101) 
        n_harm_amps = modified_sigmoid(n_harm_amps)
        global_amp, n_harm_dis = n_harm_amps[..., :1], n_harm_amps[..., 1:]

        n_harm_dis_norm =  safe_divide(n_harm_dis, n_harm_dis.sum(dim=-1, keepdim=True)) 

        return n_harm_dis_norm, global_amp


class NoiseHead(nn.Module):
    def __init__(self, in_size, noise_filter_bank):
        super().__init__()
        self.dense_noise = nn.Linear(in_size, noise_filter_bank)
    
    def forward(self, out_mlp_final):
        out_dense_noise = self.dense_noise(out_mlp_final)
        noise_filter_bank = modified_sigmoid(out_dense_noise)

        return noise_filter_bank

class TimbreTransformer(nn.Module):
    def __init__(self, timbre_emb_dim, layers=2):
        super().__init__()
        self.self_att = nn.ModuleList([AttSubBlock(timbre_emb_dim, 8) for _ in range(layers)])
        self.cross_att = nn.ModuleList([AttSubBlock(timbre_emb_dim, 8) for _ in range(layers)])
    
    def forward(self, x, timbre_emb):
        for self_att, cross_att in zip(self.self_att, self.cross_att):
            x = self_att(x, x)
            x = cross_att(x, timbre_emb)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        z_units=16,
        in_extract_size=256,
        timbre_emb_size=128,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.f0_mlp = CondMLP(1, in_extract_size)
        self.l_mlp = CondMLP(1, in_extract_size)
        self.z_mlp = MLP(z_units, z_units)
        self.timbre_mlp = MLP(timbre_emb_size, timbre_emb_size)
        self.f0_self_att = AttSubBlock(in_extract_size, 8)
        self.l_self_att = AttSubBlock(in_extract_size, 8)
        self.timbre_self_att = AttSubBlock(in_extract_size, 8)
        cat_size = in_extract_size * 2 + z_units + timbre_emb_size
        self.mix_gru = nn.GRU(cat_size, timbre_emb_size, batch_first=True)
        self.timbre_transformer = TimbreTransformer(timbre_emb_size)

        final_size = timbre_emb_size + in_extract_size * 2
        self.final_mlp = MLP(final_size, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, z, timbre_emb):
        out_f0_mlp = self.f0_mlp(f0, timbre_emb)
        out_l_mlp = self.l_mlp(loudness, timbre_emb)
        out_z_mlp = self.z_mlp(z)
        out_timbre_mlp = self.timbre_mlp(timbre_emb)
        out_f0_mlp = self.f0_self_att(out_f0_mlp, out_f0_mlp)
        out_l_mlp = self.l_self_att(out_l_mlp, out_l_mlp)
        out_timbre_mlp = self.timbre_self_att(out_timbre_mlp, out_timbre_mlp)
        timbre_P = out_timbre_mlp.permute(1, 0, 2).contiguous()
        cat_input = torch.cat(
            [
                out_f0_mlp, 
                out_f0_mlp, 
                out_z_mlp,
                out_timbre_mlp.expand(out_timbre_mlp.shape[0], 250, out_timbre_mlp.shape[-1] )
                ],
            dim=-1)
        out_mix_gru, _ = self.mix_gru(cat_input, timbre_P)
        out_mix = self.timbre_transformer(out_mix_gru, out_timbre_mlp)
        cat_final = torch.cat([out_f0_mlp, out_l_mlp, out_mix], dim=-1)

        out_final_mlp = self.final_mlp(cat_final)
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_mlp)

        # noise filter part
        noise_output = self.noise_head(out_final_mlp)

        return harmonic_output, noise_output, f0