#%%
import torch
import math
import torch.nn as nn
from ..utils_blocks import DFBlock, TCUB, AttSubBlock, GateFusionBlock

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

class InputAttBlock(nn.Module):
    def __init__(self, in_size=1, hidden_size=32, out_size=64):
        super().__init__()
        if hidden_size % 8 == 0:
            num_heads = 8
        else:
            num_heads = 1
        self.input_linear = nn.Linear(in_size, hidden_size)
        self.first_att_block = AttSubBlock(hidden_size, num_heads)
        self.out_linear = nn.Linear(hidden_size, out_size)
        self.out_att_block = AttSubBlock(out_size, num_heads)
    
    def forward(self, x):
        x = self.input_linear(x)
        x = self.first_att_block(x, x)
        x = self.out_linear(x)
        x = self.out_att_block(x, x)
        return x

class AmpStack(nn.Module):
    def __init__(self, emb_dim=8):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(emb_dim, 1),
            )
    
    def forward(self, amp, loudness):
        amp_mix = torch.cat([amp, loudness], dim=-1)
        amp_att = self.stack(amp_mix)
        return modified_sigmoid(amp + amp_att)


class HarmonicHead(nn.Module):
    def __init__(self, in_size, timbre_emb_size, n_harms):
        super().__init__()
        self.dense_harm = nn.Linear(in_size, n_harms+1)
        self.dfblock1 = DFBlock(n_harms, timbre_emb_size, affine_dim=n_harms, out_layer_mlp=True)
        self.dfblock2 = DFBlock(n_harms, timbre_emb_size, affine_dim=n_harms, out_layer_mlp=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, out_mlp_final, timbre_emb):
        n_harm_amps = self.dense_harm(out_mlp_final)

        # out_dense_harmonic output -> global_amplitude(1) + n_harmonics(101) 
        n_harm_amps = modified_sigmoid(n_harm_amps)

        global_amp, n_harm_dis = n_harm_amps[..., :1], n_harm_amps[..., 1:]

        # harmonic distribution part
        df_out = self.dfblock1(n_harm_dis, timbre_emb)
        df_out = self.dfblock2(df_out, timbre_emb)
        df_out = modified_sigmoid(df_out)
        n_harm_dis = n_harm_dis * df_out

        # global amplitude part
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


class Decoder(nn.Module):
    def __init__(
        self,
        in_extract_size=64,
        timbre_emb_size=128,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        hidden_size = in_extract_size // 2
        self.input_f0 = InputAttBlock(in_size=1, hidden_size=hidden_size, out_size=in_extract_size)
        self.input_loudness = InputAttBlock(in_size=1, hidden_size=hidden_size, out_size=in_extract_size)
        gru_in_size = in_extract_size 
        self.gru_f0 = nn.GRU(gru_in_size, gru_in_size, num_layers=3, batch_first=True)
        self.gru_loudness = nn.GRU(gru_in_size, gru_in_size, num_layers=3, batch_first=True)

        in_size = in_extract_size * 2
        self.condition_proj_1 = nn.Linear(timbre_emb_size, in_size)
        self.condition_proj_2 = nn.Linear(timbre_emb_size, in_size * 2)
        self.condition_proj_3 = nn.Linear(timbre_emb_size, in_size * 4)
        self.tcub_1 = TCUB(in_size)
        self.tcub_2 = TCUB(in_size * 2)
        self.self_att_1 = AttSubBlock(in_size)
        self.self_att_2 = AttSubBlock(in_size * 2)
        self.self_att_3 = AttSubBlock(in_size * 4)
        self.cross_att_1 = AttSubBlock(in_size * 2)
        self.cross_att_2 = AttSubBlock(in_size * 4)

        self.final_self_att = AttSubBlock(in_size * 4 + in_size) 
        self.final_self_att_proj = nn.Linear(in_size * 4 + in_size, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, timbre_emb_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, timbre_emb):
        
        out_input_f0 = self.input_f0(f0)
        out_input_loudness = self.input_loudness(loudness)
        out_gru_f0, _ = self.gru_f0(out_input_f0)
        out_gru_loudness, _ = self.gru_loudness(out_input_loudness)
        out_cat_mlp = torch.cat([out_gru_f0, out_gru_loudness], dim=-1)
        out_cat_mlp = self.self_att_1(out_cat_mlp, out_cat_mlp)

        timbre_emb_1 = self.condition_proj_1(timbre_emb)
        timbre_emb_2 = self.condition_proj_2(timbre_emb)
        timbre_emb_3 = self.condition_proj_3(timbre_emb)
        out_timbre_fusion_1 = self.tcub_1(out_cat_mlp, timbre_emb_1)
        out_timbre_fusion_1 = self.self_att_2(out_timbre_fusion_1, out_timbre_fusion_1)
        out_timbre_fusion_1 = self.cross_att_1(out_timbre_fusion_1, timbre_emb_2)
        out_timbre_fusion_2 = self.tcub_2(out_timbre_fusion_1, timbre_emb_2)
        out_timbre_fusion_2 = self.self_att_3(out_timbre_fusion_2, out_timbre_fusion_2)
        out_timbre_fusion_2 = self.cross_att_2(out_timbre_fusion_2, timbre_emb_3)

        out_cat_f0_loudness = torch.cat([out_timbre_fusion_2, out_gru_f0, out_gru_loudness], dim=-1)
        out_final_self_att = self.final_self_att(out_cat_f0_loudness, out_cat_f0_loudness)
        out_final_self_att = self.final_self_att_proj(out_final_self_att)
        
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_self_att, timbre_emb)

        # noise filter part
        noise_output = self.noise_head(out_final_self_att)

        return harmonic_output, noise_output

