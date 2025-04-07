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


def linear_stack(in_size, hidden_size):
    block = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.LeakyReLU()
        )
    return block

def linear_out(in_size, out_size):
    block = nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.LeakyReLU()
        )
    return block


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_linear = nn.Sequential(
            linear_stack(in_size, hidden_size),
            linear_stack(hidden_size, hidden_size),
        )
        self.out_linear = linear_stack(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.hidden_linear(x)
        out = self.out_linear(x)
        out = out + x
        return out

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

class TimbreFusionBlock(nn.Module):
    def __init__(self, timbre_emb_dim, fl_emb_dim):
        super().__init__()
        self.f_LR = nn.Sequential(
            linear_stack(fl_emb_dim, timbre_emb_dim),
            linear_stack(timbre_emb_dim, timbre_emb_dim),
        )
        self.l_LR = nn.Sequential(
            linear_stack(fl_emb_dim, timbre_emb_dim),
            linear_stack(timbre_emb_dim, timbre_emb_dim),
        ) 
        self.fl_LR = linear_stack(timbre_emb_dim * 2, timbre_emb_dim)
        self.mix_LR = linear_stack(timbre_emb_dim * 3, timbre_emb_dim)
        self.tanh_l = nn.Sequential(
            nn.Linear(timbre_emb_dim, timbre_emb_dim),
            nn.Tanh()
        )
        self.sigmoid_l = nn.Sequential(
            nn.Linear(timbre_emb_dim, timbre_emb_dim),
            nn.Sigmoid()
        )
        self.output_LR = linear_out(timbre_emb_dim, timbre_emb_dim)
    
    def forward(self, timbre_emb, f_emb, l_emb): 
        f = self.f_LR(f_emb)
        l = self.l_LR(l_emb)
        fl_cat = torch.cat([f, l], dim=-1)  
        mix_cat = torch.cat([fl_cat, timbre_emb.expand_as(f)], dim=-1)
        fl = self.fl_LR(fl_cat)
        mix = self.mix_LR(mix_cat)
        mix_tanh = self.tanh_l(mix)
        mix_sigmoid = self.sigmoid_l(mix)
        ehance = mix_tanh * mix_sigmoid * fl
        out = self.output_LR(timbre_emb + ehance)
        return out
  
class TimbreAffineBlcok(nn.Module):
    def __init__(self, timbre_emb, fl_emb):
        super().__init__()
        self.f_LR = nn.Sequential(
            linear_stack(fl_emb, timbre_emb),
            linear_stack(timbre_emb, timbre_emb),
        )
        self.l_LR = nn.Sequential(
            linear_stack(fl_emb, timbre_emb),
            linear_stack(timbre_emb, timbre_emb),
        )
        self.f_DF = DFBlock(timbre_emb, timbre_emb)
        self.l_DF = DFBlock(timbre_emb, timbre_emb)
        self.output_LR = linear_out(timbre_emb * 2, timbre_emb)
    
    def forward(self, timbre_emb, f_emb, l_emb):
        f = self.f_LR(f_emb)
        l = self.l_LR(l_emb)
        f_affine = self.f_DF(timbre_emb, f)
        l_affine = self.l_DF(timbre_emb, l)
        out_cat = torch.cat([f_affine, l_affine], dim=-1)
        out = self.output_LR(out_cat)
        return out

class TimbreZGenerator(nn.Module):
    def __init__(self, timbre_emb_dim, fl_emb):
        super().__init__()
        self.fusion_block = TimbreFusionBlock(timbre_emb_dim, fl_emb)
        self.fusion_block_2 = TimbreFusionBlock(timbre_emb_dim, fl_emb)
        self.affine_block = TimbreAffineBlcok(timbre_emb_dim, fl_emb)
        self.mix_weight = nn.Parameter(torch.rand(256, 3))
    
    def forward(self, timbre_emb, f_emb, l_emb):
        fusion = self.fusion_block(timbre_emb, f_emb, l_emb)
        fusion = self.fusion_block_2(fusion, f_emb, l_emb)
        affine = self.affine_block(timbre_emb, f_emb, l_emb)
        mix = torch.cat([
            fusion.unsqueeze(-1),
            affine.unsqueeze(-1),
            timbre_emb.expand_as(fusion).unsqueeze(-1)
            ],
            dim=-1)
        mix_out = mix * self.mix_weight
        out = mix_out.sum(dim=-1)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        in_extract_size=256,
        timbre_emb_size=128,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.f0_mlp = MLP(1, in_extract_size)
        self.l_mlp = MLP(1, in_extract_size)
        self.timbre_z_generator = TimbreZGenerator(timbre_emb_size, 1)
        cat_size = in_extract_size * 2 + timbre_emb_size
        self.mix_gru = nn.GRU(cat_size, timbre_emb_size, batch_first=True)
        self.timbre_transformer = TimbreTransformer(timbre_emb_size)

        final_size = timbre_emb_size + in_extract_size * 2
        self.final_mlp = MLP(final_size, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, engry, timbre_emb):
        out_f0_mlp = self.f0_mlp(f0)
        out_l_mlp = self.l_mlp(loudness)
        timbre_z = self.timbre_z_generator(timbre_emb, f0, engry)
        timbre_P = timbre_emb.permute(1, 0, 2).contiguous()
        cat_input = torch.cat(
            [
                out_f0_mlp, 
                out_f0_mlp, 
                timbre_z,
                ],
            dim=-1)
        out_mix_gru, _ = self.mix_gru(cat_input, timbre_P)
        out_mix = self.timbre_transformer(out_mix_gru, timbre_emb)
        cat_final = torch.cat([out_f0_mlp, out_l_mlp, out_mix], dim=-1)

        out_final_mlp = self.final_mlp(cat_final)
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_mlp)

        # noise filter part
        noise_output = self.noise_head(out_final_mlp)

        return harmonic_output, noise_output, f0
# %%
