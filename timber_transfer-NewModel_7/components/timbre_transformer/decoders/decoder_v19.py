#%%
import torch
import math
import torch.nn as nn
from collections import OrderedDict

from ..utils_blocks import DFBlock, AttSubBlock
from ..utils import safe_divide

# force the amplitudes, harmonic distributions, and filtered noise magnitudes 
# to be non-negative by applying a sigmoid nonlinearity to network outputs.
def modified_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7): 
    return max_value * torch.sigmoid(x)**math.log(exponent) + threshold



class NonIntHarmonicHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, enhance_dim=40):
        super().__init__()
        self.dense_condition = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.dense_harm = nn.Linear(input_dim, enhance_dim + 1)
        self.enhance_multiplier= nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_dim, enhance_dim)),
            ("relu1", nn.LeakyReLU(0.2)),
            ("linear2", nn.Linear(enhance_dim, enhance_dim)),
        ]))
        self.enhance_bias= nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_dim, enhance_dim)),
            ("relu1", nn.LeakyReLU(0.2)),
            ("linear2", nn.Linear(enhance_dim, enhance_dim)),
        ]))

    def _initialize(self):
        nn.init.zeros_(self.enhance_multiplier.linear2.weight.data)
        nn.init.zeros_(self.enhance_multiplier.linear2.bias.data)
        nn.init.zeros_(self.enhance_bias.linear2.weight.data)
        nn.init.zeros_(self.enhance_bias.linear2.bias.data)

    def forward(self, x) -> torch.Tensor:
        n_harm_amps = self.dense_harm(x)
        condition = self.dense_condition(x)

        weight = self.enhance_multiplier(condition)
        weight = self.sigmoid_for_enhance_multiplier(weight)
        bias = self.enhance_bias(condition)
        bias = nn.Sigmoid()(bias) 

        n_harm_amps = modified_sigmoid(n_harm_amps)
        global_amp, n_harm_dis = n_harm_amps[..., :1], n_harm_amps[..., 1:]

        n_harm_dis_norm =  safe_divide(n_harm_dis, n_harm_dis.sum(dim=-1, keepdim=True)) 

        enhance_harms_coef = weight + bias

        return n_harm_dis_norm, global_amp, enhance_harms_coef 


    def sigmoid_for_enhance_multiplier(self, x, max_value=100, threshold=1):
        return max_value * torch.sigmoid(x) + threshold
        


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
        nn.LeakyReLU(),
        nn.Linear(out_size, out_size),
        nn.LeakyReLU()
        )
    return block


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.hidden_linear = nn.Sequential(
            linear_stack(in_size, hidden_size),
            linear_stack(hidden_size, out_size),
            linear_stack(out_size, out_size),
        )
    
    def forward(self, x):
        out = self.hidden_linear(x)
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
        self.l_LR = nn.Sequential(
            linear_stack(fl_emb_dim, timbre_emb_dim),
            linear_stack(timbre_emb_dim, timbre_emb_dim),
            linear_stack(timbre_emb_dim, timbre_emb_dim),
        ) 
        self.mix_LR = linear_stack(timbre_emb_dim * 2, timbre_emb_dim)
        self.tanh_l = nn.Sequential(
            nn.Linear(timbre_emb_dim, timbre_emb_dim),
            nn.Tanh()
        )
        self.sigmoid_l = nn.Sequential(
            nn.Linear(timbre_emb_dim, timbre_emb_dim),
            nn.Sigmoid()
        )
        self.output_LR = linear_out(timbre_emb_dim, timbre_emb_dim)
    
    def forward(self, timbre_emb, l_emb): 
        l = self.l_LR(l_emb)
        mix_cat = torch.cat([l, timbre_emb.expand_as(l)], dim=-1)
        mix = self.mix_LR(mix_cat)
        mix_tanh = self.tanh_l(mix)
        mix_sigmoid = self.sigmoid_l(mix)
        ehance = mix_tanh * mix_sigmoid * l
        out = self.output_LR(timbre_emb + ehance)
        return out
  
class TimbreAffineBlcok(nn.Module):
    def __init__(self, timbre_emb, fl_emb):
        super().__init__()
        self.l_LR = nn.Sequential(
            linear_stack(fl_emb, timbre_emb),
            linear_stack(timbre_emb, timbre_emb),
            linear_stack(timbre_emb, timbre_emb),
        )
        self.l_DF_1 = DFBlock(timbre_emb, timbre_emb)
        self.l_DF_2 = DFBlock(timbre_emb, timbre_emb)
        self.output_LR = linear_out(timbre_emb, timbre_emb)
    
    def forward(self, timbre_emb, l_emb):
        l = self.l_LR(l_emb)
        l_affine = self.l_DF_1(timbre_emb, l)
        l_affine = self.l_DF_2(timbre_emb, l_affine)
        out = self.output_LR(l_affine)
        return out

class TimbreZGenerator(nn.Module):
    def __init__(self, timbre_emb_dim, fl_emb):
        super().__init__()
        self.fusion_block = TimbreFusionBlock(timbre_emb_dim, fl_emb)
        self.fusion_block_2 = TimbreFusionBlock(timbre_emb_dim, fl_emb)
        self.affine_block = TimbreAffineBlcok(timbre_emb_dim, fl_emb)
        self.mix_weight = nn.Parameter(torch.rand(256, 3))
    
    def forward(self, timbre_emb, l_emb):
        fusion = self.fusion_block(timbre_emb, l_emb)
        fusion = self.fusion_block_2(fusion, l_emb)
        affine = self.affine_block(timbre_emb, l_emb)
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
        in_extract_size=512,
        timbre_emb_size=286,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.f0_mlp = MLP(1, in_extract_size, timbre_emb_size)
        self.f0_linear = linear_stack(timbre_emb_size, timbre_emb_size) 
        self.l_mlp = MLP(1, in_extract_size, timbre_emb_size)
        self.l_linear = linear_stack(timbre_emb_size, timbre_emb_size)
        self.t_mlp = nn.Sequential(
            linear_stack(timbre_emb_size, timbre_emb_size),
            linear_stack(timbre_emb_size, timbre_emb_size),
        )
        self.timbre_z_generator = TimbreZGenerator(timbre_emb_size, 1)
        cat_size = timbre_emb_size * 3
        self.mix_gru = nn.GRU(cat_size, timbre_emb_size, batch_first=True)
        self.timbre_transformer = TimbreTransformer(timbre_emb_size)

        final_size = timbre_emb_size * 3
        self.final_mlp = MLP(final_size, final_embedding_size, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)

        self.enhance_harmonic_head = NonIntHarmonicHead()
        
    def forward(self, f0, loudness, energy, timbre_emb):
        out_f0_mlp = self.f0_mlp(f0)
        out_f0_linear = self.f0_linear(out_f0_mlp) + out_f0_mlp
        out_l_mlp = self.l_mlp(loudness)
        out_l_linear = self.l_linear(out_l_mlp) + out_l_mlp
        out_t_mlp = self.t_mlp(timbre_emb)
        timbre_z = self.timbre_z_generator(out_t_mlp, energy)
        cat_input = torch.cat(
            [
                out_f0_linear,
                out_l_linear,
                timbre_z,
                ],
            dim=-1)
        out_mix_gru, _ = self.mix_gru(cat_input)
        out_mix = self.timbre_transformer(out_mix_gru, timbre_emb)
        cat_final = torch.cat([out_f0_mlp, out_l_mlp, out_mix], dim=-1)

        out_final_mlp = self.final_mlp(cat_final)
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_mlp)

        # noise filter part
        noise_output = self.noise_head(out_final_mlp)

        enhance_harmonic_output = self.enhance_harmonic_head(out_final_mlp)

        return harmonic_output, noise_output, f0, enhance_harmonic_output
# %%
