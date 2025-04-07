import torch
import torch.nn as nn

class TCUB(nn.Module):
    def __init__(self, in_ch, num_heads=8):
        super().__init__()
        out_ch = in_ch * 2
        self.conv_1x1_input = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.conv_1x1_condition = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        
        # Using PyTorch's MultiheadAttention
        self.attention_block = nn.MultiheadAttention(embed_dim=in_ch, num_heads=num_heads, batch_first=True)
        self.att_norm = nn.LayerNorm(in_ch)
        self.linear_after_att = nn.Linear(in_ch, out_ch)  # to transform the output to desired dimension
        self.linear_after_att = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.LeakyReLU(0.2),
            nn.Linear(out_ch, out_ch),
            )  # to transform the output to desired dimension
        
        self.conv_1x1_output = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, condition):
        x_input = self.conv_1x1_input(x.transpose(1, 2))
        x_condition = self.conv_1x1_condition(condition.transpose(1, 2))
        
        # Applying attention
        attn_output, _ = self.attention_block(x, condition, condition)
        attn_output = self.att_norm(x + attn_output)
        x_attention = self.linear_after_att(attn_output)
        
        mix = torch.cat([x_input, x_condition.expand_as(x_input)], dim=1)
        mix_tanh = torch.tanh(mix)
        mix_sigmoid = torch.sigmoid(mix)
        mix_output = mix_tanh * mix_sigmoid
        mix_output = self.conv_1x1_output(mix_output)
        
        output = nn.LeakyReLU(0.2)(x_attention + mix_output.transpose(1, 2))
        return output 
    
class GateFusionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv_1x1_input = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.conv_1x1_condition = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        
        self.conv_1x1_output = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)

    def forward(self, x, condition):
        x_input = self.conv_1x1_input(x.transpose(1, 2))
        x_condition = self.conv_1x1_condition(condition.transpose(1, 2))
        
        mix = x_input + x_condition
        mix_tanh = torch.tanh(mix)
        mix_simoid = torch.sigmoid(mix)
        mix_output = mix_tanh * mix_simoid
        mix_output = self.conv_1x1_output(mix_output)
        
        output = nn.LeakyReLU(0.2)(mix_output + x.transpose(1, 2))
        return output.transpose(1, 2)

class AttSubBlock(nn.Module):
    def __init__(self, in_ch, num_heads=8):
        super().__init__()
        self.self_att = nn.MultiheadAttention(embed_dim=in_ch, num_heads=num_heads, batch_first=True)
        self.self_att_norm = nn.LayerNorm(in_ch)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_ch * 2, in_ch),
            nn.LeakyReLU(0.2)
        )
        self.mlp_norm = nn.LayerNorm(in_ch)
    
    def forward(self, q, kv):
        att_out, _ = self.self_att(q, kv, kv)
        att_out = self.self_att_norm(q + att_out)
        fc_out = self.mlp(att_out)
        out = self.mlp_norm(att_out + fc_out)
        return out
    
class TimbreAttFusionBlock(nn.Module):
    def __init__(self, in_emb, timbre_emb, num_heads=8) -> None:
        super().__init__()
        out_emb = in_emb * 2
        self.input_fc = nn.Linear(in_emb, out_emb)
        self.input_self_att = AttSubBlock(out_emb, num_heads)
        self.timbre_fc = nn.Linear(timbre_emb, out_emb)
        self.timbre_self_att = AttSubBlock(out_emb, num_heads)
        self.timbre_fusion_att = AttSubBlock(out_emb, num_heads)
    
    def forward(self, x, timbre_emb):
        x = self.input_fc(x)
        x = self.input_self_att(x, x)
        timbre_emb = self.timbre_fc(timbre_emb)
        timbre_emb = self.timbre_self_att(timbre_emb, timbre_emb)
        timbre_fusion_emb = self.timbre_fusion_att(x, timbre_emb)
        return timbre_fusion_emb