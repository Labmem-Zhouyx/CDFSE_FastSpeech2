import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import get_mask_from_lengths


class NormalEncoder(nn.Module):
    def __init__(self, in_dim=80, conv_channels=[256, 128, 64], kernel_size=3, stride=1, padding=1, dropout=0.5, out_dim=256):
        super(NormalEncoder, self).__init__()

        # convolution layers followed by batch normalization and ReLU activation
        K = len(conv_channels)
        
        # 1-D convolution layers
        filters = [in_dim] + conv_channels

        self.conv1ds = nn.ModuleList(
            [nn.Conv1d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
             for i in range(K)])

        # 1-D batch normalization (BN) layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features=conv_channels[i])
             for i in range(K)])

        # ReLU
        self.relu = nn.ReLU()

        # dropout
        self.dropout = nn.Dropout(dropout)

        self.outlayer = nn.Linear(in_features=conv_channels[-1], out_features=out_dim)

    def forward(self, x):
        # transpose to (B, embed_dim, T) for convolution, and then back
        out = x.transpose(1, 2)
        for conv, bn in zip(self.conv1ds, self.bns):
            out = conv(out)
            out = self.relu(out)
            out = bn(out)  # [B, 128, T//2^K, mel_dim//2^K], where 128 = conv_channels[-1]
            out = self.dropout(out)

        out = out.transpose(1, 2)  # [B, T//2^K, 128, mel_dim//2^K]
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1)  # [B, T//2^K, 128*mel_dim//2^K]

        out = self.outlayer(out)
        return out


class DownsampleEncoder(nn.Module):
    def __init__(self, in_dim=256, conv_channels=[256, 256, 256, 256], kernel_size=3, stride=1, padding=1, dropout=0.2, pooling_sizes=[2, 2, 2, 2], out_dim=256):
        super(DownsampleEncoder, self).__init__()

        K = len(conv_channels)
        
        # 1-D convolution layers
        filters = [in_dim] + conv_channels

        self.conv1ds = nn.ModuleList(
            [nn.Conv1d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
             for i in range(K)])

        # 1-D batch normalization (BN) layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features=conv_channels[i])
             for i in range(K)])

        self.pools = nn.ModuleList(
            [nn.AvgPool1d(kernel_size=pooling_sizes[i]) for i in range(K)]
        )

        # ReLU
        self.relu = nn.ReLU()

        # dropout
        self.dropout = nn.Dropout(dropout)



        self.local_outlayer = nn.Sequential(
            nn.Linear(in_features=conv_channels[-1],
                      out_features=out_dim),
            nn.Tanh()
        )

    def forward(self, inputs):
        out = inputs.transpose(1, 2)
        for conv, bn, pool in zip(self.conv1ds, self.bns, self.pools):
            out = conv(out)    
            out = self.relu(out)  
            out = bn(out) # [B, 128, T//2^K, mel_dim//2^K], where 128 = conv_channels[-1]
            out = self.dropout(out)
            out = pool(out)

        out = out.transpose(1, 2)  # [B, T//2^K, 128, mel_dim//2^K]
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1)  # [B, T//2^K, 128*mel_dim//2^K]

        local_output = self.local_outlayer(out)
        return local_output


class ReferenceAttention(nn.Module):
    '''
    embedded_text --- [N, seq_len, text_embedding_dim]
    mels --- [N, n_mels*r, Ty/r], r=1
    style_embed --- [N, seq_len, style_embedding_dim]
    alignments --- [N, seq_len, ref_len], Ty/r = ref_len
    '''
    def __init__(self, query_dim=256, key_dim=256, ref_attention_dim=128, ref_attention_dropout=0):
        super(ReferenceAttention, self).__init__()
        self.attn = ScaledDotProductAttention(query_dim, key_dim, ref_attention_dim, ref_attention_dropout)

    def forward(self, text_embeddings, text_lengths, key, value, ref_mels, ref_mel_lengths):

        if text_lengths == None and ref_mel_lengths == None:
            attn_mask = None
        else:
            # Get attention mask
            # 1. text mask
            text_total_length = text_embeddings.size(1)  # [N, T_x, dim]
            text_mask = get_mask_from_lengths(text_lengths, text_total_length).float().unsqueeze(-1)  # [B, seq_len, 1]
            # 2. mel mask (regularized to phoneme_scale)
            ref_mel_total_length = ref_mels.size(1)  # [N, T_y, n_mels]
            ref_mel_mask = get_mask_from_lengths(ref_mel_lengths, ref_mel_total_length).float().unsqueeze(-1)  # [B, rseq_len, 1]
            ref_mel_mask = F.interpolate(ref_mel_mask.transpose(1, 2), size=key.size(1))  # [B, 1, Ty]
            # 3. The attention mask
           
            attn_mask = 1 - torch.bmm(text_mask, ref_mel_mask)  # [N, seq_len, ref_len]
            

        # Attention
        style_embed, alignments = self.attn(text_embeddings, key, value, attn_mask)

        # Apply ReLU as the activation function to force the values of the prosody embedding to lie in [0, ∞].
        # style_embed = F.relu(style_embed)

        return style_embed, alignments


class ScaledDotProductAttention(
    nn.Module):  # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_dim, key_dim, ref_attention_dim, ref_attention_dropout):
        super().__init__()
        self.dropout = nn.Dropout(ref_attention_dropout) \
            if ref_attention_dropout > 0 else None

        self.d_q = query_dim
        self.d_k = key_dim

        self.linears = nn.ModuleList([
            LinearNorm(in_dim, ref_attention_dim, bias=False, w_init_gain='tanh') \
            for in_dim in (self.d_q, self.d_k)
        ])

        self.score_mask_value = -1e9

    def forward(self, q, k, v, mask=None):
        q, k = [linear(vector) for linear, vector in zip(self.linears, (q, k))]
        alignment = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [N, seq_len, ref_len]
      
        if mask is not None:
            alignment = alignment.masked_fill_(mask == 0, self.score_mask_value)
            
        attention_weights = F.softmax(alignment, dim=-1)
        attention_weights = self.dropout(attention_weights) \
            if self.dropout is not None else attention_weights

        attention_context = torch.bmm(attention_weights, v)  # [N, seq_len, prosody_embedding_dim]

        return attention_context, attention_weights


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
