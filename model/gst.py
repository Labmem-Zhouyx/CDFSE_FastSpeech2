""" adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py """

import torch
from torch import nn
import torch.nn.functional as F


class GST(nn.Module):
    """
    GlobalStyleToken (GST)
    GST is described in:
        Y. Wang, D. Stanton, Y. Zhang, R.J. Shkerry-Ryan, E. Battenberg, J. Shor, Y. Xiao, F. Ren, Y. Jia, R.A. Saurous,
        "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis,"
        in Proceedings of the 35th International Conference on Machine Learning (PMLR), 80:5180-5189, 2018.
        https://arxiv.org/abs/1803.09017
    See:
        https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
        https://github.com/NVIDIA/mellotron/blob/master/modules.py
    """

    def __init__(self, mel_dim=80, gru_units=128,
                 conv_channels=[32, 32, 64, 64, 128, 128], kernel_size=3, stride=2, padding=1,
                 num_tokens=10, token_embed_dim=256, num_heads=8):
        super().__init__()

        self.encoder = ReferenceEncoder(mel_dim, gru_units, conv_channels, kernel_size, stride, padding)
        self.stl = StyleTokenLayer(gru_units, num_tokens, token_embed_dim, num_heads)

    def forward(self, inputs, input_lengths=None):
        """
        input:
            inputs --- [B, T, mel_dim]
            input_lengths --- [B]
        output:
            style_embed --- [B, 1, token_embed_dim]
        """

        ref_embed = self.encoder(inputs, input_lengths=input_lengths)  # [B, gru_units]
        style_embed = self.stl(ref_embed.unsqueeze(1))  # [B, 1, gru_units] -> [B, 1, token_embed_dim]

        return style_embed


class ReferenceEncoder(nn.Module):
    """
    ReferenceEncoder
        - 6 2-D convolutional layers with 3*3 kernel, 2*2 stride, batch norm (BN), ReLU
        - a single-layer unidirectional GRU with 128-unit
    """

    def __init__(self, in_dim=80, gru_units=128, conv_channels=[32, 32, 64, 64, 128, 128], kernel_size=3, stride=2, padding=1):
        super().__init__()

        K = len(conv_channels)

        # 2-D convolution layers
        filters = [1] + conv_channels
        self.conv2ds = nn.ModuleList(
            [nn.Conv2d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
             for i in range(K)])

        # 2-D batch normalization (BN) layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=conv_channels[i])
             for i in range(K)])

        # ReLU
        self.relu = nn.ReLU()

        # GRU
        out_channels = self.calculate_channels(in_dim, kernel_size, stride, padding, K)
        self.gru = nn.GRU(input_size=conv_channels[-1] * out_channels,
                          hidden_size=gru_units,
                          batch_first=True)

    def forward(self, inputs, input_lengths=None):
        """
        input:
            inputs --- [B, T, mel_dim] mels
            input_lengths --- [B] lengths of the mels
        output:
            out --- [B, gru_units]
        """

        out = inputs.unsqueeze(1)  # [B, 1, T, mel_dim]
        for conv, bn in zip(self.conv2ds, self.bns):
            out = conv(out)
            out = bn(out)
            out = self.relu(out)   # [B, 128, T//2^K, mel_dim//2^K], where 128 = conv_channels[-1]

        out = out.transpose(1, 2)  # [B, T//2^K, 128, mel_dim//2^K]
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1)  # [B, T//2^K, 128*mel_dim//2^K]

        # get precise last step by excluding paddings
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.conv2ds))
            input_lengths = input_lengths.cpu().numpy().astype(int)
            out = nn.utils.rnn.pack_padded_sequence(out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)  # out --- [1, B, gru_units]

        return out.squeeze(0)  # [B, gru_units]

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class StyleTokenLayer(nn.Module):
    """
    StyleTokenLayer (STL)
        - A bank of style token embeddings
        - An attention module
    """

    def __init__(self, query_dim, num_tokens=10, token_embed_dim=256, num_heads=8):
        super(StyleTokenLayer, self).__init__()

        # style token embeddings
        self.embeddings = nn.Parameter(torch.FloatTensor(num_tokens, token_embed_dim // num_heads))
        nn.init.normal_(self.embeddings, mean=0, std=0.5)

        # multi-head attention
        d_q = query_dim
        d_k = token_embed_dim // num_heads
        self.attention = MultiHeadAttention(d_q, d_k, d_k, token_embed_dim, num_heads)

    def forward(self, inputs):
        """
        input:
            inputs --- [B, 1, query_dim]
        output:
            style_embed --- [B, 1, token_embed_dim]
        """

        B = inputs.size(0)
        query = inputs  # [B, 1, query_dim]
        keys = torch.tanh(self.embeddings).unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, token_embed_dim // num_heads]
        style_embed = self.attention(query, keys, keys)  # [B, 1, token_embed_dim]

        return style_embed

    def from_token(self, token_scores):
        """
        Get style embedding by specifying token_scores
        input:
            token_scores --- [B, 1, num_tokens]
        output:
            style_embed --- [B, 1, token_embed_dim]
        """

        B = token_scores.size(0)
        tokens = torch.tanh(self.embeddings).unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, token_embed_dim // num_heads]
        tokens = self.attention.W_value(tokens)  # [B, num_tokens, token_embed_dim]
        style_embed = torch.matmul(token_scores, tokens)  # [B, 1, token_embed_dim]

        return style_embed


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention
    """

    def __init__(self, query_dim, key_dim, val_dim, num_units, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.split_size = num_units // num_heads
        self.scale_factor = key_dim ** 0.5

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=val_dim, out_features=num_units, bias=False)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, query, key, value):
        """
        input:
            query --- [B, T_q, query_dim]
            key   --- [B, T_k, key_dim]
            value --- [B, T_k, val_dim]
        output:
            out --- [B, T_q, num_units]
        """

        querys = self.W_query(query)  # [B, T_q, num_units]
        keys   = self.W_key(key)      # [B, T_k, num_units]
        values = self.W_value(value)  # [B, T_k, num_units]

        querys = torch.stack(torch.split(querys, self.split_size, dim=2), dim=0)  # [h, B, T_q, num_units/h]
        keys   = torch.stack(torch.split(keys,   self.split_size, dim=2), dim=0)  # [h, B, T_k, num_units/h]
        values = torch.stack(torch.split(values, self.split_size, dim=2), dim=0)  # [h, B, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3)) / self.scale_factor  # [h, B, T_q, T_k]
        scores = F.gumbel_softmax(scores, hard=False, dim=-1)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, B, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [B, T_q, num_units]

        return out
