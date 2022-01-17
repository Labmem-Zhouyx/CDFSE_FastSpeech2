import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet, MelEncoder
from .modules import AdversarialClassifier, VarianceAdaptor
from .gst import StyleTokenLayer
from .reference import NormalEncoder, DownsampleEncoder, ReferenceAttention
from utils.tools import get_mask_from_lengths
from text.symbols import symbols

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        
        
        # Frame-level Encoder for Reference Mel
        self.frame_encoder = NormalEncoder(
            conv_channels=model_config["frame_encoder"]["conv_filters"],
            kernel_size=model_config["frame_encoder"]["kernel_size"],
            stride=model_config["frame_encoder"]["stride"],
            padding=model_config["frame_encoder"]["padding"],
            dropout=model_config["frame_encoder"]["dropout"],
            out_dim=model_config["frame_encoder"]["out_dim"],
        )
        self.content_encoder = MelEncoder(model_config)

        self.phoneme_classifier = AdversarialClassifier(
            in_dim=model_config["frame_encoder"]["out_dim"],
            out_dim=len(symbols),
            hidden_dims=model_config["classifier"]["cls_hidden"]
        )
        self.content_embedding = nn.Embedding(
            len(symbols), model_config["frame_encoder"]["out_dim"]
        )
        
        self.ds_content_encoder = DownsampleEncoder(
            in_dim=model_config["frame_encoder"]["out_dim"],
            conv_channels=model_config["downsample_encoder"]["conv_filters"],
            kernel_size=model_config["downsample_encoder"]["kernel_size"],
            stride=model_config["downsample_encoder"]["stride"],
            padding=model_config["downsample_encoder"]["padding"],
            dropout=model_config["downsample_encoder"]["dropout"],
            pooling_sizes=model_config["downsample_encoder"]["pooling_sizes"],
            out_dim=model_config["downsample_encoder"]["out_dim"],
        )

        self.ds_speaker_encoder = DownsampleEncoder(
            in_dim=model_config["frame_encoder"]["out_dim"],
            conv_channels=model_config["downsample_encoder"]["conv_filters"],
            kernel_size=model_config["downsample_encoder"]["kernel_size"],
            stride=model_config["downsample_encoder"]["stride"],
            padding=model_config["downsample_encoder"]["padding"],
            dropout=model_config["downsample_encoder"]["dropout"],
            pooling_sizes=model_config["downsample_encoder"]["pooling_sizes"],
            out_dim=model_config["downsample_encoder"]["out_dim"],
        )
        
        #BPT
        self.use_BPT = model_config["use_BPT"]
        if self.use_BPT == True:
            self.bpt = StyleTokenLayer(
                query_dim=model_config["content_encoder"]["out_dim"],
                num_tokens=model_config["bpt"]["num_tokens"],
                token_embed_dim=model_config["bpt"]["token_embed_dim"],
                num_heads=model_config["bpt"]["num_heads"],
            )

        # Reference Attention
        self.ref_atten = ReferenceAttention(
            query_dim=model_config["transformer"]["encoder_hidden"], 
            key_dim=model_config["reference_attention"]["key_dim"],
            ref_attention_dim=model_config["reference_attention"]["attention_dim"], 
            ref_attention_dropout=model_config["reference_attention"]["attention_dropout"],
        )



    def forward(
        self,
        texts,
        src_lens,
        max_src_len,
        ref_mels,
        ref_mel_lens,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        ref_linguistics=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        ref_mel_masks = get_mask_from_lengths(ref_mel_lens, max(ref_mel_lens))

        output = self.encoder(texts, src_masks)

        frame_feature = self.frame_encoder(ref_mels)
        content_feature = self.content_encoder(frame_feature, ref_mel_masks)
        ref_content_predict = self.phoneme_classifier(content_feature, is_reversal=False)
        
        if ref_linguistics is not None:
            ref_content_embed = self.content_embedding(ref_linguistics)
        else:
            ref_content_embed = self.content_embedding(ref_content_predict.argmax(-1))

        if self.use_BPT == True:
            content_feature = self.bpt(content_feature)

        ref_local_content_emb = self.ds_content_encoder(content_feature)
        ref_local_speaker_emb = self.ds_speaker_encoder(frame_feature)
        
        #if self.use_BPT == True:
        #    ref_local_content_emb = self.bpt(ref_local_content_emb)
        # print(ref_local_speaker_emb)
        local_spk_emb, ref_alignments = self.ref_atten(
            output, src_lens, ref_local_content_emb, ref_local_speaker_emb, ref_mels, ref_mel_lens
        )
        # print(local_spk_emb)  
        output = output + local_spk_emb
      
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            ref_content_predict,
            ref_alignments,
            ref_mel_masks,
            local_spk_emb
        )
