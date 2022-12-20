import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet, MelEncoder
from .modules import AdversarialClassifier, VarianceAdaptor
from .cdfse_modules import NormalEncoder, DownsampleEncoder, ReferenceAttention
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

        self.ds_times = 1
        for i in model_config["downsample_encoder"]["pooling_sizes"]:
            self.ds_times *= i 

        # If using speaker classification loss after local speaker embeddings
        self.use_spkcls = model_config["use_spkcls"]
        if self.use_spkcls:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_classifier = AdversarialClassifier(
                in_dim=model_config["downsample_encoder"]["out_dim"],
                out_dim=n_speaker,
                hidden_dims=model_config["classifier"]["cls_hidden"]
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
        speakers=None,
        ref_mels=None,
        ref_mel_lens=None,
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
        max_ref_mel_lens = ref_mels.shape[1]
        ref_mel_masks = get_mask_from_lengths(ref_mel_lens, max_ref_mel_lens)
        output = self.encoder(texts, src_masks)

        frame_feature = self.frame_encoder(ref_mels)
        content_feature = self.content_encoder(frame_feature, ref_mel_masks)
        ref_content_predict = self.phoneme_classifier(content_feature, is_reversal=False)
        ref_local_content_emb = self.ds_content_encoder(content_feature)
        ref_local_speaker_emb = self.ds_speaker_encoder(frame_feature)
        if self.use_spkcls:
            ref_local_lens = ref_mel_lens // self.ds_times
            ref_local_lens[ref_local_lens == 0] = 1
            max_ref_local_lens = max_ref_mel_lens // self.ds_times
            ref_local_spk_masks = (1 - get_mask_from_lengths(ref_local_lens, max_ref_local_lens).float()).unsqueeze(-1).expand(-1, -1, 256)
            spkemb = torch.sum(ref_local_speaker_emb * ref_local_spk_masks, axis=1) / ref_local_lens.unsqueeze(-1).expand(-1, 256)
            speaker_predicts = self.speaker_classifier(spkemb, is_reversal=False)
        else:
            speaker_predicts = None

        local_spk_emb, ref_alignments = self.ref_atten(
            output, src_lens, ref_local_content_emb, ref_local_speaker_emb, ref_mels, ref_mel_lens
        )

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
            local_spk_emb,
            speaker_predicts,
        )
