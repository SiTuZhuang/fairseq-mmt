# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, MultimodelMultiheadAttention
from torch import Tensor

from fairseq.modules import MultiheadAttention_Image

import math


class HighWayNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = args.attention_dropout

        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim * 2),
                                  nn.Sigmoid()))
        self.highway_linear = nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim)

    def forward(self, x, x1):

        x = torch.cat([x, x1], dim=-1)

        for i in range(2):
            h = getattr(self, 'highway_linear{}'.format(i))(x)
            g = getattr(self, 'highway_gate{}'.format(i))(x)
            x = g * h + (1 - g) * x
        x = self.highway_linear(x)
        x = nn.functional.dropout(x, self.dropout, self.training)
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.pre_mix = args.pre_mix
        self.image_encoder = TransformerEncoderLayer_image(args)

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        self.self_attn2 = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        self.gating = GatingMechanism(args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

        # self.fc_con = Linear(2*self.embed_dim, self.embed_dim)
        self.fc_con_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        # self.highway_net = HighWayNet(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def getBinaryTensor(self, i, boundary):
        one_matrix = torch.ones_like(i)
        zero_matrix = torch.zeros_like(i)

        return torch.where(i > boundary, one_matrix, zero_matrix)

    def mask(self, x, src_img_features, lay_idx):

        x = x.transpose(0, 1)  # batch * len * dim
        src_img_features = src_img_features.transpose(0, 1)  # batch * 49 * dim
        ########  mask  img ######### batch * 49 * len
        mask_img = torch.bmm(src_img_features, x.transpose(1, 2)) / math.sqrt(128)
        mask_img = F.softmax(mask_img, dim=-1)

        # mask_matrix = torch.mean(mask_img, dim=2, keepdim=True).repeat(1,1,49)
        # mask_img = F.sigmoid(mask_img)
        # mask_img = self.getBinaryTensor(mask_img,0.015)

        ########  mask  txt ######### batch * len * 49
        mask_txt = torch.bmm(x, src_img_features.transpose(1, 2)) / math.sqrt(128)
        mask_txt = F.softmax(mask_txt, dim=-1)

        mask_matrix = torch.bmm(mask_img, mask_txt).cuda()

        mask_matrix_output = self.getBinaryTensor(mask_matrix, 0.02)

        # mask_matrix_output = []
        # for i in mask_matrix:
        #     mask_list = i.reshape(1, -1)  # Ascending    # or i.view(src_img_features.size(1) * src_img_features.size(1))
        #     mask_list = sorted(mask_list.squeeze().tolist())
        #     num_tmp = int(len(mask_list) * 0.15)
        #     mask_matrix_tmp = self.getBinaryTensor(i, mask_list[num_tmp])
        #     mask_matrix_output.append(mask_matrix_tmp.tolist())

        return mask_matrix_output.detach()

    def forward(self, x, encoder_padding_mask, syn_img_features=None,
                lay_idx=0, kl_loss_coeff=0.0,
                attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        # residual = x
        # src_img_features = x[batch_len:]
        # x = x[:batch_len]
        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e4)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Gating with image features
        if syn_img_features is not None and lay_idx >= 3:
            gated, consistency, entity_score = self.gating(x, syn_img_features)
            self._gating_stats = (consistency.detach(), entity_score)  # No detach: allow gradient from adversarial reg
            x = x + gated

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        ####################################
        ########  image encoder  ###########

        # src_img_features_tmp = src_img_features

        # if lay_idx <= 0:
        #     encoder_padding_mask_image = torch.sum(src_img_features, dim=-1).eq(0).transpose(0, 1)
        #     src_img_mask = torch.zeros(src_img_features.size(1), src_img_features.size(0), src_img_features.size(0)).eq(
        #         1)
        #     src_img_features = self.image_encoder(lay_idx, src_img_features, encoder_padding_mask_image, src_img_mask)
        #
        # if lay_idx >= 3:
        #     encoder_padding_mask_image = torch.sum(src_img_features, dim=-1).eq(0).transpose(0, 1)
        #     src_img_features = self.image_encoder(lay_idx, src_img_features, encoder_padding_mask_image,
        #                                           mask_matrix_tmp)
		#
        # # ########  mask #########
        # # src_img_features = src_img_features + src_img_features_tmp
        #
        #
        # mask_matrix = self.mask(x, src_img_features, lay_idx)
        #
        #
        # ########  gating ########
        # src_img_features_tmp = src_img_features
        # if lay_idx >= 3:
        #
        #     src_img_features, gate = self.gating(x, src_img_features)
        #     x = x + src_img_features

        return x, syn_img_features



class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            ##########  cross attention src and tgt  #########
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            ##########  cross attention img and tgt  ##########
            # encoder_padding_mask_img = torch.zeros(encoder_out.size(1), 49).eq(1).cuda()
            # x_img, attn = self.encoder_attn(
            #     query=x,
            #     key=encoder_out[encoder_padding_mask.size(1):],
            #     value=encoder_out[encoder_padding_mask.size(1):],
            #     key_padding_mask=encoder_padding_mask_img,
            #     incremental_state=incremental_state,
            #     static_kv=True,
            #     need_weights=need_attn or (not self.training and self.need_attn),
            #     need_head_weights=need_head_weights,
            # )
            # x_img = F.dropout(x_img, p=self.dropout, training=self.training)
            # x = x + x_img
            ####################################################
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)



    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


####################################
#########  image encoder #########
class TransformerEncoderLayer_image(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention_Image(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

        # self.fc_con = Linear(2*self.embed_dim, self.embed_dim)
        self.fc_con_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        # self.highway_net = HighWayNet(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, lay_idx, src_img_features, encoder_padding_mask, mask_matrix_tmp,
                attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = src_img_features

        if self.normalize_before:
            src_img_features = self.self_attn_layer_norm(src_img_features)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        src_img_features, _ = self.self_attn(
            query=src_img_features,
            key=src_img_features,
            value=src_img_features,
            mask_matrix_tmp=mask_matrix_tmp.cuda(),
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        src_img_features = F.dropout(src_img_features, p=self.dropout, training=self.training)


        # src_img_features = residual + src_img_features

        if not self.normalize_before:
            src_img_features = self.self_attn_layer_norm(src_img_features)
        residual = src_img_features
        if self.normalize_before:
            src_img_features = self.final_layer_norm(src_img_features)
        src_img_features = self.activation_fn(self.fc1(src_img_features))
        src_img_features = F.dropout(src_img_features, p=float(self.activation_dropout), training=self.training)
        src_img_features = self.fc2(src_img_features)
        src_img_features = F.dropout(src_img_features, p=self.dropout, training=self.training)
        src_img_features = residual + src_img_features
        if not self.normalize_before:
            src_img_features = self.final_layer_norm(src_img_features)

        return src_img_features


######### gating  ##########
class GatingMechanism(nn.Module):
    """Consistency-Aware Gating: each token cross-attends to image grid regions,
    then computes a consistency score that modulates the visual contribution.
    Low consistency = image region doesn't match token = gate is suppressed."""
    def __init__(self, args):
        super().__init__()
        embed_dim = args.encoder_embed_dim
        self.consis_fc = nn.Linear(embed_dim * 2, 1)
        self.gate_fc = nn.Linear(embed_dim * 2, 1)
        self.advocate = nn.Linear(embed_dim, 1)  # adversarial: wants to USE visual
        self.skeptic  = nn.Linear(embed_dim, 1)  # adversarial: wants to SKIP visual
        nn.init.constant_(self.advocate.bias, 1.0)
        nn.init.constant_(self.skeptic.bias, -1.0)  # break saddle-point
        self.scale = embed_dim ** -0.5
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, grid_img_features):
        T, B, C = x.shape
        # Token-to-Image Cross-Attention
        attn = torch.bmm(
            x.permute(1, 0, 2),
            grid_img_features.permute(1, 2, 0)
        ) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        # Per-token selective visual feature (fine-grained)
        selective_visual = torch.bmm(
            attn_weights,
            grid_img_features.permute(1, 0, 2)
        ).permute(1, 0, 2)  # (T, B, C)
        # Global visual feature (coarse, for non-entity tokens)
        global_visual = grid_img_features.mean(dim=0, keepdim=True).expand(T, B, C)
        # Consistency scoring
        merge = torch.cat([x, selective_visual], dim=-1)
        consistency = torch.sigmoid(self.consis_fc(merge))
        adv = self.advocate(x)        # (T, B, 1)
        skp = self.skeptic(x)          # (T, B, 1)
        entity_score = torch.sigmoid(adv - skp)  # competition, (T, B, 1)
        gate = torch.sigmoid(self.gate_fc(merge))
        # === Phase2: DISABLED (entity_score not trained yet) ===
        # High entity_score (>0.7): full fine-grained cross-modal attention
        # Low entity_score (<0.3): pure text, no visual injection
        # Mid (0.3~0.7): global summary only (coarse visual context)
        mask_high = torch.ones_like(entity_score)  # Phase2 disabled
        mask_low  = torch.zeros_like(entity_score)  # Phase2 disabled
        mask_mid  = 1.0 - mask_high - mask_low
        effective_visual = (
            mask_high * selective_visual +
            mask_mid  * global_visual
            # mask_low * 0 = no visual contribution
        )
        # Consistency-modulated output
        final_gate = gate * consistency
        output = self.layer_norm(torch.mul(final_gate, effective_visual).transpose(0,1)).transpose(0,1)
        return output, consistency, entity_score


