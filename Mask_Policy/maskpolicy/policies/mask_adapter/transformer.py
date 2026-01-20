# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, no_pos=False,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.no_pos = no_pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed,
                return_attn_map_enc: bool = False, return_attn_map_dec: bool = False,
                src_mask=None, memory_mask=None):
        assert len(src.shape) == 3
        # flatten NxHWxC to HWxNxC
        src = src.permute(1, 0, 2)                              # (HW, N, C)
        if not self.no_pos:
            pos_embed = pos_embed.permute(1, 0, 2)
        else:
            pos_embed = None
        query_embed = query_embed.permute(1, 0, 2)              # (Nq, N, C)

        tgt = torch.zeros_like(query_embed)

        # Convert src_mask to (B * num_heads, seq_len, seq_len) for encoder
        if src_mask is not None and src_mask.dim() == 3:
            B = src.shape[1]
            num_heads = self.encoder.layers[0].self_attn.num_heads
            seq_len = src_mask.shape[1]
            src_mask = src_mask.unsqueeze(
                1).expand(-1, num_heads, -1, -1).reshape(B * num_heads, seq_len, seq_len)

        memory, attn_map_enc = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed,
                                            src_mask=src_mask, need_weights=return_attn_map_enc)   # (HW, N, C)

        hs, attn_maps_dec = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                         pos=pos_embed, query_pos=query_embed,
                                         memory_mask=memory_mask,
                                         need_weights=return_attn_map_dec)      # (Nq, N, C)
        if return_attn_map_enc:
            attn_map_enc = torch.stack(attn_map_enc)
            attn_map_enc = attn_map_enc.cpu()
            if attn_map_enc.requires_grad:
                attn_map_enc = attn_map_enc.detach()
            attn_map_enc = attn_map_enc.numpy()
        else:
            attn_map_enc = None
        if return_attn_map_dec:
            attn_maps_dec = torch.stack(attn_maps_dec)
            attn_maps_dec = attn_maps_dec.cpu()
            if attn_maps_dec.requires_grad:
                attn_maps_dec = attn_maps_dec.detach()
            attn_maps_dec = attn_maps_dec.numpy()
        else:
            attn_maps_dec = None
        attn_maps = (attn_map_enc, attn_maps_dec)
        # (N, Nq, C)
        hs = hs.transpose(0, 1)
        return hs, attn_maps

    def forward_encoder(self, src, mask, pos_embed, return_attn_map_enc: bool = False):
        assert len(src.shape) == 3
        # flatten NxHWxC to HWxNxC
        src = src.permute(1, 0, 2)                              # (HW, N, C)
        if not self.no_pos:
            pos_embed = pos_embed.permute(1, 0, 2)
        else:
            pos_embed = None

        memory, attn_map_enc = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed,
                                            need_weights=return_attn_map_enc)   # (HW, N, C)
        if return_attn_map_enc:
            attn_map_enc = torch.stack(attn_map_enc)
            attn_map_enc = attn_map_enc.cpu()
            if attn_map_enc.requires_grad:
                attn_map_enc = attn_map_enc.detach()
            attn_map_enc = attn_map_enc.numpy()
        else:
            attn_map_enc = None
        return memory, attn_map_enc

    def forward_decoder(self, query_embed, memory, mask, pos_embed, return_attn_map_dec: bool = False):
        # flatten NxHWxC to HWxNxC
        if not self.no_pos:
            pos_embed = pos_embed.permute(1, 0, 2)
        else:
            pos_embed = None
        query_embed = query_embed.permute(1, 0, 2)              # (Nq, N, C)

        tgt = torch.zeros_like(query_embed)
        hs, attn_maps_dec = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                         pos=pos_embed, query_pos=query_embed,
                                         need_weights=return_attn_map_dec)      # (Nq, N, C)
        if return_attn_map_dec:
            attn_maps_dec = torch.stack(attn_maps_dec)
            attn_maps_dec = attn_maps_dec.cpu()
            if attn_maps_dec.requires_grad:
                attn_maps_dec = attn_maps_dec.detach()
            attn_maps_dec = attn_maps_dec.numpy()
        else:
            attn_maps_dec = None
        # (N, Nq, C)
        hs = hs.transpose(0, 1)
        return hs, attn_maps_dec


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None,
                need_weights: bool = False):
        output = src

        attn_maps = []

        # Use src_mask if provided (for label-based attention), otherwise use mask
        attn_mask = src_mask if src_mask is not None else mask

        for layer in self.layers:
            output, attn_map = layer(output, src_mask=attn_mask,
                                     src_key_padding_mask=src_key_padding_mask, pos=pos,
                                     need_weights=need_weights)
            if need_weights:
                attn_maps.append(attn_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_maps


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                need_weights: bool = False):
        output = tgt

        intermediate = []
        attn_maps = []

        for layer in self.layers:
            output, attn_map = layer(output, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     pos=pos, query_pos=query_pos,
                                     need_weights=need_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if need_weights:
                attn_maps.append(attn_map)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attn_maps

        return output, attn_maps


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     need_weights: bool = False):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_map = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask,
                                        need_weights=need_weights)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_map

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    need_weights: bool = False):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_map = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask,
                                        need_weights=need_weights)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn_map

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                need_weights: bool = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos,
                                    need_weights=need_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,
                                 need_weights=need_weights)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     need_weights: bool = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                             key=self.with_pos_embed(
                                                 memory, pos),
                                             value=memory, attn_mask=memory_mask,
                                             key_padding_mask=memory_key_padding_mask,
                                             need_weights=need_weights)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    need_weights: bool = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn_map = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                             key=self.with_pos_embed(
                                                 memory, pos),
                                             value=memory, attn_mask=memory_mask,
                                             key_padding_mask=memory_key_padding_mask,
                                             need_weights=need_weights)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_map

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                need_weights: bool = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                    need_weights=need_weights)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 need_weights=need_weights)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_label_based_attention_mask(
    patch_labels: torch.Tensor,
    label_attention_rules: dict,
    lowdim_label: int = -1,
    device: torch.device = None
) -> torch.Tensor:
    if device is None:
        device = patch_labels.device

    B, seq_len = patch_labels.shape
    attn_masks = []

    for b in range(B):
        labels = patch_labels[b]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        labels = labels.flatten().to(torch.long)

        # False = allow, True = masked
        attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        for query_label, allowed_labels in label_attention_rules.items():
            query_label_val = int(query_label)
            query_indices = torch.where(labels == query_label_val)[0]
            if query_indices.numel() == 0:
                continue

            key_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for allowed_label in allowed_labels:
                key_mask |= (labels == int(allowed_label))
                
            if not key_mask.any():
                continue

            disallowed = ~key_mask
            for q_idx in query_indices:
                attn_mask[q_idx, disallowed] = True

        attn_masks.append(attn_mask)

    return torch.stack(attn_masks, dim=0)