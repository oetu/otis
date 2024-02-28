# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import trunc_normal_

from util.patch_embed import PatchEmbed
from util.pos_embed import get_1d_sincos_pos_embed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_map = None

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape # C = embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3) # (QKV, B, Heads, N, head_dim)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (B, Heads, N, head_dim)

        attn, attn_weights = self.mha(q, k, v, key_padding_mask=attn_mask)
        self.attn_map = attn_weights

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x
    

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size, modalities:dict, patch_size=(1, 100), global_pool=False, attention_pool=False, 
                 masking_blockwise=False, mask_ratio=0.0, mask_c_ratio=0.0, mask_t_ratio=0.0, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        embed_dim = kwargs['embed_dim']
        self.patch_embed = PatchEmbed(img_size[0], patch_size, embed_dim, flatten=False) # set flatten to False

        self.grid_size = {}
        for modality, shape in modalities.items():
            grid_size = (shape[1] // patch_size[0], shape[2] // patch_size[1])
            self.grid_size.update( {modality: grid_size} )

        assert embed_dim % 2 == 0
        self.max_num_patches_x = img_size[-1] // patch_size[1]
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.max_num_patches_x + 1, embed_dim // 2), requires_grad=False) # +1 cls embed

        total_num_embeddings_y = sum([v[0] for k, v in self.grid_size.items()])
        self.pos_embed_y = nn.Embedding(total_num_embeddings_y + 1, embed_dim // 2, padding_idx=0) # +1 padding embed

        # split into pos_embed_x and pos_embed_y
        del self.pos_embed

        self.masking_blockwise = masking_blockwise
        self.mask_ratio = mask_ratio
        self.mask_c_ratio = mask_c_ratio
        self.mask_t_ratio = mask_t_ratio

        if global_pool:
            self.pool = "global_pool" 
        elif attention_pool:
            self.pool = "attention_pool"
            self.attention_pool = nn.MultiheadAttention(embed_dim=kwargs['embed_dim'], 
                                                        num_heads=kwargs['num_heads'], batch_first=True)
        else:
            self.pool = False
            
        if self.pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        for block in self.blocks:
            # block.attn.forward = self._attention_forward_wrapper(block.attn)
            block.attn = Attention(kwargs['embed_dim'], kwargs['num_heads'], qkv_bias=kwargs['qkv_bias'])

        self.initialize_weights()

    def initialize_weights(self):
        # initialize learnable pos_embed for the vertical axis
        _pos_embed_y = torch.nn.Parameter(torch.randn(self.pos_embed_y.num_embeddings-1, 
                                                      self.pos_embed_y.embedding_dim) * .02)
        trunc_normal_(_pos_embed_y, std=.02)
        with torch.no_grad():
            self.pos_embed_y.weight[1:] = _pos_embed_y
                
        # initialize (and freeze) pos_embed for the horizontal axis by sin-cos embedding
        _pos_embed_x = get_1d_sincos_pos_embed(self.pos_embed_x.shape[-1], 
                                               self.pos_embed_x.shape[-2]-1, 
                                               cls_token=True)
        self.pos_embed_x.data.copy_(torch.from_numpy(_pos_embed_x).float().unsqueeze(0))

        # # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_blockwise(self, x, mask_c_ratio, mask_t_ratio):
        """
        2D: ECG recording (N, 1, C, T) (masking c and t under mask_c_ratio and mask_t_ratio)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        C, T = int(self.img_size[-2] / self.patch_size[-2]), int(self.img_size[-1] / self.patch_size[-1])
        
        # mask C
        x = x.reshape(N, C, T, D)
        len_keep_C = int(C * (1 - mask_c_ratio))
        noise = torch.rand(N, C, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_C]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_C(C'), T, D

        # mask T
        x = x.permute(0, 2, 1, 3) # N C' T D => N T C' D
        len_keep_T = int(T * (1 - mask_t_ratio))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_C, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3) # N T' C' D => N C' T' D 
        
        x_masked = x_masked.reshape(N, len_keep_T*len_keep_C, D) # N C' T' D => N L' D
            
        return x_masked, None, None

    def forward_features(self, x, pos_embed_y):
        """
        x: [B=N, L, D], sequence
        pos_embed_y: [B=N, C', T'], with C'*T'=L and C'=H/p, T'=W/q

        Note: patch_size: (p, q) 
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)

        # add pos embed X w/o cls token
        # (1, 1+T'_max, D/2)
        pos_embed_x = self.pos_embed_x
        # (1, 1+T'_max, D), padding left
        pos_embed_x = torch.nn.functional.pad(pos_embed_x, (x.shape[1]//2, 0), "constant", 0)
        # (1, D, 1, 1+T'_max)
        pos_embed_x_batch = torch.permute(pos_embed_x, (0, 2, 1)).unsqueeze(-2)
        # (1, D, 1, T')
        pos_embed_x_batch = pos_embed_x_batch[..., 1:x.shape[-1]+1]
        # (1, D, C', T')
        pos_embed_x_batch = pos_embed_x_batch.expand(-1, -1, x.shape[2], -1)

        # (B, D, C', T')
        x = x + pos_embed_x_batch

        # add pos embed Y
        # (B, C', T', D/2)
        pos_embed_y_batch = self.pos_embed_y(pos_embed_y)
        # (B, C', T', D), padding right
        pos_embed_y_batch = torch.nn.functional.pad(pos_embed_y_batch, (0, x.shape[1]//2), "constant", 0)
        # (B, D, C', T')
        pos_embed_y_batch = torch.permute(pos_embed_y_batch, (0, 3, 1, 2))
        
        # (B, D, C', T')
        x = x + pos_embed_y_batch

        # flatten
        # (B, N, D), with N=C'*T'
        x = x.flatten(2).transpose(1, 2)

        if self.masking_blockwise:
            x, _, _ = self.random_masking_blockwise(x, self.mask_c_ratio, self.mask_t_ratio)
        else:
            x, _, _ = self.random_masking(x, self.mask_ratio)

        # append cls token
        # (1, 1, D)
        cls_token = self.cls_token + pos_embed_x[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # (B, 1+N, D)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.pool == "attention_pool":
            q = x[:, 1:, :].mean(dim=1, keepdim=True)
            k = x[:, 1:, :]
            v = x[:, 1:, :]
            x, x_weights = self.attention_pool(q, k, v) # attention pool without cls token
            outcome = self.fc_norm(x.squeeze(dim=1))
        elif self.pool == "global_pool":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else: # cls token
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, :]
        x = self.fc_norm(x)
        
        return x if pre_logits else self.head(x)
    
    def forward(self, x, pos_embed_y):
        x = self.forward_features(x, pos_embed_y)
        x = self.forward_head(x)
        return x


def vit_pluto_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=256, depth=2, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tinyDeep_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_smallDeep_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_medium_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=640, depth=5, num_heads=10, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_mediumDeep_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=576, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=864, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge(**kwargs):
    model = VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

        
# def _attention_forward_wrapper(self, attn_obj):
#     """
#     Modified version of def forward() of class Attention() in timm.models.vision_transformer
#     """
#     def my_forward(x):
#         B, N, C = x.shape # C = embed_dim
#         # (3, B, Heads, N, head_dim)
#         qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

#         # (B, Heads, N, N)
#         attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
#         attn = attn.softmax(dim=-1)
        
#         # (B, Heads, N, N)
#         attn_obj.attn_map = attn # this was added 

#         # (B, Heads, N, N)
#         attn = attn_obj.attn_drop(attn)

#         # (B, N, Heads*head_dim)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = attn_obj.proj(x)
#         x = attn_obj.proj_drop(x)
#         return x
#     return my_forward