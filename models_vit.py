# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae?tab=readme-ov-file
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import math 

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import trunc_normal_

from util.patch_embed import PatchEmbed
from util.pos_embed import get_1d_sincos_pos_embed
from util.transformer import Attention, DyT
    

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size, domains:dict, patch_size=(1, 100), global_pool=False, attention_pool=False, 
                 masking_blockwise=False, mask_ratio=0.0, mask_c_ratio=0.0, mask_t_ratio=0.0, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        embed_dim = kwargs['embed_dim']
        self.patch_embed = PatchEmbed(img_size[0], patch_size, embed_dim, flatten=False) # set flatten to False

        self.grid_height = {}
        for domain, input_size in domains.items():
            grid_height = input_size[1] // patch_size[0]      # number of variates
            self.grid_height.update( {domain: grid_height} )

        assert embed_dim % 2 == 0
        self.max_num_patches_x = img_size[-1] // patch_size[1]
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.max_num_patches_x + 1, embed_dim // 2), requires_grad=False) # +1 cls embed

        total_num_embeddings_y = sum([v for k, v in self.grid_height.items()])
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
        len_keep = math.ceil(L * (10 - 10 * mask_ratio)/10) # factor 10 to compensate float precision 
        
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
        Time series of shape (N, 1, C, T), where C and T are masked separately.
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
        x: [B=N, 1, C, T], sequence
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
            # (B, 1, D)
            x, x_weights = self.attention_pool(q, k, v) # attention pool without cls token
            # (B, D)
            outcome = x.squeeze(dim=1)
        elif self.pool == "global_pool":
            # (B, D)
            outcome = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else: # cls token
            # (B, 1+N, D)
            x = self.norm(x)
            # (B, D)
            outcome = x[:, 0]

        return outcome

    def forward_head(self, x, pre_logits: bool = False):
        x = self.fc_norm(x)

        return x if pre_logits else self.head(x)
    
    def forward(self, x, pos_embed_y):
        x = self.forward_features(x, pos_embed_y)
        x = self.forward_head(x)
        return x


def vit_baseDeep_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def vit_largeDeep_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def vit_hugeDeep_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=576, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model