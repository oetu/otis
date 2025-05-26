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

import random
import math

import torch
import torch.nn as nn

import numpy as np

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_
from timm.models.layers import Mlp

from util.patch_embed import PatchEmbed
from util.pos_embed import get_1d_sincos_pos_embed
import util.statistics as statistics
from util.transformer import Attention, TemporalRoPEAttention, DyT


class OTiS(nn.Module):
    """ 
        Open model for general time series analysis 
    """
    def __init__(self, domains:dict, domain_weights:dict, domain_agnostic:str=False, 
                 input_channels=1, time_steps=2500, patch_size=(1, 100),
                 embed_dim=1024, depth=24, num_heads=16,
                 output_projection='decoder',
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, separate_dec_pos_embed_y=False,
                 head_mlp_ratio=4., head_dropout=0.1, head_activation=nn.GELU,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False, masked_patch_loss=False, domain_weighted_loss=False, contrastive_loss=False,
                 probabilistic_masking=False, include_forecasting=False, forecasting_probability=0.33, forecasting_mask_ratio=0.5,
                 downstream=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # OTiS encoder specifics
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_channels, patch_size, embed_dim, flatten=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.grid_height = {}
        for domain, input_size in domains.items():
            grid_height = input_size[1] // patch_size[0]      # number of variates
            self.grid_height.update( {domain: grid_height} )

        assert embed_dim % 2 == 0
        max_num_patches_x = time_steps // patch_size[1]
        self.max_num_patches_x = max_num_patches_x
        self.pos_embed_x = nn.Parameter(torch.zeros(1, max_num_patches_x + 1, embed_dim // 2), requires_grad=False) # +1 cls embed

        self.domain_agnostic = domain_agnostic
        if self.domain_agnostic:
            # domain-agnostic pos_embed_y (i.e., shared across all domains)
            total_num_embeddings_y = 256
        else:
            # domain-specific pos_embed_y
            total_num_embeddings_y = sum([v for k, v in self.grid_height.items()])
        self.pos_embed_y = nn.Embedding(total_num_embeddings_y + 1, embed_dim // 2, padding_idx=0) # +1 padding embed

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # modify the attention operation to consider attention masks
        for block in self.blocks:
            block.forward = self._block_forward_wrapper(block)
            block.attn = Attention(embed_dim, num_heads, qkv_bias=True)
            # block.attn = TemporalRoPEAttention(embed_dim, num_heads, max_num_patches_x, qkv_bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # OTiS output projection specifics
        self.output_projection = output_projection
        
        if self.output_projection == 'mlp':
            self.mask_token_encoder = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
            self.mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * head_mlp_ratio),
                act_layer=head_activation,
                drop=head_dropout,
            )
            
            self.mlp_norm = norm_layer(embed_dim)
            self.mlp_pred = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * input_channels, bias=True)
        else: # decoder
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            assert decoder_embed_dim % 2 == 0
            self.decoder_pos_embed_x = nn.Parameter(torch.zeros(1, max_num_patches_x + 1, decoder_embed_dim // 2), requires_grad=False) # +1 cls embed
            self.separate_dec_pos_embed_y = separate_dec_pos_embed_y
            if self.separate_dec_pos_embed_y:
                self.decoder_pos_embed_y = nn.Embedding(total_num_embeddings_y + 1, decoder_embed_dim // 2, padding_idx=0) # +1 padding embed
            else:
                self.decoder_pos_embed_y = nn.Linear(embed_dim // 2, decoder_embed_dim // 2)

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, act_layer=nn.GELU, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * input_channels, bias=True) # decoder to patch

            # modify the attention operation to consider attention masks
            for block in self.decoder_blocks:
                block.forward = self._block_forward_wrapper(block)
                block.attn = Attention(decoder_embed_dim, decoder_num_heads, qkv_bias=True)
                # block.attn = TemporalRoPEAttention(decoder_embed_dim, decoder_num_heads, max_num_patches_x, qkv_bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Contrastive specifics
        self.criterion = torch.nn.CosineSimilarity(dim=1)

        proj_dim = int(1024)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            # nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False)
            # nn.LayerNorm(embed_dim),
        )

        pred_dim = int(128)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            # nn.LayerNorm(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, embed_dim, bias=False),
            # nn.BatchNorm1d(embed_dim, affine=False)
            # nn.LayerNorm(embed_dim),
        )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.masked_patch_loss = masked_patch_loss

        self.domain_weights = domain_weights
        self.domain_weighted_loss = domain_weighted_loss

        self.contrastive_loss = contrastive_loss

        self.probabilistic_masking = probabilistic_masking
        self.include_forecasting = include_forecasting
        self.forecasting_probability = forecasting_probability
        self.forecasting_mask_ratio = forecasting_mask_ratio

        self.downstream = downstream

        self.initialize_weights()

    def activate_masked_loss(self):
        self.masked_patch_loss = True

    def _block_forward_wrapper(self, block_obj):
        """
        Modified version of def forward() of class Block() in timm.models.vision_transformer
        """
        def my_forward(x, attn_mask=None):
            x = x + block_obj.drop_path1(block_obj.ls1(block_obj.attn(block_obj.norm1(x), attn_mask)))
            x = x + block_obj.drop_path2(block_obj.ls2(block_obj.mlp(block_obj.norm2(x))))
            return x
    
        # # RoPE (rotary position embeddings)
        # def my_forward(x, attn_mask=None, attn_mask_input=None, ids_restore=None, V=1, T=1):
        #     x = x + block_obj.drop_path1(block_obj.ls1(block_obj.attn(block_obj.norm1(x), attn_mask, 
        #                                                               attn_mask_input, ids_restore, V, T)))
        #     x = x + block_obj.drop_path2(block_obj.ls2(block_obj.mlp(block_obj.norm2(x))))
        #     return x

        return my_forward

    def initialize_weights(self):
        # initialize learnable pos_embed for the vertical axis
        _pos_embed_y = torch.nn.Parameter(torch.randn(self.pos_embed_y.num_embeddings-1, 
                                                      self.pos_embed_y.embedding_dim) * .02)
        trunc_normal_(_pos_embed_y, std=.02)
        with torch.no_grad():
            self.pos_embed_y.weight[1:] = _pos_embed_y

        if self.output_projection == "decoder" and self.separate_dec_pos_embed_y:
            _decoder_pos_embed_y = torch.nn.Parameter(torch.randn(self.decoder_pos_embed_y.num_embeddings-1, 
                                                                self.decoder_pos_embed_y.embedding_dim) * .02)
            trunc_normal_(_decoder_pos_embed_y, std=.02)
            with torch.no_grad():
                self.decoder_pos_embed_y.weight[1:] = _decoder_pos_embed_y
                
        # initialize (and freeze) pos_embed for the horizontal axis by sin-cos embedding
        _pos_embed_x = get_1d_sincos_pos_embed(self.pos_embed_x.shape[-1], 
                                               self.pos_embed_x.shape[-2]-1, 
                                               cls_token=True)
        self.pos_embed_x.data.copy_(torch.from_numpy(_pos_embed_x).float().unsqueeze(0))

        if self.output_projection == "decoder":
            _decoder_pos_embed_x = get_1d_sincos_pos_embed(self.decoder_pos_embed_x.shape[-1], 
                                                            self.decoder_pos_embed_x.shape[-2]-1, 
                                                            cls_token=True)
            self.decoder_pos_embed_x.data.copy_(torch.from_numpy(_decoder_pos_embed_x).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        if self.output_projection == "decoder":
            torch.nn.init.normal_(self.mask_token, std=.02)
        else: # mlp
            torch.nn.init.normal_(self.mask_token_encoder, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, p*q*C)
        """
        p, q = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % q == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // q
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, q))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * q * imgs.shape[1]))
        return x

    def unpatchify(self, x, attn_mask):
        """
        x: (N, L, p*q*C)
        attn_mask: (N, C', T'], with C'=h=H/p, T'=w=W/q, L=C'*T'
        imgs: (N, C, H, W)
        """
        p, q = self.patch_size
        h, w = attn_mask.shape[1], attn_mask.shape[2]
        assert h * w == x.shape[1]
        
        img_channels = int(x.shape[2] / (p*q))

        x = x.reshape(shape=(x.shape[0], h, w, p, q, img_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], img_channels, h * p, w * q))
        return imgs

    def random_masking(self, x, attn_mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        attn_mask: [N, C', T'], with L=C'*T'
        """
        N, L, D = x.shape  # batch, length, dim
        if self.probabilistic_masking:
            masking_ratio = mask_ratio - 0.05 + 0.1 * np.abs(np.random.normal(loc=0.0, scale=0.65))
        else:
            masking_ratio = mask_ratio
        len_keep = math.ceil(L * (10 - 10 * masking_ratio)/10) # factor 10 to compensate float precision 
        
        if self.downstream == "forecasting" or (self.include_forecasting and random.random() < self.forecasting_probability):
            if self.include_forecasting:
                # pretraining
                if self.probabilistic_masking:
                    forecasting_ratio = self.forecasting_mask_ratio - 0.05 + 0.1 * np.abs(np.random.normal(loc=0.0, scale=0.65))
                else:
                    forecasting_ratio = self.forecasting_mask_ratio
            else:
                # downstream finetuning
                forecasting_ratio = mask_ratio

            # how much to keep (= 1 - mask out)
            len_keep = math.ceil(L * (10 - 10 * forecasting_ratio)/10) # factor 10 to compensate float precision 
            
            # [N, C', T']
            N, nb_of_channels, nb_of_patches = attn_mask.shape

            # Generate noise
            # [C', T']
            noise = torch.arange(0, nb_of_patches, device=x.device).to(torch.float32).repeat(nb_of_channels, 1).unsqueeze(0)
            # [C', T']
            noise.add_(torch.linspace(0, 0.5, steps=nb_of_channels, device=x.device).view(-1, 1))
            # [N, C', T']
            noise = noise.repeat(N, 1, 1)
            # [N, C', T']
            noise.mul_(attn_mask)

            # Determine maximum noise value
            # [N, 1, 1]
            noise_max = noise.flatten(1).max(dim=-1)[0].view(-1, 1, 1)

            # Create auxiliary mask
            # to set values of masked patches to infinity such that they are certainly removed
            # [N, 1, 1]
            len_keeps = torch.ceil( attn_mask[:, 0, ...].sum(dim=-1) * (10 - 10 * forecasting_ratio) / 10 ).to(torch.long).view(-1, 1, 1)
            # [N, C', T']
            aux_mask = torch.arange(nb_of_patches, device=x.device).expand(N, nb_of_channels, nb_of_patches) < len_keeps
            aux_mask = 1 - aux_mask.to(torch.float32)
            aux_mask.mul_(attn_mask)
            aux_mask = torch.nan_to_num(aux_mask * torch.inf, nan=0.0)

            # Apply auxiliary mask
            # [N, C', T']
            noise.add_(aux_mask)

            # Assign random values to padding tokens such that
            # visible_patches.values < padding_tokens.values < masked_patches.values
            # [N, C', T']
            padding_noise = noise_max + torch.rand(nb_of_channels, nb_of_patches, device=x.device).unsqueeze(0)
            noise.add_((1 - attn_mask) * padding_noise)

            # Flatten noise
            # [N, L]
            noise = noise.flatten(1)
        else:
            # [N, C', T']
            N, nb_of_channels, nb_of_patches = attn_mask.shape

            # Generate noise
            # [N, C', T']
            noise = torch.rand(N, nb_of_channels, nb_of_patches, device=x.device)
            noise.mul_(attn_mask)
            noise.add_(torch.nan_to_num((1 - attn_mask) * torch.inf, nan=0.0))

            # Create auxiliary mask
            # to set values of masked patches to infinity such that they are certainly removed
            # [N, 1, 1], nb of unmasked tokens per variate
            len_keeps = torch.ceil(attn_mask[:, 0, :].sum(dim=-1) * (10-10*masking_ratio)/10).to(torch.long).view(-1, 1, 1)
            # [N, C', T']
            aux_mask = torch.arange(nb_of_patches, device=x.device).expand(N, nb_of_channels, nb_of_patches) < len_keeps
            aux_mask = 1 - aux_mask.to(torch.float32)
            aux_mask.mul_(attn_mask)
            aux_mask = torch.nan_to_num(aux_mask * torch.inf, nan=0.0)

            # Sort noise
            # [N, C', T']
            noise, aux_ids_shuffle = torch.sort(noise, dim=-1)
            aux_ids_restore = torch.argsort(aux_ids_shuffle, dim=-1)

            # Apply auxiliary mask and restore noise
            # [N, C', T']
            noise.add_(aux_mask)
            noise = torch.gather(noise, dim=-1, index=aux_ids_restore)

            # Apply attention mask
            # [N, C', T']
            noise.mul_(attn_mask)

            # Assign random values to padding tokens such that
            # visible_patches.values < padding_tokens.values < masked_patches.values
            # [N, C', T']
            padding_noise = torch.rand(1, nb_of_channels, nb_of_patches, device=x.device) + 1
            noise.add_((1 - attn_mask) * padding_noise)

            # Flatten noise
            # [N, L]
            noise = noise.flatten(1)
        
        # sort noise for each sample
        # [N, L], the first len_keep indices correspond to the visible tokens, the remaining L-len_keep to the masked tokens
        ids_shuffle = torch.argsort(noise, dim=1)   # ascend: small is keep, large is remove
        # [N, L], to restore the original token order after concatenating masked tokens to the visible tokens (before the decoding)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # [N, len_keep]
        ids_keep = ids_shuffle[:, :len_keep]
        # [N, len_keep, D]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # [N, L]
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, attn_mask, pos_embed_y, mask_ratio):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            attn_mask: (B, C', T'), with N=C'*T' patches 
            pos_embed_y: (B, C', T'), with N=C'*T' embedding ids

        output:
            x: (B, 1+N', D), with 1 cls token + N' visible patches
            mask: (B, N), with N (visible + mask) patches, 0 is keep, 1 is remove
            ids_restore: (B, N)
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)
        B, D, V, T = x.shape

        # add pos embed X w/o cls token
        # (B, D, C', T')
        pos_embed_x_mask = attn_mask.unsqueeze(1).expand(-1, x.shape[1], -1, -1)

        # (1, 1+T'_max, D/2)
        pos_embed_x = self.pos_embed_x
        # (1, 1+T'_max, D), padding left
        pos_embed_x = torch.nn.functional.pad(pos_embed_x, (x.shape[1]//2, 0), "constant", 0)
        # (1, D, 1, 1+T'_max)
        pos_embed_x_batch = torch.permute(pos_embed_x, (0, 2, 1)).unsqueeze(-2)
        # (B, D, C', T')
        pos_embed_x_batch = pos_embed_x_mask * pos_embed_x_batch[..., 1:pos_embed_x_mask.shape[-1]+1]

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

        # masking: length -> length * mask_ratio
        # (B, N', D), with N'=C'*T'*(1-mask_ratio)
        x, mask, ids_restore = self.random_masking(x, attn_mask, mask_ratio)

        # append cls token
        # (1, 1, D)
        cls_token = self.cls_token + pos_embed_x[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # (B, 1+N', D)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # (B, N')
        attn_mask_visible_patches = attn_mask.flatten(1)[mask==0].view(attn_mask.shape[0], -1)
        # (B, 1+N'), add cls token to attn mask
        attn_mask_visible_patches = torch.cat((torch.ones(size=(attn_mask.shape[0], 1), device=x.device), attn_mask_visible_patches), dim=1)

        for blk in self.blocks:
            x = blk(x, attn_mask_visible_patches)
            # x = blk(x, attn_mask_visible_patches, attn_mask, ids_restore, V, T) # RoPE

        # (B, 1+N', D)        
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_encoder_with_masked_patches(self, x, attn_mask, pos_embed_y, mask_ratio):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            attn_mask: (B, C', T'), with N=C'*T' patches 
            pos_embed_y: (B, C', T'), with N=C'*T' embedding ids

        output:
            x: (B, 1+N, D), with 1 cls token + N (visible + masked) patches
            mask: (B, N), with N (visible + mask) patches, 0 is keep, 1 is remove
            ids_restore: (B, N)
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)
        B, D, V, T = x.shape

        # flatten
        # (B, N, D), with N=C'*T'
        x = x.flatten(2).transpose(1, 2)

        # masking: length -> length * mask_ratio
        # (B, N', D), with N'=C'*T'*(1-mask_ratio)
        x, mask, ids_restore = self.random_masking(x, attn_mask, mask_ratio)

        # append mask tokens to sequence
        # (B, N-N', D)
        mask_tokens = self.mask_token_encoder.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        # (B, N, D)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # permute and reshape
        # (B, D, N)
        x = torch.permute(x, (0, 2, 1))

        # (B, D, C', T')
        x = x.view(x.shape[0], x.shape[1], attn_mask.shape[-2], -1)

        # add pos embed X w/o cls token
        # (B, D, C', T')
        pos_embed_x_mask = attn_mask.unsqueeze(1).expand(-1, x.shape[1], -1, -1)

        # (1, 1+T'_max, D/2)
        pos_embed_x = self.pos_embed_x
        # (1, 1+T'_max, D), padding left
        pos_embed_x = torch.nn.functional.pad(pos_embed_x, (x.shape[1]//2, 0), "constant", 0)
        # (1, D, 1, 1+T'_max)
        pos_embed_x_batch = torch.permute(pos_embed_x, (0, 2, 1)).unsqueeze(-2)
        # (B, D, C', T')
        pos_embed_x_batch = pos_embed_x_mask * pos_embed_x_batch[..., 1:pos_embed_x_mask.shape[-1]+1]

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

        # append cls token
        # (1, 1, D)
        cls_token = self.cls_token + pos_embed_x[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # (B, 1+N, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # (B, N)
        attn_mask_all_patches = attn_mask.flatten(1)
        # (B, 1+N), add cls token to attn mask
        attn_mask_all_patches = torch.cat((torch.ones(size=(attn_mask.shape[0], 1), device=x.device), attn_mask_all_patches), dim=1)

        for blk in self.blocks:
            x = blk(x, attn_mask_all_patches)
            # x = blk(x, attn_mask_all_patches, None, None, V, T) # RoPE

        # (B, 1+N, D)        
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_all_patches(self, x,  pos_embed_y):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            pos_embed_y: (B, C', T'), with N=C'*T' embedding ids

        output:
            x: (B, 1+N, D), with 1 cls token + N (visible + masked) patches
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)
        B, D, V, T = x.shape

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

        # append cls token
        # (1, 1, D)
        cls_token = self.cls_token + pos_embed_x[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # (B, 1+N, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # (B, 1+N), add cls token to attn mask
        for blk in self.blocks:
            x = blk(x)
            # x = blk(x, None, None, None, V, T) # RoPE

        x = self.norm(x)

        return x

    def forward_decoder(self, x, attn_mask, pos_embed_y, ids_restore):
        """
        input:
            x: (B, 1+N', D_dec), with 1 cls token + N' visible patches
            attn_mask: (B, C', T'), with N=C'*T' (visible + mask) patches 
            pos_embed_y: (B, C', T'), with N=C'*T' embedding ids 
            ids_restore: (B, N)

        output:
            x: (B, N, p*q*input_channels), with p=patch_size_x, q=patch_size_y
        """
        # embed tokens
        # (B, 1+N', D_dec)
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # + 1 because x includes cls token
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # (B, 1+N, D_dec)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
 
        # preserve cls token
        # (B, 1, D_dec)
        cls_token = x[:, :1, :]
        
        # permute and reshape
        # (B, D_dec, N)
        x = torch.permute(x[:, 1:, :], (0, 2, 1))

        # (B, D_dec, C', T')
        x = x.view(x.shape[0], x.shape[1], attn_mask.shape[-2], -1)
        B, D_dec, V, T = x.shape

        # add pos embed x
        # (B, D_dec, C', T')
        decoder_pos_embed_x_mask = attn_mask.unsqueeze(1).expand(-1, x.shape[1], -1, -1)

        # (1, 1+T'_max, D_dec/2)
        decoder_pos_embed_x = self.decoder_pos_embed_x
        # (1, 1+T'_max, D_dec), padding left
        decoder_pos_embed_x = torch.nn.functional.pad(decoder_pos_embed_x, (x.shape[1]//2, 0), "constant", 0)
        # (1, D_dec, 1, 1+T'_max)
        decoder_pos_embed_x_batch = torch.permute(decoder_pos_embed_x, (0, 2, 1)).unsqueeze(-2)
        # (B, D_dec, C', T')
        decoder_pos_embed_x_batch = decoder_pos_embed_x_mask * decoder_pos_embed_x_batch[..., 1:decoder_pos_embed_x_mask.shape[-1]+1]

        # (B, D_dec, C', T')
        x = x + decoder_pos_embed_x_batch

        # add pos embed Y
        # (B, C', T', D_dec/2)
        if self.separate_dec_pos_embed_y:
            decoder_pos_embed_y_batch = self.decoder_pos_embed_y(pos_embed_y)
        else:
            decoder_pos_embed_y_batch = self.decoder_pos_embed_y(self.pos_embed_y(pos_embed_y))
        # (B, C', T', D_dec), padding right
        decoder_pos_embed_y_batch = torch.nn.functional.pad(decoder_pos_embed_y_batch, (0, x.shape[1]//2), "constant", 0)
        # (B, D_dec, C', T')
        decoder_pos_embed_y_batch = torch.permute(decoder_pos_embed_y_batch, (0, 3, 1, 2))

        # (B, D_dec, C', T')
        x = x + decoder_pos_embed_y_batch

        # flatten
        # (B, N, D_dec), with N=C'*T'
        x = x.flatten(2).transpose(1, 2) 

        # append cls token
        # (B, 1, D_dec)
        cls_token = cls_token + decoder_pos_embed_x[:, :1, :]
        # (B, 1+N, D_dec)
        x = torch.cat((cls_token, x), dim=1)

        # apply Transformer blocks
        # (B, 1+N), add cls token to attn mask
        attn_mask_batch = torch.cat((torch.ones(size=(attn_mask.shape[0], 1), device=x.device), attn_mask.flatten(1)), dim=1)

        for blk in self.decoder_blocks:
            x = blk(x, attn_mask_batch)
            # x = blk(x, attn_mask_batch, None, None, V, T) # RoPE

        # (B, 1+N, D_dec)
        x = self.decoder_norm(x)

        # predictor projection
        # (B, 1+N, p*q*input_channels)
        x = self.decoder_pred(x)

        # remove cls token
        # (B, N, p*q*input_channels)
        x = x[:, 1:, :]

        return x
    
    def forward_mlp(self, x):
        """
        input:
            x: (B, 1+N, D), with 1 cls token + N (visible + masked) patches

        output:
            x: (B, N, p*q*input_channels), with p=patch_size_x, q=patch_size_y
        """
        # preserve cls token
        # (B, 1, D)
        cls_token = x[:, :1, :]
        
        # remove cls token
        x = x[:, 1:, :]

        # apply MLP
        # (B, N, D)
        x = self.mlp(x)

        # (B, N, D)
        x = self.mlp_norm(x)

        # predictor projection
        # (B, N, p*q*input_channels)
        x = self.mlp_pred(x)

        return x

    def forward_loss(self, imgs, pred, attn_mask, mask, domain):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*q*C]
        attn_mask: [N, C', T'], with C'=H/p, T'=W/q and C'*T'=L
        mask: [N, L], 0 is keep, 1 is remove
        domain: [N], the domain of the sample
        """
        # [N, L, p*q*C]
        target = self.patchify(imgs) 

        if self.norm_pix_loss:
            # mean over last dim does not require consideration of the attention mask
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # [N, L, p*q*C]
        loss = (pred - target) ** 2
        # [N, L, 1]
        attn_mask_loss = attn_mask.flatten(1).unsqueeze(-1)
        # [N, L, p*q*C]
        loss = loss * attn_mask_loss
        # [N, L], mean loss per patch
        # mean over last dim does not require consideration of the attention mask
        loss = loss.sum(dim=-1) / (loss.shape[-1] + 1e-9)

        # REGULARIZATION (using normalized correlation coefficient of the actual signals)
        # [N, C, H, W]
        imgs_hat = self.unpatchify(pred, attn_mask)
        # [N, C, H, W]
        attn_mask_input_space = torch.nn.functional.interpolate(attn_mask.unsqueeze(1), scale_factor=self.patch_size, mode="nearest")
        # [N, C, H, W]
        imgs_hat = imgs_hat * attn_mask_input_space

        # batch_size = len(imgs)
        if self.masked_patch_loss:
            # compute loss only on masked patches
            # [N, C, H, W]
            combined_mask = attn_mask.flatten(1) * mask # attention mask combined with the actual mask (visible vs masked tokens)
            # combined_mask_input_space = torch.nn.functional.interpolate(combined_mask.reshape(attn_mask.shape).unsqueeze(1), 
            #                                                             scale_factor=self.patch_size, 
            #                                                             mode="nearest")
            
            # [N]
            # number of reconstructed (masked only) patches
            # may also be 0 if time series length < patch size
            # hence only consider the samples with nb_patches > 0 
            nb_patches = torch.sum(combined_mask, dim=-1)

            # [N]
            loss = (torch.sum(loss * mask, dim=-1) / nb_patches)[nb_patches > 0]

            # # [N, C, H, W]
            # imgs_masked_patches = imgs * combined_mask_input_space
            # imgs_hat_masked_patches = imgs_hat * combined_mask_input_space

            # # [N]
            # ncc = statistics.ncc(imgs_masked_patches, imgs_hat_masked_patches, combined_mask_input_space, keep_batch=True)[nb_patches > 0]
            
            # # [N]
            # nb_patches = nb_patches[nb_patches > 0]
        else:
            # compute loss on all (masked + visible) patches
            # [N]
            # number of reconstructed (masked + visible) patches
            nb_patches = torch.sum(attn_mask.flatten(1), dim=-1)
            
            # [N]
            loss = (torch.sum(loss, dim=-1) / nb_patches)[nb_patches > 0]

        # [N]
        ncc = statistics.ncc(imgs, imgs_hat, attn_mask_input_space, keep_batch=True)[nb_patches > 0]

        # [N]
        nb_patches = nb_patches[nb_patches > 0]

        if self.domain_weighted_loss:
            # weighted mean
            domain_weights_batch = torch.stack( [self.domain_weights[mod] for mod in domain] ).to(device=imgs.device, non_blocking=True)
            
            batch_weight = torch.sum(domain_weights_batch) + 1e-9
            loss_batch = torch.sum(domain_weights_batch * loss) / batch_weight
            ncc_batch = torch.sum(domain_weights_batch * ncc) / batch_weight
        else:
            # mean
            loss_batch = torch.mean(loss)
            ncc_batch = torch.mean(ncc)

        return loss_batch, ncc_batch, imgs_hat

    def forward(self, imgs, attn_mask, pos_embed_y, domain, mask_ratio=0.75):
        """
        imgs: [N, C, H, W]
        attn_mask: [N, C', T'], with C'*T'=L and C'=H/p, T'=W/q
        pos_embed_y: [N, C', T'], with C'*T'=L and C'=H/p, T'=W/q 
        """
        if self.output_projection == 'decoder':
            # latent of visible tokens
            latent, mask, ids_restore = self.forward_encoder(imgs, attn_mask, pos_embed_y, mask_ratio)
            pred = self.forward_decoder(latent, attn_mask, pos_embed_y, ids_restore)  # [N, L, p*q*C]
        else: # mlp
            # latent of all tokens
            latent, mask, ids_restore = self.forward_encoder_with_masked_patches(imgs, attn_mask, pos_embed_y, mask_ratio)
            pred = self.forward_mlp(latent)  # [N, L, p*q*C]
        
        loss, ncc, imgs_hat = self.forward_loss(imgs, pred, attn_mask, mask, domain)

        if self.contrastive_loss:
            # contrastive part
            latent2, _, _ = self.forward_encoder(imgs, attn_mask, pos_embed_y, mask_ratio)

            attn_mask_visible_patches = attn_mask.flatten(1)[mask==0].view(attn_mask.shape[0], -1)
            z1 = statistics.masked_mean(latent[:, 1:, ...], attn_mask_visible_patches, dim=1)     # global average pooling
            z2 = statistics.masked_mean(latent2[:, 1:, ...], attn_mask_visible_patches, dim=1)    # global average pooling
            
            p1 = self.projector(z1)
            p2 = self.projector(z2)

            h1 = self.predictor(p1)
            h2 = self.predictor(p2)

            # cos_sim = - (self.criterion(h1, p2).mean() + self.criterion(h2, p1).mean()) * 0.5
            cos_sim = - (self.criterion(h1, p2.detach()).mean() + self.criterion(h2, p1.detach()).mean()) * 0.5
            # cos_sim = - (self.criterion(h1, z2).mean() + self.criterion(h2, z1).mean()) * 0.5
            # cos_sim = - (self.criterion(h1, z2.detach()).mean() + self.criterion(h2, z1.detach()).mean()) * 0.5

            # compare the similarity between the actual embeddings
            cos_sim_embed = self.criterion(z1, z2).mean()

            # determine the std across all embeddings in the batch
            z_std = torch.nn.functional.normalize(z1, dim=-1).std(dim=0).mean() * z1.shape[-1]**0.5 
        else:
            cos_sim = torch.tensor([0.0], dtype=torch.float32, device=imgs.device)
            cos_sim_embed = torch.tensor([0.0], dtype=torch.float32, device=imgs.device)
            z_std = torch.tensor([0.0], dtype=torch.float32, device=imgs.device)

        return loss, ncc, cos_sim, cos_sim_embed, z_std, imgs_hat, mask, latent


def otis_baseDeep_patchX_dec160d4b(**kwargs):    # nb_params: 7.58M encoder, 1.70M decoder
    model = OTiS(
        embed_dim=192, depth=12, num_heads=3,                               # dim=64 per head
        decoder_embed_dim=160, decoder_depth=4, decoder_num_heads=5,        # dim=32 per head
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_baseDeep_patchX_dec128d2b(**kwargs):    # nb_params: 7.58M encoder, 0.57M decoder
    model = OTiS(
        embed_dim=192, depth=12, num_heads=3,                               # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,        # dim=32 per head
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_largeDeep_patchX_dec160d4b(**kwargs):   # nb_params: 43.52M encoder, 1.74M decoder
    model = OTiS(
        embed_dim=384, depth=18, num_heads=6,                               # dim=64 per head
        decoder_embed_dim=160, decoder_depth=4, decoder_num_heads=5,        # dim=32 per head
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_largeDeep_patchX_dec128d2b(**kwargs):   # nb_params: 43.52M encoder, 0.60M decoder
    model = OTiS(
        embed_dim=384, depth=18, num_heads=6,                               # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,        # dim=32 per head
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_hugeDeep_patchX_dec160d4b(**kwargs):    # nb_params: 130.81M encoder, 1.78M decoder
    model = OTiS(
        embed_dim=576, depth=24, num_heads=8,                               # dim=72 per head
        decoder_embed_dim=160, decoder_depth=4, decoder_num_heads=5,        # dim=32 per head
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_hugeDeep_patchX_dec128d2b(**kwargs):    # nb_params: 130.81M encoder, 0.63M decoder
    model = OTiS(
        embed_dim=576, depth=24, num_heads=8,                               # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,        # dim=32 per head
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

# set recommended archs
otis_baseDeep_dec160d4b_patchX = otis_baseDeep_patchX_dec160d4b  # decoder: 160 dim, 4 blocks
otis_baseDeep_dec128d2b_patchX = otis_baseDeep_patchX_dec128d2b  # decoder: 128 dim, 2 blocks

otis_largeDeep_dec160d4b_patchX = otis_largeDeep_patchX_dec160d4b  # decoder: 160 dim, 4 blocks
otis_largeDeep_dec128d2b_patchX = otis_largeDeep_patchX_dec128d2b  # decoder: 128 dim, 2 blocks

otis_hugeDeep_dec160d4b_patchX = otis_hugeDeep_patchX_dec160d4b  # decoder: 160 dim, 4 blocks
otis_hugeDeep_dec128d2b_patchX = otis_hugeDeep_patchX_dec128d2b  # decoder: 128 dim, 2 blocks