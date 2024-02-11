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

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_

from util.patch_embed import PatchEmbed
from util.pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
import util.statistics as statistics


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, modalities, input_channels=1, patch_size=(1, 100),
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 ncc_weight:float=0.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_channels, patch_size, embed_dim, flatten=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.grid_size = {}
        for modality, input_size in modalities:
            grid_size = (input_size[1] // patch_size[0], input_size[2] // patch_size[1])
            self.grid_size.update( {modality: grid_size} )

        assert embed_dim % 2 == 0
        max_num_patches_x = max([v[1] for k, v in self.grid_size.items()])
        self.pos_embed_x = nn.Parameter(torch.zeros(1, max_num_patches_x + 1, embed_dim // 2), requires_grad=False) # +1 cls embed

        total_num_embeddings_y = sum([v[0] for k, v in self.grid_size.items()])
        self.pos_embed_y = nn.Embedding(total_num_embeddings_y + 1, embed_dim // 2, padding_idx=0) # +1 padding embed

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # modify the attention operation to consider attention masks
        for block in self.blocks:
            block.forward = self._block_forward_wrapper(block)
            block.attn.forward = self._attention_forward_wrapper(block.attn)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        assert decoder_embed_dim % 2 == 0
        self.decoder_pos_embed_x = nn.Parameter(torch.zeros(1, max_num_patches_x + 1, decoder_embed_dim // 2), requires_grad=False) # +1 cls embed
        # self.decoder_pos_embed_y = nn.Embedding(total_num_embeddings_y + 1, decoder_embed_dim // 2, padding_idx=0) # +1 padding embed
        self.decoder_pos_embed_y = nn.Linear(embed_dim // 2, decoder_embed_dim // 2)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, act_layer=nn.GELU, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * input_channels, bias=True) # decoder to patch

        # modify the attention operation to consider attention masks
        for block in self.decoder_blocks:
            block.forward = self._block_forward_wrapper(block)
            block.attn.forward = self._attention_forward_wrapper(block.attn)
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
        
        self.ncc_weight = ncc_weight

        self.initialize_weights()

    def _block_forward_wrapper(self, block_obj):
        """
        Modified version of def forward() of class Block() in timm.models.vision_transformer
        """
        def my_forward(x, attn_mask=None):
            if attn_mask is None:
                x = x + block_obj.drop_path1(block_obj.ls1(block_obj.attn(block_obj.norm1(x))))
            else:
                x = x + block_obj.drop_path1(block_obj.ls1(block_obj.attn(block_obj.norm1(x), attn_mask)))
            x = x + block_obj.drop_path2(block_obj.ls2(block_obj.mlp(block_obj.norm2(x))))
            return x

        return my_forward

    def _attention_forward_wrapper(self, attn_obj):
        """
        Modified version of def forward() of class Attention() in timm.models.vision_transformer
        """
        def my_forward(x, attn_mask=None):
            B, N, C = x.shape # C = embed_dim
            # (3, B, Heads, N, head_dim)
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            # TODO: implement the pytorch multiheadattention, it is more efficient

            # (B, Heads, N, N)
            attn = (q @ k.transpose(-2, -1)) * attn_obj.scale

            if attn_mask is not None:
                # (B, 1, N)
                attn_mask_batch = attn_mask.unsqueeze(1).clone()
                # (B, 1, N, N)
                attn_mask_batch = torch.einsum("bhn,bhl->bhnl", attn_mask_batch, attn_mask_batch)
                # (B, Heads, N, N)
                attn_mask_batch = attn_mask_batch.expand(-1, attn_obj.num_heads, -1, -1)

                # (B, Heads, N, N)
                attn[attn_mask_batch==0] = -torch.inf # will be zero after softmax, exp(-inf)=0

            # attn = attn.softmax(dim=-1) # returns nan if there is rows with -inf only :S
            # implement a modified softmax version that includes a small positive term in the denominator
            attn = torch.exp(attn - attn.max()) # avoid numerical instability by subtracting the maximum value
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

            attn = attn_obj.attn_drop(attn)

            # (B, Heads, N, N)
            attn_obj.attn_map = attn # this was added 

            # (B, N, Heads*head_dim)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_obj.proj(x)
            x = attn_obj.proj_drop(x)
            return x
        
        return my_forward

    def initialize_weights(self):
        # initialize learnable pos_embed for the vertical axis
        _pos_embed_y = torch.nn.Parameter(torch.randn(self.pos_embed_y.num_embeddings-1, 
                                                      self.pos_embed_y.embedding_dim) * .02)
        trunc_normal_(_pos_embed_y, std=.02)
        with torch.no_grad():
            self.pos_embed_y.weight[1:] = _pos_embed_y

        # _decoder_pos_embed_y = torch.nn.Parameter(torch.randn(self.decoder_pos_embed_y.num_embeddings-1, 
        #                                                       self.decoder_pos_embed_y.embedding_dim) * .02)
        # trunc_normal_(_decoder_pos_embed_y, std=.02)
        # with torch.no_grad():
        #     self.decoder_pos_embed_y.weight[1:] = _decoder_pos_embed_y
                
        # initialize (and freeze) pos_embed for the horizontal axis by sin-cos embedding
        _pos_embed_x = get_1d_sincos_pos_embed(self.pos_embed_x.shape[-1], 
                                               self.pos_embed_x.shape[-2]-1, 
                                               cls_token=True)
        self.pos_embed_x.data.copy_(torch.from_numpy(_pos_embed_x).float().unsqueeze(0))

        _decoder_pos_embed_x = get_1d_sincos_pos_embed(self.decoder_pos_embed_x.shape[-1], 
                                                       self.decoder_pos_embed_x.shape[-2]-1, 
                                                       cls_token=True)
        self.decoder_pos_embed_x.data.copy_(torch.from_numpy(_decoder_pos_embed_x).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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
        x = self.patch_embed(x) # TODO: maybe mask this one as well

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
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # (1, 1, D)
        cls_token = self.cls_token + pos_embed_x[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # (B, 1+N', D)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # (B, N)
        attn_mask_visible_patches = attn_mask.flatten(1)[mask==0].view(attn_mask.shape[0], -1)
        # (B, 1+N'), add cls token to attn mask
        attn_mask_visible_patches = torch.cat((torch.ones(size=(attn_mask.shape[0], 1), device=x.device), attn_mask_visible_patches), dim=1)

        for blk in self.blocks:
            x = blk(x, attn_mask_visible_patches)

        # (B, 1+N', D)        
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_all_patches(self, x,  pos_embed_y):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            pos_embed_y: (B, C', T'), with N=C'*T' embedding ids

        output:
            x: (B, 1+N', D), with 1 cls token + N' visible patches
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
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
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
        # decoder_pos_embed_y_batch = self.decoder_pos_embed_y(pos_embed_y)
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

        # (B, 1+N, D_dec)
        x = self.decoder_norm(x)

        # predictor projection
        # (B, 1+N, p*q*input_channels)
        x = self.decoder_pred(x)

        # remove cls token
        # (B, N, p*q*input_channels)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, attn_mask, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*q*C]
        attn_mask: [N, C', T'], with C'=H/p, T'=W/q and C'*T'=L
        mask: [N, L], 0 is keep, 1 is remove
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

        # mean over last dim (for normalization) does not require consideration of the attention mask
        ncc = statistics.ncc(imgs, imgs_hat)

        # loss_patches = (loss * mask).sum() / mask.sum()  # TODO: mean loss on removed patches
        loss = loss.sum() / (torch.sum(attn_mask) + 1e-9)  # neglects the paddings

        return (1-self.ncc_weight)*loss + self.ncc_weight*(1-ncc)

    def forward(self, imgs, attn_mask, pos_embed_y, mask_ratio=0.75):
        """
        imgs: [N, C, H, W]
        attn_mask: [N, C', T'], with C'*T'=L and C'=H/p, T'=W/q
        pos_embed_y: [N, C', T'], with C'*T'=L and C'=H/p, T'=W/q 
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, attn_mask, pos_embed_y, mask_ratio)
        pred = self.forward_decoder(latent, attn_mask, pos_embed_y, ids_restore)  # [N, L, p*q*C]
        loss = self.forward_loss(imgs, pred, attn_mask, mask)

        # orig_patched = self.patchify(imgs)
        # orig_masked_unpatched = self.unpatchify(orig_patched*(1-mask).unsqueeze(dim=-1), attn_mask)
        imgs_hat = self.unpatchify(pred, attn_mask)
        imgs_hat_masked = self.unpatchify(pred*(1-mask).unsqueeze(dim=-1), attn_mask)

        # contrastive part
        latent2, _, _ = self.forward_encoder(imgs, attn_mask, pos_embed_y, mask_ratio)

        attn_mask_visible_patches = attn_mask.flatten(1)[mask==0].view(attn_mask.shape[0], -1)
        z1 = statistics.masked_mean(latent[:, 1:, ...], attn_mask_visible_patches, dim=1)     # global average pooling
        z2 = statistics.masked_mean(latent2[:, 1:, ...], attn_mask_visible_patches, dim=1)    # global average pooling
        
        p1 = self.projector(z1)
        p2 = self.projector(z2)

        h1 = self.predictor(p1)
        h2 = self.predictor(p2)

        # loss_cos = - (self.criterion(h1, p2).mean() + self.criterion(h2, p1).mean()) * 0.5
        loss_cos = - (self.criterion(h1, p2.detach()).mean() + self.criterion(h2, p1.detach()).mean()) * 0.5
        # loss_cos = - (self.criterion(h1, z2).mean() + self.criterion(h2, z1).mean()) * 0.5
        # loss_cos = - (self.criterion(h1, z2.detach()).mean() + self.criterion(h2, z1.detach()).mean()) * 0.5

        # compare the similarity between the actual embeddings
        cos_embed = self.criterion(z1, z2).mean()

        # determine the std across all embeddings in the batch
        z_std = torch.nn.functional.normalize(z1, dim=-1).std(dim=0).mean() * z1.shape[-1]**0.5 

        return loss, loss_cos, cos_embed, z_std, imgs_hat, imgs_hat_masked


def mae_vit_pluto_patchX_dec192d2b(**kwargs): # nb_params: 1.61M encoder, 0.37M decoder
    model = MaskedAutoencoderViT(
        embed_dim=256, depth=2, num_heads=8, # dim=32 per head
        decoder_embed_dim=160, decoder_depth=1, decoder_num_heads=8, # dim=20 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny_patchX_dec256d2b(**kwargs): # nb_params: 5.36M encoder, 1.7M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=3, num_heads=6, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny2_patchX_dec256d2b(**kwargs): # nb_params: 5.36M encoder, 1.7M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=3, num_heads=6, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tinyDeep_patchX_dec256d2b(**kwargs): # nb_params: 21.34M encoder, 1.8M decoder
    model = MaskedAutoencoderViT(
        embed_dim=192, depth=12, num_heads=3, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tinyDeep2_patchX_dec256d2b(**kwargs): # nb_params: 21.34M encoder, 1.8M decoder
    model = MaskedAutoencoderViT(
        embed_dim=192, depth=12, num_heads=3, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_small_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=512, depth=4, num_heads=8, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_small2_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=512, depth=4, num_heads=8, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_smallDeep_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_smallDeep2_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_medium_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=640, depth=5, num_heads=10, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_medium2_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=640, depth=5, num_heads=10, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_mediumDeep_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=576, depth=12, num_heads=8, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_mediumDeep2_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=576, depth=12, num_heads=8, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patchX_dec256d4b(**kwargs): # nb_params: 53.91M encoder, 1.84M decoder
    model = MaskedAutoencoderViT(
        embed_dim=864, depth=6, num_heads=12, # dim=72 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patchX_dec512d8b(**kwargs): # 86M params in total
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12, # dim=64 per head
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patchX_dec512d8b(**kwargs): # 307M params in total
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16, # dim=64 per head
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patchX_dec512d8b(**kwargs): # 632M params in total
    model = MaskedAutoencoderViT(
        embed_dim=1280, depth=32, num_heads=16, # dim=80 per head
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_pluto_patchX = mae_vit_pluto_patchX_dec192d2b  # decoder: 256 dim, 2 blocks

mae_vit_tiny_patchX = mae_vit_tiny_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_tiny2_patchX = mae_vit_tiny2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks
mae_vit_tinyDeep_patchX = mae_vit_tinyDeep_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_tinyDeep2_patchX = mae_vit_tinyDeep2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks

mae_vit_small_patchX = mae_vit_small_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_small2_patchX = mae_vit_small2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks
mae_vit_smallDeep_patchX = mae_vit_smallDeep_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_smallDeep2_patchX = mae_vit_smallDeep2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks

mae_vit_medium_patchX = mae_vit_medium_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_medium2_patchX = mae_vit_medium2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks
mae_vit_mediumDeep_patchX = mae_vit_mediumDeep_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_mediumDeep2_patchX = mae_vit_mediumDeep2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks

mae_vit_large_patchX = mae_vit_large_patchX_dec256d4b  # decoder: 256 dim, 2 blocks

mae_vit_base = mae_vit_base_patchX_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large = mae_vit_large_patchX_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge = mae_vit_huge_patchX_dec512d8b  # decoder: 512 dim, 8 blocks