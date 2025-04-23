# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DyT:  https://github.com/jiachenzhu/DyT
# --------------------------------------------------------

import torch
import torch.nn as nn


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
        B, N, D = x.shape # D = embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, D).permute(2, 0, 1, 3) # (QKV, B, N, D)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (B, N, D)

        if attn_mask is not None:
            attn_mask = 1 - attn_mask
        attn, attn_weights = self.mha(q, k, v, key_padding_mask=attn_mask)
        self.attn_map = attn_weights

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x


class TemporalRoPEAttention(nn.Module):
    """
        Apply RoPE positional embeddings only to the temporal dimension
    """
    def __init__(self, dim, num_heads=8, T_max=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.T_max = T_max

        self.compute_rope_embeddings = True
        self.cos = None
        self.sin = None

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_map = None

    def forward(self, x, attn_mask=None, attn_mask_input=None, ids_restore=None, V_input=1, T_input=1):
        """
            x: (B, 1+N_v, D) input signal, with N_v visible tokens and 1 cls token
            attn_mask: (B, 1+N_v)
            attn_mask_input: (B, V, T), with N = V * T = N_v + N_m
            ids_restore: (B, N), 0 is keep, 1 is remove
            V_input: number variate tokens
            T_input: number temporal tokens
        """
        B, N, D = x.shape # D = embed_dim

        self.mask_ratio = 1
        if attn_mask_input is not None:
            _, V, T = attn_mask_input.shape
            self.mask_ratio = 1 - (attn_mask.shape[-1] - 1) / ids_restore.shape[-1] # the cls token is the 1 subtracted
            # print("N:", N, "V:", V, "T:", T, "mask_ratio:", self.mask_ratio)
            expected_N = round(V * T * (1 - self.mask_ratio), 0) + 1
            # print("Expected N:", expected_N)
            assert N == expected_N, f"Sequence length N ({N}) must match V * T * (1 - mask_ratio) + 1 cls token ({expected_N})."
        else:
            V, T  = V_input, T_input
            assert N == V*T + 1, "Sequence length N must match V * T + 1 cls token."

        qkv = self.qkv(x).reshape(B, N, 3, D).permute(2, 0, 1, 3) # (QKV, B, 1+N_v, D)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (B, 1+N_v, D)

        q, k = self.apply_rope_to_time(q, k, V, ids_restore) # (B, N, D)

        if attn_mask is not None:
            attn_mask = 1 - attn_mask
        attn, attn_weights = self.mha(q, k, v, key_padding_mask=attn_mask)
        self.attn_map = attn_weights

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x
    
    def apply_rope_to_time(self, q, k, V, ids_restore=None):
        """
            Apply RoPE embeddings to the query and key vectors, 
            but only along the temporal dimension
        """
        B, N, D = q.shape   # (B, 1+N_v, D), with N_v=V*T*(1-masking_ratio)
        device = q.device

        # Compute RoPE angles
        if self.compute_rope_embeddings == True:
            self.compute_rope_embeddings = False

            theta = 10000.0 ** (-torch.arange(0, self.head_dim, 2, device=device).type_as(q) / self.head_dim) # (head_dim // 2)
            position = torch.arange(1+self.T_max, device=device).type_as(q) # (1+T_max)
            angles = torch.einsum('t,d->td', position, theta)               # (1+T_max, head_dim // 2)

            # TODO
            # Compute rope angles for all heads, but only apply it to the respective temporal heads? 

            # Convert to complex representation
            self.cos, self.sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]   # (1, 1+T_max, 1, head_dim // 2)

        def rotate(x, ids_restore):
            """
                x: (B, 1+N_v, D)
                ids_restore: (B, N)
            """
            x_rot_cls = x[:, :1].view(B, 1, -1, self.head_dim)[..., ::2]    # Select even indices, (B,    1, nb_heads, head_dim // 2)
            x_im_cls = x[:, :1].view(B, 1, -1, self.head_dim)[..., 1::2]    # Select odd indices,  (B,    1, nb_heads, head_dim // 2)

            x_cls_new = torch.cat([x_rot_cls * self.cos[:, :1] - x_im_cls * self.sin[:, :1], 
                                   x_rot_cls * self.sin[:, :1] + x_im_cls * self.cos[:, :1]], dim=-1) # (B, 1, nb_heads, head_dim)
            
            x_rot = x[:, 1:].view(B, N-1, -1, self.head_dim)[..., ::2]      # Select even indices, (B,  N_v, nb_heads, head_dim // 2)
            x_im = x[:, 1:].view(B, N-1, -1, self.head_dim)[..., 1::2]      # Select odd indices,  (B,  N_v, nb_heads, head_dim // 2)

            cos = self.cos[:, 1:].expand(B, -1, -1, -1)    # (B, T_max, 1, head_dim // 2)
            sin = self.sin[:, 1:].expand(B, -1, -1, -1)    # (B, T_max, 1, head_dim // 2)
            if ids_restore is not None:
                ids_shuffle = torch.argsort(ids_restore, dim=1)[:, :N-1] % V        # (B, N_v)
            else:
                ids_shuffle = torch.arange(start=0, end=N-1, device=device).unsqueeze(0).expand(B, -1) % V  # (B, N_v)
            cos = torch.gather(cos, 1, ids_shuffle[..., None, None].repeat(1, 1, 1, self.head_dim // 2))    # (B, N_v, 1, head_dim // 2)
            sin = torch.gather(sin, 1, ids_shuffle[..., None, None].repeat(1, 1, 1, self.head_dim // 2))    # (B, N_v, 1, head_dim // 2)

            x_new = torch.cat([x_rot * cos - x_im * sin, 
                               x_rot * sin + x_im * cos], dim=-1)   # (B,   N_v, nb_heads, head_dim)

            x_new = torch.cat([x_cls_new, x_new], dim=1)            # (B, 1+N_v, nb_heads, head_dim)
            x_new = x_new.view(B, N, -1)                            # (B, 1+N_v, D)

            return x_new
        
        q_rope, k_rope = rotate(q, ids_restore), rotate(k, ids_restore)     # (B, 1+N_v, D)

        return q_rope, k_rope


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias