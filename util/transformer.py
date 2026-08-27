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
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Multi-head self-attention.

    The released OTiS checkpoints were trained with an inner
    ``nn.MultiheadAttention``, so the trained function is

        qkv -> mha.in_proj -> attention -> mha.out_proj -> proj

    -- note the *second* projection inside ``mha``. ``self.mha`` is kept as a
    submodule so those checkpoints keep loading unchanged, but the attention is
    computed with the fused SDPA kernel instead of by calling
    ``nn.MultiheadAttention`` (which returns attention weights by default and so
    cannot take its own fast path).

    In eval mode the two consecutive projections are *folded* into one, which is
    exact -- both are affine, so their composition is affine -- and brings the
    cost down to that of a plain SDPA block. Training keeps the unfolded path so
    gradients still reach the original parameters.

    Masking: the pre-dc7c81f implementation passed ``1 - attn_mask`` as a
    *float* ``key_padding_mask``. PyTorch treats a float mask as **additive**, so
    padded positions received ``+1.0`` on their scores rather than ``-inf`` and
    were still attended to. This uses a boolean mask, so padded patches are
    genuinely excluded. For an unpadded input (``attn_mask`` all ones) the two
    agree exactly, so pre-trained weights are unaffected; they differ only where
    padding is present, and there the old behaviour was a bug.

    SDPA does not expose attention weights, so ``attn_map`` stays None.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_map = None  # SDPA does not expose attention weights
        self._fused = None    # (cache key, folded weights), eval only

    def _fold(self):
        """Fold (qkv, mha.in_proj) and (mha.out_proj, proj) into single affine
        maps. Cached, and re-derived whenever a parameter changes: the key
        carries each tensor's ``_version`` (which load_state_dict and optimiser
        steps bump) plus dtype/device (which ``.to()`` changes without bumping
        the version)."""
        W_in, b_in = self.mha.in_proj_weight, self.mha.in_proj_bias
        W_out, b_out = self.mha.out_proj.weight, self.mha.out_proj.bias
        W_qkv, b_qkv = self.qkv.weight, self.qkv.bias
        W_p, b_p = self.proj.weight, self.proj.bias

        tensors = [W_in, b_in, W_out, b_out, W_qkv, b_qkv, W_p, b_p]
        key = (tuple(-1 if t is None else t._version for t in tensors),
               W_qkv.dtype, W_qkv.device)
        if self._fused is not None and self._fused[0] == key:
            return self._fused[1]

        D = W_qkv.shape[1]
        with torch.no_grad():
            w_parts, b_parts = [], []
            for i, (w_i, b_i) in enumerate(zip(W_in.chunk(3, dim=0),
                                               (b_in.chunk(3, dim=0) if b_in is not None else (None,) * 3))):
                w_parts.append(w_i @ W_qkv[i * D:(i + 1) * D])
                b = None if b_qkv is None else w_i @ b_qkv[i * D:(i + 1) * D]
                if b_i is not None:
                    b = b_i if b is None else b + b_i
                b_parts.append(b)
            fused_w = torch.cat(w_parts, dim=0)
            fused_b = None if any(b is None for b in b_parts) else torch.cat(b_parts, dim=0)

            out_w = W_p @ W_out
            out_b = None if b_out is None else W_p @ b_out
            if b_p is not None:
                out_b = b_p if out_b is None else out_b + b_p

        folded = (fused_w, fused_b, out_w, out_b)
        self._fused = (key, folded)
        return folded

    def train(self, mode=True):
        # Only on an actual mode change. Clearing unconditionally would also fire
        # on the no-op ``train(False)`` that torch.onnx.export issues, which would
        # push the fold inside the traced graph instead of leaving it a constant.
        if mode != self.training:
            self._fused = None
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        self._fused = None          # .to() / .cuda() / .half()
        return super()._apply(*args, **kwargs)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape # D = embed_dim

        if self.training:
            # unfolded, so gradients reach qkv / mha.in_proj / mha.out_proj / proj
            qkv = self.qkv(x).reshape(B, N, 3, D).permute(2, 0, 1, 3) # (QKV, B, N, D)
            q, k, v = qkv.unbind(0)
            w_q, w_k, w_v = self.mha.in_proj_weight.chunk(3, dim=0)
            if self.mha.in_proj_bias is not None:
                b_q, b_k, b_v = self.mha.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None
            q = F.linear(q, w_q, b_q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = F.linear(k, w_k, b_k).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = F.linear(v, w_v, b_v).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            fused_w, fused_b, _, _ = self._fold()
            qkv = F.linear(x, fused_w, fused_b).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (B, num_heads, N, head_dim)

        if attn_mask is not None:
            # attn_mask: (B, N) with 1=keep, 0=pad -> (B, 1, 1, N) boolean, where
            # True means "attend to this position" (SDPA mask semantic)
            attn_mask = attn_mask.bool().unsqueeze(1).unsqueeze(2)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )  # (B, num_heads, N, head_dim)

        # (B, N, D)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        if self.training:
            x = self.proj(self.mha.out_proj(attn))
        else:
            _, _, out_w, out_b = self._fold()
            x = F.linear(attn, out_w, out_b)

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
            # attn_mask: (B, N) with 1=keep, 0=pad. key_padding_mask must be
            # boolean with True meaning "ignore": a float mask is treated as
            # *additive*, so 1 - attn_mask would add +1.0 to padded scores
            # instead of masking them and padded patches would still be
            # attended to.
            attn_mask = ~attn_mask.bool()
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