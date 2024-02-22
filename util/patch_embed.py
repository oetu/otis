""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn


class PatchEmbed(nn.Module):
    """ 
    Multi-Variate Signal to Patch Embedding
    """
    def __init__(self, input_channels=1, patch_size=(1, 100), embed_dim=192, norm_layer=None, flatten=True):
        super().__init__()
        self.flatten = flatten

        self.proj = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act_ft = nn.GELU()
        
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # (B, D, H', W')
        x = self.proj(x)

        # (B, H', W', D)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act_ft(x)

        # (B, D, H', W')
        x = x.permute(0, 3, 1, 2)

        if self.flatten:
            # (B, N, D)
            x = x.flatten(2).transpose(1, 2)  # BDH'W' -> BND

        return x