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
        # self.input_size = input_size
        # self.patch_size = patch_size
        # self.grid_size = (input_size[1] // patch_size[0], input_size[2] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.input_size[1], f"Input image height ({H}) doesn't match model ({self.input_size[1]})."
        # assert W == self.input_size[2], f"Input image width ({W}) doesn't match model ({self.input_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BDH'W' -> BND
        x = self.norm(x)
        return x