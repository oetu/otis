# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')           # prevents tkinter error
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

import numpy as np

import wandb


def plot_attention(original_signal, attention_map, sample_idx, head_idx=0):
    """
    Plot the attention as a heatmap over the signal.
    All channels of the signal for one head.

    :input:
    original_signal (B, C, C_sig, T_sig)
    attention_map (B, Heads, C_sig*N_(C_sig), C_sig*N_(C_sig))
    """
    B, C, C_sig, T_sig = original_signal.shape
    B, Heads, N, N = attention_map.shape
    NpC = int((N-1) / C_sig)

    # Only for nice visualization 
    original_signal = (original_signal + 0.5 * abs(original_signal.min()))

    # fig, axes = plt.subplots(nrows=C_sig, sharex=True, figsize=(16, 8))
    fig, axes = plt.subplots(nrows=C_sig, figsize=(16, 8))

    for channel in range(C_sig):
        # Retrieve the attention of the channel
        # (B, Heads, N_(C_sig), N-1)
        attention_map_ch = attention_map[:, :, 1+channel*NpC:1+(channel+1)*NpC, 1:]  # Ignore the cls token

        # Average the attention of all tokens to this channel
        # (B, Heads, N_(C_sig))
        attention_map_ch = attention_map_ch.mean(dim=-1)

        # Normalize
        attention_map_ch = (attention_map_ch - attention_map_ch.min()) / (attention_map_ch.max() - attention_map_ch.min())

        # Interpolate to the original signal length
        # (B, Heads, T_sig)
        attention_map_ch = F.interpolate(attention_map_ch, size=T_sig, mode='linear')

        # Get the original signal
        # (T_sig)
        original_signal_ch = original_signal[sample_idx, 0, channel].cpu()
        # (Heads, T_sig)
        attention_map_ch = attention_map_ch[sample_idx].cpu()

        # define the axis
        t = np.arange(T_sig)
        vertices = np.column_stack([t, original_signal_ch])

        axes[channel].plot(t, original_signal_ch, color='white', linewidth=2)

        # Use LineCollection to draw varying colors
        segments = np.stack([vertices[:-1], vertices[1:]], axis=1)
        lc = LineCollection(segments, cmap='YlGnBu', norm=plt.Normalize(attention_map_ch[head_idx].min(), attention_map_ch[head_idx].max()), linewidth=1)
        lc.set_array(attention_map_ch[head_idx])
        axes[channel].add_collection(lc)

        axes[channel].set_ylim(original_signal_ch.min(), original_signal_ch.max())
        # axes[channel].set_title(f'Channel {channel + 1}')
        if channel < C_sig-1: axes[channel].set_xticks([])

        axes[channel].spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    # Add color bar at the very right
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(lc, cax=cbar_ax)
    cbar.set_label('Attention Weights')

    # Remove y labels of all subplots
    [ax.yaxis.set_visible(False) for ax in axes.ravel()]

    plt.tight_layout()

    fig_attn = wandb.Image(fig)
    plt.close('all')

    return fig_attn