# Copyright (c) Oezguen Turgut.
# All rights reserved.
"""
Track per-layer L2 norms of patch tokens during pre-training.

Inspired by Figure 4(a) of "Vision Transformers Need Registers" (Darcet et al.,
``rebuttal/pdfs/registers.pdf``), which visualises how the L2 norm of patch
tokens evolves across the layers of a pre-trained ViT and shows that a small
fraction of tokens grow into high-norm "artifact" outliers.

Here we mirror that diagnostic for OTIS pre-training. For a fixed subset of
random pre-training samples (4096 by default) we run the encoder and decoder
with **all patches visible** (no masking, no mask tokens) and record the L2
norm of every non-padding patch token at every layer boundary. Note this
intentionally diverges from the pre-training compute path, which only feeds
visible patches through the encoder — we want a clean per-layer diagnostic on
the same set of tokens for every sample / step. The CLS token is excluded;
only output / patch tokens are tracked.

The result is logged to wandb as a single figure with two panels (encoder and
decoder), each showing percentile bands (p50 / p90 / p99 / max) and the mean
of the patch-token L2 norm distribution per layer.
"""

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # prevents tkinter error
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import wandb


def _flatten_valid(norms_per_layer, valid_mask):
    """Flatten per-batch (B, N) tensors over the valid (non-padding) tokens.

    norms_per_layer: list of (B, N) tensors
    valid_mask:      (B, N) 0/1 tensor

    Returns: list of 1D float CPU tensors, one per layer.
    """
    flat_mask = valid_mask.bool()
    return [n.float()[flat_mask].detach().cpu() for n in norms_per_layer]


@torch.no_grad()
def track_token_l2_norms(model, data_loader, device, *, num_samples: int = 4096,
                         mode: str = "both"):
    """Run a fixed dataloader through the model and collect per-layer L2 norms
    of the (non-padding) patch tokens for the encoder and/or decoder.

    All patches are fed through the encoder (no masking) and the full encoder
    output is then fed through the decoder (no mask tokens), so the per-layer
    norms are computed on the same set of tokens for every sample.

    The model is temporarily switched to eval mode and restored afterwards.

    Args:
        mode: one of ``"encoder"``, ``"decoder"``, or ``"both"`` — which
              side(s) to record. The encoder forward pass is always run
              because the decoder needs its latent.

    Returns:
        encoder_norms: list of 1D float tensors, one per encoder layer
                       (length = depth + 2: pre-block-0, post each block,
                        post final norm). ``None`` when ``mode="decoder"``.
        decoder_norms: list of 1D float tensors, one per decoder layer.
                       ``None`` when ``mode="encoder"`` or when the model
                       uses an MLP head instead of a decoder.
        encoder_embed_dim: int, the encoder hidden dimension.
        decoder_embed_dim: int or ``None``, the decoder hidden dimension.
    """
    if mode not in ("encoder", "decoder", "both"):
        raise ValueError(f"mode must be 'encoder', 'decoder', or 'both', got {mode!r}")

    underlying = model.module if hasattr(model, "module") else model
    has_decoder = getattr(underlying, "output_projection", "decoder") == "decoder"
    if mode in ("decoder", "both") and not has_decoder:
        raise ValueError("model has no decoder; cannot track decoder norms")

    encoder_embed_dim = int(underlying.cls_token.shape[-1])
    decoder_embed_dim = int(underlying.mask_token.shape[-1]) if has_decoder else None

    track_encoder = mode in ("encoder", "both")
    track_decoder = mode in ("decoder", "both")

    was_training = model.training
    model.eval()

    encoder_acc = None
    decoder_acc = None
    seen = 0
    try:
        for samples, attn_mask in data_loader:
            if seen >= num_samples:
                break
            samples = samples.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # The encoder forward is always run — even when only the
                # decoder is reported — because the decoder needs ``latent``.
                enc_norms, enc_mask, latent = \
                    underlying.forward_encoder_collect_patch_norms(samples, attn_mask)
                if track_decoder:
                    dec_norms, dec_mask = \
                        underlying.forward_decoder_collect_patch_norms(latent, attn_mask)

            if track_encoder:
                enc_layer_flat = _flatten_valid(enc_norms, enc_mask)
                if encoder_acc is None:
                    encoder_acc = [[] for _ in enc_layer_flat]
                for i, t in enumerate(enc_layer_flat):
                    encoder_acc[i].append(t)

            if track_decoder:
                dec_layer_flat = _flatten_valid(dec_norms, dec_mask)
                if decoder_acc is None:
                    decoder_acc = [[] for _ in dec_layer_flat]
                for i, t in enumerate(dec_layer_flat):
                    decoder_acc[i].append(t)

            seen += samples.shape[0]
    finally:
        if was_training:
            model.train()

    encoder_norms = [torch.cat(layer, dim=0) for layer in encoder_acc] if encoder_acc else None
    decoder_norms = [torch.cat(layer, dim=0) for layer in decoder_acc] if decoder_acc else None
    return encoder_norms, decoder_norms, encoder_embed_dim, decoder_embed_dim


def _plot_panel(ax, norms_per_layer, title, *, embed_dim: int = None,
                n_bins: int = 80, cmap: str = "magma"):
    """Render a per-layer density heatmap of token L2 norms.

    For each layer, the L2 norms of every (non-padding) patch token are
    binned into ``n_bins`` shared bins covering the global value range and
    converted to a probability density (matches the convention in Darcet et
    al. 2024, Fig. 4a). The resulting (n_bins × n_layers) matrix is shown as
    a 2D heatmap with a log-scaled colour axis so both the dense bulk and
    rare outlier tokens remain visible. No summary statistics are drawn.

    If ``embed_dim`` is provided, the expected post-LayerNorm L2 norm
    (≈ √d for unit-variance LayerNorm outputs) is appended to the panel
    title for quick visual calibration.
    """
    n_layers = len(norms_per_layer)
    all_vals = torch.cat(norms_per_layer).numpy()
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    hist = np.zeros((n_bins, n_layers), dtype=np.float64)
    for i, t in enumerate(norms_per_layer):
        density, _ = np.histogram(t.numpy(), bins=bin_edges, density=True)
        hist[:, i] = density

    # Mask zero-density cells so they show as the colormap "bad" colour
    # rather than the lowest value, and use log colour scaling so a few
    # outlier tokens are still visible alongside the dense bulk.
    hist_masked = np.ma.masked_where(hist <= 0, hist)
    nonzero = hist[hist > 0]
    color_vmin = float(nonzero.min()) if nonzero.size else 1e-4
    color_vmax = float(hist.max()) if hist.max() > 0 else 1.0
    color_norm = LogNorm(vmin=color_vmin, vmax=color_vmax)

    x_edges = np.arange(n_layers + 1) - 0.5
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    mesh = ax.pcolormesh(x_edges, bin_edges, hist_masked,
                         norm=color_norm, cmap=cmap_obj, shading="flat")

    if embed_dim is not None:
        expected = float(np.sqrt(embed_dim))
        title = f"{title} (d={embed_dim}, √d≈{expected:.2f})"
    ax.set_title(title)
    ax.set_xlabel("Layer (0 = pre-block, last = post-norm)")
    ax.set_ylabel("Patch-token L2 norm")
    ax.set_xticks(np.arange(n_layers))
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(vmin, vmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("density (log scale)")


def build_token_l2_norm_figure(encoder_norms=None, decoder_norms=None,
                               encoder_embed_dim=None, decoder_embed_dim=None):
    """Build a 1- or 2-panel matplotlib figure of per-layer patch-token L2
    norm statistics. Returns the ``Figure``; the caller is responsible for
    closing it. Either ``encoder_norms`` or ``decoder_norms`` (or both) may
    be passed; sides set to ``None`` are simply omitted.

    Pass ``encoder_embed_dim`` / ``decoder_embed_dim`` to have the expected
    post-LayerNorm L2 norm (≈ √d) appended to each panel title.
    """
    panels = []
    if encoder_norms is not None:
        panels.append(("Encoder", encoder_norms, encoder_embed_dim))
    if decoder_norms is not None:
        panels.append(("Decoder", decoder_norms, decoder_embed_dim))
    if not panels:
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
    for ax, (label, norms, dim) in zip(axes[0], panels):
        _plot_panel(ax, norms,
                    f"{label} — patch-token L2 norm per layer",
                    embed_dim=dim)

    fig.tight_layout()
    return fig


def plot_token_l2_norms(encoder_norms=None, decoder_norms=None,
                        encoder_embed_dim=None, decoder_embed_dim=None):
    """Build a 1- or 2-panel figure of per-layer patch-token L2 norm
    statistics. Returns a ``wandb.Image``."""
    plt.close("all")
    fig = build_token_l2_norm_figure(encoder_norms, decoder_norms,
                                     encoder_embed_dim=encoder_embed_dim,
                                     decoder_embed_dim=decoder_embed_dim)
    if fig is None:
        return None
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _plot_evolution_panel(ax, history, title, *, embed_dim: int = None,
                          n_bins: int = 100, cmap: str = "magma"):
    """Render last-block (pre-final-norm) patch-token L2 norm distributions as
    a 2D density heatmap with one column per recorded epoch. Mirrors Figure
    4(b) of Darcet et al., "Vision Transformers Need Registers": x-axis is
    training progress (epoch), y-axis is patch-token L2 norm, colour is
    per-column density on a log scale so the gradual emergence of a few
    high-norm outliers is visible alongside the dense bulk.

    Visual style matches :func:`_plot_panel` (the per-layer plot), differing
    only in the x-axis quantity (epoch instead of layer index).
    """
    epochs = np.asarray([e for e, _ in history], dtype=np.float64)
    norms_per_epoch = [t.numpy() for _, t in history]

    all_vals = np.concatenate(norms_per_epoch)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    n_epochs = len(epochs)
    hist = np.zeros((n_bins, n_epochs), dtype=np.float64)
    for i, vals in enumerate(norms_per_epoch):
        density, _ = np.histogram(vals, bins=bin_edges, density=True)
        hist[:, i] = density

    hist_masked = np.ma.masked_where(hist <= 0, hist)
    nonzero = hist[hist > 0]
    color_vmin = float(nonzero.min()) if nonzero.size else 1e-4
    color_vmax = float(hist.max()) if hist.max() > 0 else 1.0
    color_norm = LogNorm(vmin=color_vmin, vmax=color_vmax)

    if n_epochs == 1:
        x_edges = np.array([epochs[0] - 0.5, epochs[0] + 0.5])
    else:
        mids = 0.5 * (epochs[1:] + epochs[:-1])
        first = epochs[0] - 0.5 * (epochs[1] - epochs[0])
        last = epochs[-1] + 0.5 * (epochs[-1] - epochs[-2])
        x_edges = np.concatenate([[first], mids, [last]])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    mesh = ax.pcolormesh(x_edges, bin_edges, hist_masked,
                         norm=color_norm, cmap=cmap_obj, shading="flat")

    if embed_dim is not None:
        expected = float(np.sqrt(embed_dim))
        title = f"{title} (d={embed_dim}, √d≈{expected:.2f})"
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Patch-token L2 norm (last block, pre-final-norm)")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(vmin, vmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("density (log scale)")


def build_last_layer_norm_evolution_figure(encoder_history=None,
                                           decoder_history=None,
                                           encoder_embed_dim=None,
                                           decoder_embed_dim=None):
    """Build a 1- or 2-panel matplotlib figure of last-layer (pre-final-norm)
    patch-token L2 norm distributions across pre-training epochs.

    Each ``*_history`` argument is a list of ``(epoch, norms_1d_tensor)``
    pairs. Returns the ``Figure``; the caller is responsible for closing it.
    """
    panels = []
    if encoder_history:
        panels.append(("Encoder", encoder_history, encoder_embed_dim))
    if decoder_history:
        panels.append(("Decoder", decoder_history, decoder_embed_dim))
    if not panels:
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
    for ax, (label, history, dim) in zip(axes[0], panels):
        _plot_evolution_panel(ax, history,
                              f"{label} — last-layer patch-token L2 norm over training",
                              embed_dim=dim)

    fig.tight_layout()
    return fig


def plot_last_layer_norm_evolution(encoder_history=None, decoder_history=None,
                                   encoder_embed_dim=None, decoder_embed_dim=None):
    """Wandb-friendly wrapper around
    :func:`build_last_layer_norm_evolution_figure`. Returns a ``wandb.Image``."""
    plt.close("all")
    fig = build_last_layer_norm_evolution_figure(
        encoder_history=encoder_history,
        decoder_history=decoder_history,
        encoder_embed_dim=encoder_embed_dim,
        decoder_embed_dim=decoder_embed_dim)
    if fig is None:
        return None
    img = wandb.Image(fig)
    plt.close(fig)
    return img