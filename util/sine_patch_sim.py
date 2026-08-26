# Copyright (c) Oezguen Turgut.
# All rights reserved.
"""Patch-level cosine similarity diagnostic on synthetic sine waves.

The patch slice starts right after the single CLS token; OTIS's
``forward_encoder_all_patches`` takes an explicit ``pos_embed_y`` argument,
which we build locally as raw variate indices 1..V.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # prevents tkinter error
import matplotlib.pyplot as plt

import wandb


def _first_existing(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


SINE_DATA_PATH = _first_existing(
    "/home/oturgut/data/processed/synthetic/synthetic_sine.pt",
    "/vol/miltank/users/tuo/data/synthetic/synthetic_sine.pt",
)
SINE_MOD_DATA_PATH = _first_existing(
    "/home/oturgut/data/processed/synthetic/synthetic_sine_mod.pt",
    "/vol/miltank/users/tuo/data/synthetic/synthetic_sine_mod.pt",
)

DEFAULT_SOURCES = (
    ("sine",     SINE_DATA_PATH,     10,   "peak"),
    ("sine_mod", SINE_MOD_DATA_PATH, 30,   "peak"),
    ("sine_mod", SINE_MOD_DATA_PATH, 1667, "flat"),
)

_SAMPLE_CACHE = {}


def _estimate_period(signal: np.ndarray) -> float:
    from scipy.signal import find_peaks
    prom = max(1e-6, 0.5 * (signal.max() - signal.min()))
    peaks, _ = find_peaks(signal, prominence=prom)
    if len(peaks) >= 2:
        return float(np.mean(np.diff(peaks)))
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(len(signal))
    dom = int(np.argmax(spectrum[1:])) + 1
    return float(1.0 / freqs[dom])


def _normalise_sources(sources):
    out = []
    for entry in sources:
        if len(entry) == 3:
            label, path, idx = entry
            out.append((label, path, idx, "peak"))
        else:
            out.append(tuple(entry))
    return tuple(out)


def _load_samples(time_steps: int, sources=DEFAULT_SOURCES):
    sources = _normalise_sources(sources)
    key = (time_steps, sources)
    if key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[key]

    for _, path, _, _ in sources:
        if not os.path.exists(path):
            _SAMPLE_CACHE[key] = (None, None, None, None)
            return _SAMPLE_CACHE[key]

    raw_cache = {}
    tensors, periods, labels, modes = [], [], [], []
    for label, path, idx, mode in sources:
        if path not in raw_cache:
            raw_cache[path] = torch.load(path, map_location="cpu", weights_only=False)
        _, s = raw_cache[path][idx]
        s = s.float()[..., :time_steps]  # (1, 1, T)
        tensors.append(s)
        periods.append(_estimate_period(s[0, 0].numpy()))
        labels.append(label)
        modes.append(mode)

    _SAMPLE_CACHE[key] = (torch.stack(tensors, dim=0), periods, labels, modes)
    return _SAMPLE_CACHE[key]


def _patch_means(signal: np.ndarray, patch_width: int) -> np.ndarray:
    n_patches = len(signal) // patch_width
    return np.array([
        signal[i * patch_width : (i + 1) * patch_width].mean()
        for i in range(n_patches)
    ])


def _patch_stds(signal: np.ndarray, patch_width: int) -> np.ndarray:
    n_patches = len(signal) // patch_width
    return np.array([
        signal[i * patch_width : (i + 1) * patch_width].std()
        for i in range(n_patches)
    ])


def _draw_signal_with_patch_colors(ax, signal, sims, patch_width, cmap, norm,
                                   highlight_idx=None):
    ax.plot(np.arange(len(signal)), signal, color="black", linewidth=0.6, alpha=0.7)
    n_patches = len(signal) // patch_width
    for i in range(n_patches):
        start = i * patch_width
        ax.axvspan(start, start + patch_width, alpha=0.55,
                   color=cmap(norm(sims[i])), linewidth=0)
    if highlight_idx is not None:
        ax.axvspan(highlight_idx * patch_width, (highlight_idx + 1) * patch_width,
                   facecolor="none", edgecolor="black", linewidth=2.0)
    ax.set_xlim(0, len(signal))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Amplitude")


@torch.no_grad()
def plot_sine_patch_similarity(model, *, time_steps: int = 2400,
                               patch_width: int = 24,
                               sources=DEFAULT_SOURCES,
                               device=None):
    """Return a ``wandb.Image`` with Kx2 patch-cosine-similarity subplots.

    ``None`` if any source file is missing. OTIS layout is ``[CLS, patches]``,
    so patch tokens start at index 1. An explicit ``pos_embed_y`` of raw
    variate indices 1..V is passed to ``forward_encoder_all_patches``.
    """
    samples, periods, labels, modes = _load_samples(time_steps, sources)
    if samples is None:
        return None

    if device is None:
        device = next(model.parameters()).device
    samples_dev = samples.to(device, non_blocking=True)  # (K, 1, 1, T)

    K, _, V, T = samples_dev.shape
    pw = patch_width
    Tp = T // pw
    max_idx = int(model.pos_embed_y.num_embeddings) - 1
    idx = torch.arange(V, device=device, dtype=torch.long) + 1
    idx = idx.clamp(max=max_idx)
    pos_embed_y = idx.view(1, V, 1).expand(K, V, Tp).contiguous()

    was_training = model.training
    model.eval()
    try:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = model.forward_encoder_all_patches(samples_dev, pos_embed_y)  # (K, 1+N, D)
        # drop CLS token, L2-normalise patches
        patch_tokens = F.normalize(emb[:, 1:, :].float(), dim=-1).cpu()  # (K, N, D)
    finally:
        if was_training:
            model.train()

    n_rows = len(samples)
    n_patches = time_steps // patch_width
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    plt.close("all")
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row in range(n_rows):
        label = labels[row]
        mode = modes[row]
        signal = samples[row, 0, 0].numpy()
        period = periods[row]
        n_cycles = max(1, int(round(time_steps / period)))
        patches = patch_tokens[row]  # (N, D)
        amps = _patch_means(signal, patch_width)

        if mode == "peak":
            n_peak = min(n_cycles, n_patches - 1)
            peak_pct = 100.0 * (1.0 - n_peak / n_patches)
            peak_idx = np.where(amps >= np.percentile(amps, peak_pct))[0]
            if len(peak_idx) == 0:
                peak_idx = np.array([int(np.argmax(amps))])
            other_idx = np.array(
                [i for i in range(n_patches) if i not in set(peak_idx.tolist())]
            )

            ref_idx = int(np.argmax(amps))
            ref = patches[ref_idx : ref_idx + 1]
            sim_to_ref = (patches @ ref.T).squeeze(-1).numpy()
            p2p_ref = float(sim_to_ref[peak_idx].mean())
            p2o_ref = float(sim_to_ref[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 0]
            _draw_signal_with_patch_colors(ax, signal, sim_to_ref, patch_width,
                                           cmap, norm, highlight_idx=ref_idx)
            ax.set_title(
                f"{label} period {period:.0f} ({n_cycles} cycles): "
                f"cos-sim to reference peak patch (idx={ref_idx})\n"
                f"avg peak→peak={p2p_ref:.3f}, avg peak→other={p2o_ref:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

            proto = F.normalize(patches[peak_idx].mean(dim=0, keepdim=True), dim=-1)
            sim_to_proto = (patches @ proto.T).squeeze(-1).numpy()
            p2p_proto = float(sim_to_proto[peak_idx].mean())
            p2o_proto = float(sim_to_proto[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 1]
            _draw_signal_with_patch_colors(ax, signal, sim_to_proto, patch_width,
                                           cmap, norm)
            ax.set_title(
                f"{label} period {period:.0f}: cos-sim to mean peak proto "
                f"(n_peak={len(peak_idx)}, pct={peak_pct:.0f})\n"
                f"avg peak→peak={p2p_proto:.3f}, avg peak→other={p2o_proto:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

        elif mode == "flat":
            stds = _patch_stds(signal, patch_width)
            std_tol = 1e-3
            flat_all = np.where(stds < std_tol)[0]
            if len(flat_all) == 0:
                flat_all = np.array([int(np.argmin(stds))])

            ref_idx = int(flat_all[0])
            ref_val = float(amps[ref_idx])
            val_tol = max(1e-3, 0.01 * abs(ref_val))
            match_mask = (stds < std_tol) & (np.abs(amps - ref_val) < val_tol)
            match_idx = np.where(match_mask)[0]
            if len(match_idx) == 0:
                match_idx = np.array([ref_idx])
            other_idx = np.array(
                [i for i in range(n_patches) if i not in set(match_idx.tolist())]
            )

            ref = patches[ref_idx : ref_idx + 1]
            sim_to_ref = (patches @ ref.T).squeeze(-1).numpy()
            f2f_ref = float(sim_to_ref[match_idx].mean())
            f2o_ref = float(sim_to_ref[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 0]
            _draw_signal_with_patch_colors(ax, signal, sim_to_ref, patch_width,
                                           cmap, norm, highlight_idx=ref_idx)
            ax.set_title(
                f"{label} period {period:.0f} ({n_cycles} cycles): "
                f"cos-sim to reference flat patch (idx={ref_idx}, val={ref_val:.2f})\n"
                f"avg flat→flat={f2f_ref:.3f}, avg flat→other={f2o_ref:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

            proto = F.normalize(patches[match_idx].mean(dim=0, keepdim=True), dim=-1)
            sim_to_proto = (patches @ proto.T).squeeze(-1).numpy()
            f2f_proto = float(sim_to_proto[match_idx].mean())
            f2o_proto = float(sim_to_proto[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 1]
            _draw_signal_with_patch_colors(ax, signal, sim_to_proto, patch_width,
                                           cmap, norm)
            ax.set_title(
                f"{label} period {period:.0f}: cos-sim to mean flat proto "
                f"(n_flat={len(match_idx)}, val={ref_val:.2f})\n"
                f"avg flat→flat={f2f_proto:.3f}, avg flat→other={f2o_proto:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

        else:
            raise ValueError(f"Unknown sine_patch_sim mode: {mode!r}")

    plt.tight_layout()
    img = wandb.Image(plt)
    plt.close(fig)
    return img
