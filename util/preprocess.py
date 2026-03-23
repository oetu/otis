# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Preprocess raw time series data (normalise and clamp) for OTiS.
#
# Usage:
#   python preprocess.py --input /path/to/raw_data.pt --output /path/to/processed_data.pt
#   python preprocess.py --input /path/to/raw_data.npy --output /path/to/processed_data.pt
#   python preprocess.py --input /path/to/raw_data.pt --output /path/to/processed_data.pt \
#       --clamp_min -4 --clamp_max 4 --outlier_threshold 2.0
# --------------------------------------------------------
import argparse

import numpy as np
import torch


def _resolve_axes(mode: str, ndim: int, global_stats: bool = False) -> tuple:
    if mode == "sample":
        if global_stats:
            # Stats over all samples, variates, and time steps
            return tuple(range(ndim))
        # Stats over the entire sample (all variates and time steps)
        # For a single sample (V, T) : axes=(-2, -1); for batch (N, V, T) : same
        return tuple(range(ndim - 2, ndim)) if ndim >= 2 else (-1,)
    elif mode == "variate":
        if global_stats:
            # Stats per variate across all samples: (N, V, T) -> axes=(0, -1)
            return tuple(i for i in range(ndim) if i != ndim - 2)
        # Stats per variate (along the time axis only)
        return (-1,)
    else:
        raise ValueError(f"Unknown normalisation mode: '{mode}'. Use 'sample' or 'variate'.")


def compute_stats(data: np.ndarray, mode: str = "sample",
                  ignore_outliers: bool = True, outlier_threshold: float = 3.0,
                  global_stats: bool = False) -> dict:
    """
    Compute normalisation statistics (mean, std) from data.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (..., V, T).
    mode : str
        "sample" or "variate".
    ignore_outliers : bool
        If True, compute robust statistics ignoring outlier points.
    outlier_threshold : float
        Outlier detection threshold in standard deviations.
    global_stats : bool
        If True, compute statistics across all samples (collapsing the batch
        dimension). Use this when computing training-set statistics that will
        be applied to other splits.

    Returns
    -------
    dict with keys "mean" and "std", each an np.ndarray.
    """
    axes = _resolve_axes(mode, data.ndim, global_stats=global_stats)

    if not ignore_outliers:
        mean = data.mean(axis=axes, keepdims=True)
        std = data.std(axis=axes, keepdims=True)
        return {"mean": mean, "std": std}

    # Robust statistics: first pass to identify outliers
    initial_mean = data.mean(axis=axes, keepdims=True)
    initial_std = data.std(axis=axes, keepdims=True)

    z_scores = (data - initial_mean) / (initial_std + 1e-9)
    mask = np.abs(z_scores) <= outlier_threshold

    # Re-compute statistics using only non-outlier points
    mean = np.sum(data * mask, axis=axes, keepdims=True) / (np.sum(mask, axis=axes, keepdims=True) + 1e-9)
    squared_diff = (data - mean) ** 2
    var = np.sum(squared_diff * mask, axis=axes, keepdims=True) / (np.sum(mask, axis=axes, keepdims=True) + 1e-9)
    std = np.sqrt(var)

    return {"mean": mean, "std": std}


def normalise(data: np.ndarray, mode: str = "sample",
              ignore_outliers: bool = True, outlier_threshold: float = 3.0,
              stats: dict = None) -> np.ndarray:
    """
    Zero-normalisation (mean=0, std=1).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (..., V, T) where V is the variate and T is the time dimension.
    mode : str
        "sample" to compute statistics over the entire sample (all variates and time steps).
        "variate" to compute statistics per variate (along the time axis only).
    ignore_outliers : bool
        If True, compute mean/std from non-outlier points only (robust normalisation).
    outlier_threshold : float
        Points beyond this many standard deviations are considered outliers
        when computing the robust statistics.
    stats : dict, optional
        Pre-computed {"mean": ..., "std": ...} (e.g. from the training set).
        When provided, these statistics are used directly and ignore_outliers
        / outlier_threshold are ignored.

    Returns
    -------
    np.ndarray : normalised data, same shape as input.
    """
    if stats is not None:
        mean, std = stats["mean"], stats["std"]
        return (data - mean) / (std + 1e-9)

    s = compute_stats(data, mode=mode, ignore_outliers=ignore_outliers, outlier_threshold=outlier_threshold)
    return (data - s["mean"]) / (s["std"] + 1e-9)


def clamp(data: np.ndarray, min_val: float = -3.0, max_val: float = 3.0) -> np.ndarray:
    """
    Clamp values to [min_val, max_val].

    Parameters
    ----------
    data : np.ndarray
        Normalised data.
    min_val, max_val : float
        Clamp bounds (typically +/- 3 standard deviations).

    Returns
    -------
    np.ndarray : clamped data, same shape as input.
    """
    return np.clip(data, min_val, max_val)


def preprocess(data: np.ndarray,
               norm_mode: str = "sample",
               ignore_outliers: bool = True,
               outlier_threshold: float = 3.0,
               clamp_min: float = -3.0,
               clamp_max: float = 3.0,
               stats: dict = None) -> np.ndarray:
    """
    Full preprocessing pipeline: normalise then clamp.

    Parameters
    ----------
    data : np.ndarray
        Raw time series of shape (..., V, T).
    norm_mode : str
        "sample" (default) or "variate".
    ignore_outliers : bool
        Use robust normalisation (ignore outliers when computing stats).
    outlier_threshold : float
        Outlier detection threshold in standard deviations.
    clamp_min, clamp_max : float
        Clamp bounds after normalisation.
    stats : dict, optional
        Pre-computed {"mean": ..., "std": ...} from the training set.

    Returns
    -------
    np.ndarray : preprocessed data, same shape as input.
    """
    data = normalise(data, mode=norm_mode, ignore_outliers=ignore_outliers,
                     outlier_threshold=outlier_threshold, stats=stats)
    data = clamp(data, min_val=clamp_min, max_val=clamp_max)
    return data


def load_data(path: str):
    """
    Load raw data from .pt or .npy file.

    Returns
    -------
    data : list[tuple[str, np.ndarray]]  or  np.ndarray
        If .pt and list-of-tuples format: [(domain, array), ...]
        Otherwise: a single np.ndarray.
    """
    if path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data
    elif path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".npz"):
        return np.load(path)["data"]
    else:
        raise ValueError(f"Unsupported file format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw time series data for OTiS.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw data file (.pt, .npy, or .npz).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save preprocessed data (.pt).")
    parser.add_argument("--norm_mode", type=str, default="sample", choices=["sample", "variate"],
                        help="Normalisation mode: 'sample' (default) computes stats over the entire sample, "
                             "'variate' computes stats per variate along the time axis.")
    parser.add_argument("--ignore_outliers", action="store_true", default=True,
                        help="Use robust normalisation that ignores outliers (default: True).")
    parser.add_argument("--no_ignore_outliers", action="store_true",
                        help="Disable robust normalisation.")
    parser.add_argument("--outlier_threshold", type=float, default=3.0,
                        help="Outlier threshold in std deviations for robust normalisation (default: 3.0).")
    parser.add_argument("--clamp_min", type=float, default=-3.0,
                        help="Lower clamp bound (default: -3.0).")
    parser.add_argument("--clamp_max", type=float, default=3.0,
                        help="Upper clamp bound (default: 3.0).")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data file. When provided, normalisation statistics "
                             "are computed from this file and applied to --input.")
    parser.add_argument("--save_stats", type=str, default=None,
                        help="Save computed normalisation statistics to this .npz file.")
    parser.add_argument("--load_stats", type=str, default=None,
                        help="Load pre-computed normalisation statistics from this .npz file "
                             "(overrides --train_data).")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain name for OTiS output format. When provided, output is saved "
                             "as [(domain, tensor), ...] list-of-tuples instead of a single tensor.")
    args = parser.parse_args()

    ignore_outliers = not args.no_ignore_outliers

    # Resolve normalisation statistics
    stats = None
    if args.load_stats:
        print(f"Loading statistics from {args.load_stats}")
        npz = np.load(args.load_stats)
        stats = {"mean": npz["mean"], "std": npz["std"]}
        print(f"  mean shape: {stats['mean'].shape}, std shape: {stats['std'].shape}")
    elif args.train_data:
        print(f"Computing statistics from training data: {args.train_data}")
        train_raw = load_data(args.train_data)
        if isinstance(train_raw, list) and len(train_raw) > 0 and isinstance(train_raw[0], tuple):
            train_arrays = []
            for _, sample in train_raw:
                s = sample.numpy() if isinstance(sample, torch.Tensor) else np.asarray(sample)
                train_arrays.append(s)
            train_np = np.stack(train_arrays)
        else:
            train_np = train_raw.numpy() if isinstance(train_raw, torch.Tensor) else np.asarray(train_raw, dtype=np.float32)
        stats = compute_stats(train_np, mode=args.norm_mode,
                              ignore_outliers=ignore_outliers,
                              outlier_threshold=args.outlier_threshold,
                              global_stats=True)
        print(f"  mean shape: {stats['mean'].shape}, std shape: {stats['std'].shape}")
        del train_raw, train_np

    if args.save_stats and stats is not None:
        print(f"Saving statistics to {args.save_stats}")
        np.savez(args.save_stats, mean=stats["mean"], std=stats["std"])

    print(f"Loading data from {args.input}")
    raw_data = load_data(args.input)

    # Handle OTiS list-of-tuples format: [(domain_str, tensor), ...]
    if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], tuple):
        print(f"Detected OTiS format: {len(raw_data)} samples")
        processed = []
        for domain, sample in raw_data:
            if isinstance(sample, torch.Tensor):
                sample_np = sample.numpy()
            else:
                sample_np = np.asarray(sample)

            sample_np = preprocess(sample_np,
                                   norm_mode=args.norm_mode,
                                   ignore_outliers=ignore_outliers,
                                   outlier_threshold=args.outlier_threshold,
                                   clamp_min=args.clamp_min,
                                   clamp_max=args.clamp_max,
                                   stats=stats)
            processed.append((domain, torch.tensor(sample_np, dtype=torch.float32)))

        print(f"Saving {len(processed)} samples to {args.output}")
        torch.save(processed, args.output)

    # Handle plain tensor / array: shape (N, V, T) or (V, T)
    else:
        if isinstance(raw_data, torch.Tensor):
            raw_data = raw_data.numpy()
        raw_data = np.asarray(raw_data, dtype=np.float32)

        print(f"Data shape: {raw_data.shape}")
        print(f"Before - min: {raw_data.min():.4f}, max: {raw_data.max():.4f}, "
              f"mean: {raw_data.mean():.4f}, std: {raw_data.std():.4f}")

        processed = preprocess(raw_data,
                               norm_mode=args.norm_mode,
                               ignore_outliers=ignore_outliers,
                               outlier_threshold=args.outlier_threshold,
                               clamp_min=args.clamp_min,
                               clamp_max=args.clamp_max,
                               stats=stats)

        print(f"After - min: {processed.min():.4f}, max: {processed.max():.4f}, "
              f"mean: {processed.mean():.4f}, std: {processed.std():.4f}")

        if args.domain:
            processed = [(args.domain, torch.tensor(processed[i], dtype=torch.float32))
                         for i in range(processed.shape[0])]
            print(f"Saving {len(processed)} samples in OTiS format to {args.output}")
        else:
            processed = torch.tensor(processed, dtype=torch.float32)
            print(f"Saving to {args.output}")
        torch.save(processed, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
