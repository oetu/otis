# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae?tab=readme-ov-file
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os

import math
import sys
import time
from contextlib import contextmanager
from typing import Iterable
import random

import torch

import numpy as np

import wandb

import util.misc as misc
import util.lr_sched as lr_sched
import util.statistics as statistics
from util.pos_embed import get_1d_sincos_pos_embed
from util.sine_patch_sim import plot_sine_patch_similarity

import matplotlib
matplotlib.use('Agg')           # prevents tkinter error
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from sklearn.feature_selection import r_regression


def _amp_dtype(args):
    """Resolve CUDA AMP autocast dtype from args (default fp16, legacy OTIS)."""
    return torch.bfloat16 if getattr(args, 'amp_dtype', 'fp16') == 'bf16' else torch.float16


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    training_history = {}

    for data_iter_step, (samples, attn_mask, pos_embed_y, domain) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            global_step = epoch * len(data_loader) + data_iter_step
            lr_sched.adjust_learning_rate_schedule(optimizer, global_step,
                                                   args.epochs * len(data_loader), args)

        # push samples to device
        samples = samples.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        # compute model prediction
        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            loss, ncc, cos_sim, cos_sim_embed, z_std, samples_hat, mask, latent = model(samples,
                                                                                        attn_mask,
                                                                                        pos_embed_y,
                                                                                        domain,
                                                                                        mask_ratio=args.mask_ratio)

        batch_size = len(samples)

        loss_value = loss.item()
        ncc_value = ncc.item()
        cos_sim_value = cos_sim.item()
        cos_sim_embed_value = cos_sim_embed.item()
        z_std_value = z_std.item()
        total_loss_value = loss_value + args.ncc_weight * (1 - ncc_value) + args.cos_weight * cos_sim

        if not math.isfinite(loss_value) and misc.is_main_process():
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        ncc /= accum_iter
        cos_sim /= accum_iter

        total_loss = loss + args.ncc_weight * (1 - ncc) + args.cos_weight * cos_sim
        grad_norm = loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        total_time = time.time() - start_time

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.meters['total_loss'].update(total_loss.item(), n=batch_size)

        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)
        metric_logger.meters['cos_sim'].update(cos_sim_value, n=batch_size)
        metric_logger.meters['cos_sim_embed'].update(cos_sim_embed_value, n=batch_size)
        metric_logger.meters['z_std'].update(z_std_value, n=batch_size)

        # Token-level cosine similarity diagnostics.
        # Layout is [CLS, patches]; skip the leading cls token.
        cls_token = torch.nn.functional.normalize(latent[:, 0, :].float(), dim=-1)           # (B, D)
        patch_tokens = torch.nn.functional.normalize(latent[:, 1:, :].float(), dim=-1)       # (B, N', D)
        cls_patch_sim_value = torch.einsum('bd,bnd->bn', cls_token, patch_tokens).mean().item()
        pp_sim = torch.einsum('bid,bjd->bij', patch_tokens, patch_tokens)
        pp_N = pp_sim.shape[1]
        pp_mask = ~torch.eye(pp_N, dtype=torch.bool, device=pp_sim.device).unsqueeze(0).expand_as(pp_sim)
        patch_patch_sim_value = pp_sim[pp_mask].mean().item()
        # RankMe — effective rank of patch outputs (Garrido et al., ICML 2023). Computed
        # on raw (un-normalised) patch tokens; higher = more directions used.
        rankme_value = statistics.rankme(latent[:, 1:, :]).item()
        metric_logger.meters['cls_patch_sim'].update(cls_patch_sim_value, n=batch_size)
        metric_logger.meters['patch_patch_sim'].update(patch_patch_sim_value, n=batch_size)
        metric_logger.meters['rankme'].update(rankme_value, n=batch_size)

        # compute MSE and MAE only of the masked patches
        # (B, 1, C, T)
        # 0 is padding, 1 is actual value
        attn_mask_input_space = torch.nn.functional.interpolate(attn_mask.unsqueeze(1), 
                                                                scale_factor=args.patch_size, 
                                                                mode="nearest")

        # (B, 1, C, T)
        # 0 is keep, 1 is remove
        mask_input_space = torch.nn.functional.interpolate(mask.reshape(attn_mask.shape).unsqueeze(1), 
                                                           scale_factor=args.patch_size, 
                                                           mode="nearest")

        # (B, 1, C, T)
        combined_mask = attn_mask_input_space * mask_input_space

        # (B, 1, C, T)
        samples_diff = samples - samples_hat

        # evaluation only on the masked patches
        mse = ((samples_diff**2) * combined_mask).sum() / (combined_mask.sum() + 1e-9)
        mae = (abs(samples_diff) * combined_mask).sum() / (combined_mask.sum() + 1e-9)
        
        mse_value = mse.item()
        mae_value = mae.item()

        metric_logger.meters['mse'].update(mse_value, n=batch_size)
        metric_logger.meters['mae'].update(mae_value, n=batch_size)

        # Pre-clip gradient norm (only produced on optimiser update steps). Logging
        # the time-series lets us read p50/p90/p99 off wandb and pick a sensible
        # --clip_grad max-norm (e.g. slightly above the bulk of stable training).
        grad_norm_value_reduce = None
        if grad_norm is not None:
            grad_norm_value = grad_norm.item()
            metric_logger.meters['grad_norm'].update(grad_norm_value, n=batch_size)
            grad_norm_value_reduce = misc.all_reduce_mean(grad_norm_value)

        total_loss_value_reduce = misc.all_reduce_mean(total_loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        ncc_value_reduce = misc.all_reduce_mean(ncc_value)
        cos_sim_value_reduce = misc.all_reduce_mean(cos_sim_value)
        cos_sim_embed_value_reduce = misc.all_reduce_mean(cos_sim_embed_value)
        z_std_value_reduce = misc.all_reduce_mean(z_std_value)
        mse_value_reduce = misc.all_reduce_mean(mse_value)
        mae_value_reduce = misc.all_reduce_mean(mae_value)
        cls_patch_sim_value_reduce = misc.all_reduce_mean(cls_patch_sim_value)
        patch_patch_sim_value_reduce = misc.all_reduce_mean(patch_patch_sim_value)
        rankme_value_reduce = misc.all_reduce_mean(rankme_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            log_writer.add_scalar('train/train_total_loss', total_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_ncc', ncc_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_cos_sim', cos_sim_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_cos_sim_embed', cos_sim_embed_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_z_std', z_std_value_reduce, epoch_1000x)
            # evaluation only on the masked patches
            log_writer.add_scalar('train/train_mse', mse_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_mae', mae_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_cls_patch_sim', cls_patch_sim_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_patch_patch_sim', patch_patch_sim_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_rankme', rankme_value_reduce, epoch_1000x)
            if grad_norm_value_reduce is not None:
                log_writer.add_scalar('train/train_grad_norm', grad_norm_value_reduce, epoch_1000x)

        if args.wandb == True and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if misc.is_main_process():
                wandb.log({"epoch_1000x": epoch_1000x,
                           "time_per_step[sec]": total_time,
                           "lr": lr,
                           "train_total_loss": total_loss_value_reduce,
                           "train_loss": loss_value_reduce,
                           "train_ncc": ncc_value_reduce,
                           "train_cos_sim": cos_sim_value_reduce,
                           "train_cos_sim_embed": cos_sim_embed_value_reduce,
                           "train_z_std": z_std_value_reduce,
                           # evaluation only on the masked patches
                           "train_mse": mse_value_reduce,
                           "train_mae": mae_value_reduce,
                           "train_cls_patch_sim": cls_patch_sim_value_reduce,
                           "train_patch_patch_sim": patch_patch_sim_value_reduce,
                           "train_rankme": rankme_value_reduce,
                           **({"train_grad_norm": grad_norm_value_reduce}
                              if grad_norm_value_reduce is not None else {})},step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # wandb
    if args.wandb == True:
        training_history['epoch'] = epoch

        if (epoch % 10) == 0:
            steps = 1
            idx = random.randint(0, len(samples)-1)

            # T_indie
            max_steps = int(attn_mask_input_space[idx, 0, 0, :].sum())

            # (1, C, T)
            x = samples[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat = samples_hat[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat_masked = (samples_hat[idx] * combined_mask[idx])[..., :max_steps:steps].detach().cpu().numpy()

            ncc_0 = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0])
            ncc_0_maskedOnly = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0], combined_mask[idx, 0, 0])

            mask_0 = (mask_input_space[idx, 0, 0, :max_steps:steps]==1).cpu().numpy()

            # samples of shape (Batch, 1, Channel, Time)
            max_channels = int(attn_mask_input_space[idx, 0, :, 0].sum())
            if max_channels > 1:
                ch_idx = random.randint(1, max_channels-1)
                ncc_1 = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx])
                ncc_1_maskedOnly = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx], combined_mask[idx, 0, ch_idx])
                mask_1 = (mask_input_space[idx, 0, ch_idx, :max_steps:steps]==1).cpu().numpy()
            else:
                ch_idx = 0
                ncc_1 = ncc_0
                ncc_1_maskedOnly = ncc_0_maskedOnly
                mask_1 = mask_0

            # Plot reconstructed time series
            plt.close('all')
            plt.figure(figsize=(8, 8))

            plt.subplot(811)
            plt.title(f"Input ({domain[idx]}, channel {0})")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')

            plt.subplot(812)
            plt.title(f"Input vs Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, 0, :].min(), x_hat[0, 0, :].min()), 
                             y2=max(x[0, 0, :].max(), x_hat[0, 0, :].max()), 
                             where=mask_0, color='gray', alpha=0.25)
            
            plt.subplot(813)
            plt.title(f"Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, 0, :].min(), 
                             y2=x_hat[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)

            plt.subplot(814)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_0_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, 0, :].min(), 
                             y2=x_hat_masked[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_0 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, 0, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

            plt.subplot(815)
            plt.title(f"Input ({domain[idx]}, channel {ch_idx})")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')

            plt.subplot(816)
            plt.title(f"Input vs Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, ch_idx, :].min(), x_hat[0, ch_idx, :].min()), 
                             y2=max(x[0, ch_idx, :].max(), x_hat[0, ch_idx, :].max()), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(817)
            plt.title(f"Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, ch_idx, :].min(), 
                             y2=x_hat[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(818)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_1_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, ch_idx, :].min(), 
                             y2=x_hat_masked[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_1 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, ch_idx, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

            plt.tight_layout()
            training_history["Reconstruction"] = wandb.Image(plt)

            # Sine-wave patch-cosine-similarity diagnostic (main process only).
            if misc.is_main_process():
                try:
                    sine_model = model.module if hasattr(model, "module") else model
                    sine_img = plot_sine_patch_similarity(
                        sine_model,
                        time_steps=2400,
                        patch_width=args.patch_size[-1],
                    )
                    if sine_img is not None:
                        training_history["Sine Patch Similarity"] = sine_img
                except Exception as e:
                    print(f"[sine_patch_sim] skipped due to {type(e).__name__}: {e}")

    return train_stats, training_history

@torch.no_grad()
def evaluate_online(estimator, model, device, train_dataloader, val_dataloader, args=None):
    # switch to evaluation mode
    model.eval()

    online_history = {}
    
    # training
    train_embeddings = []
    train_labels = []
    for data, label, label_mask, pos_embed_y in train_dataloader:
        data = data.to(device, non_blocking=True)
        label = label * label_mask
        train_labels.append(label.to(device, non_blocking=True))
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            train_embeddings.append(model.forward_encoder_all_patches(data, pos_embed_y))

    train_embeddings = torch.cat(train_embeddings, dim=0)[:, 1:, :].mean(dim=1) # globally average pooled token
    train_embeddings = train_embeddings.cpu().numpy()  # Convert to numpy for sklearn
    train_labels = torch.cat(train_labels, dim=0)
    train_labels = train_labels.cpu()

    # Convert one-hot labels to class indices for fitting if needed
    if args.online_evaluation_task == "classification":
        train_labels_for_fit = train_labels.argmax(dim=-1).numpy() if train_labels.ndim > 1 and train_labels.shape[-1] > 1 else train_labels.numpy()
    else:
        train_labels_for_fit = train_labels.numpy()

    estimator.fit(train_embeddings, train_labels_for_fit) # only fit with training data

    if args.online_evaluation_task == "classification":
        train_probs = estimator.predict_proba(train_embeddings)  # Keep as numpy, don't convert to torch
        # Convert one-hot labels to class indices if needed
        train_labels_indices = train_labels.argmax(dim=-1).numpy() if train_labels.ndim > 1 and train_labels.shape[-1] > 1 else train_labels.numpy()

        classifier_f1_train = f1_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1), average="macro")
        classifier_precision_train = precision_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1), average="macro")
        classifier_recall_train = recall_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1), average="macro")
        classifier_acc_train = accuracy_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1))
        classifier_acc_balanced_train = balanced_accuracy_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1))
        if args.online_num_classes > 2:
            classifier_auc_train = roc_auc_score(y_true=train_labels_indices, y_score=train_probs, average="macro", multi_class="ovr")
        else:
            classifier_auc_train = roc_auc_score(y_true=train_labels_indices, y_score=train_probs[:, 1], average="macro")
        # Use one-hot labels directly if already one-hot, otherwise convert
        train_labels_onehot = train_labels.numpy() if train_labels.ndim > 1 and train_labels.shape[-1] > 1 else np.eye(args.online_num_classes)[train_labels.numpy()]
        classifier_auprc_train = average_precision_score(y_true=train_labels_onehot, y_score=train_probs, average="macro")
    elif args.online_evaluation_task == "regression":
        train_preds = estimator.predict(train_embeddings)  # Keep as numpy
        classifier_rmse_train = np.float64(root_mean_squared_error(train_preds, train_labels.numpy(), multioutput="raw_values"))
        classifier_mae_train = np.float64(mean_absolute_error(train_preds, train_labels.numpy(), multioutput="raw_values"))
        classifier_pcc_train = np.concatenate([r_regression(train_preds[:, i].reshape(-1, 1), train_labels.numpy()[:, i]) for i in range(train_labels.shape[-1])], axis=0)
        classifier_r2_train = np.stack([r2_score(train_labels.numpy()[:, i], train_preds[:, i]) for i in range(train_labels.shape[-1])], axis=0)

    # validation
    val_embeddings = []
    val_labels = []
    for data, label, label_mask, pos_embed_y in val_dataloader:
        data = data.to(device, non_blocking=True)
        label = label * label_mask
        val_labels.append(label.to(device, non_blocking=True))
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            val_embeddings.append(model.forward_encoder_all_patches(data, pos_embed_y))

    val_embeddings = torch.cat(val_embeddings, dim=0)[:, 1:, :].mean(dim=1) # globally average pooled token
    val_embeddings = val_embeddings.cpu().numpy()  # Convert to numpy for sklearn
    val_labels = torch.cat(val_labels, dim=0)
    val_labels = val_labels.cpu()

    if args.online_evaluation_task == "classification":
        val_probs = estimator.predict_proba(val_embeddings)  # Keep as numpy, don't convert to torch
        # Convert one-hot labels to class indices if needed
        val_labels_indices = val_labels.argmax(dim=-1).numpy() if val_labels.ndim > 1 and val_labels.shape[-1] > 1 else val_labels.numpy()

        classifier_f1_val = f1_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1), average="macro")
        classifier_precision_val = precision_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1), average="macro")
        classifier_recall_val = recall_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1), average="macro")
        classifier_acc_val = accuracy_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1))
        classifier_acc_balanced_val = balanced_accuracy_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1))
        if args.online_num_classes > 2:
            classifier_auc_val = roc_auc_score(y_true=val_labels_indices, y_score=val_probs, average="macro", multi_class="ovr")
        else:
            classifier_auc_val = roc_auc_score(y_true=val_labels_indices, y_score=val_probs[:, 1], average="macro")
        # Use one-hot labels directly if already one-hot, otherwise convert
        val_labels_onehot = val_labels.numpy() if val_labels.ndim > 1 and val_labels.shape[-1] > 1 else np.eye(args.online_num_classes)[val_labels.numpy()]
        classifier_auprc_val = average_precision_score(y_true=val_labels_onehot, y_score=val_probs, average="macro")
    elif args.online_evaluation_task == "regression":
        val_preds = estimator.predict(val_embeddings)  # Keep as numpy
        classifier_rmse_val = np.float64(root_mean_squared_error(val_preds, val_labels.numpy(), multioutput="raw_values"))
        classifier_mae_val = np.float64(mean_absolute_error(val_preds, val_labels.numpy(), multioutput="raw_values"))
        classifier_pcc_val = np.concatenate([r_regression(val_preds[:, i].reshape(-1, 1), val_labels.numpy()[:, i]) for i in range(val_labels.shape[-1])], axis=0)
        classifier_r2_val = np.stack([r2_score(val_labels.numpy()[:, i], val_preds[:, i]) for i in range(val_labels.shape[-1])], axis=0)

    # stats
    if args.online_evaluation_task == "classification":
        online_history['online/train_f1'] = classifier_f1_train
        online_history['online/train_precision'] = classifier_precision_train
        online_history['online/train_recall'] = classifier_recall_train
        online_history['online/train_acc'] = classifier_acc_train
        online_history['online/train_acc_balanced'] = classifier_acc_balanced_train
        online_history['online/train_auc'] = classifier_auc_train
        online_history['online/train_auprc'] = classifier_auprc_train

        online_history['online/val_f1'] = classifier_f1_val
        online_history['online/val_precision'] = classifier_precision_val
        online_history['online/val_recall'] = classifier_recall_val
        online_history['online/val_acc'] = classifier_acc_val
        online_history['online/val_acc_balanced'] = classifier_acc_balanced_val
        online_history['online/val_auc'] = classifier_auc_val
        online_history['online/val_auprc'] = classifier_auprc_val
    elif args.online_evaluation_task == "regression":
        online_history['online/train_rmse'] = classifier_rmse_train.mean(axis=-1)
        online_history['online/train_mae'] = classifier_mae_train.mean(axis=-1)
        online_history['online/train_pcc'] = classifier_pcc_train.mean(axis=-1)
        online_history['online/train_r2'] = classifier_r2_train.mean(axis=-1)

        online_history['online/val_rmse'] = classifier_rmse_val.mean(axis=-1)
        online_history['online/val_mae'] = classifier_mae_val.mean(axis=-1)
        online_history['online/val_pcc'] = classifier_pcc_val.mean(axis=-1)
        online_history['online/val_r2'] = classifier_r2_val.mean(axis=-1)

    return online_history

@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    # switch to evaluation mode
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    test_history = {}  
    embeddings = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        # push samples to device
        samples = batch[0]
        samples = samples.to(device, non_blocking=True)
        
        attn_mask = batch[1]
        attn_mask = attn_mask.to(device, non_blocking=True)

        pos_embed_y = batch[2]
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        domain = batch[3]

        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            loss, ncc, cos_sim, cos_sim_embed, z_std, samples_hat, mask, latent = model(samples, 
                                                                                        attn_mask, 
                                                                                        pos_embed_y, 
                                                                                        domain, 
                                                                                        mask_ratio=args.mask_ratio)

        if args.save_embeddings:
            # latent of shape (B, 1+N', D)
            embedding = latent[:, :1, :].mean(dim=1) # (B, D)
            embeddings.append(embedding)

        batch_size = len(samples)

        loss_value = loss.item()
        ncc_value = ncc.item()
        cos_sim_value = cos_sim.item()
        cos_sim_embed_value = cos_sim_embed.item()
        z_std_value = z_std.item()
        
        metric_logger.update(loss=loss_value)

        total_loss_value = loss_value + args.ncc_weight * (1 - ncc_value) + args.cos_weight * cos_sim_value
        metric_logger.meters['total_loss'].update(total_loss_value, n=batch_size)
        
        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)
        metric_logger.meters['cos_sim'].update(cos_sim_value, n=batch_size)
        metric_logger.meters['cos_sim_embed'].update(cos_sim_embed_value, n=batch_size)
        metric_logger.meters['z_std'].update(z_std_value, n=batch_size)

        # Token-level cosine similarity diagnostics.
        # Layout is [CLS, patches]; skip the leading cls token.
        cls_token = torch.nn.functional.normalize(latent[:, 0, :].float(), dim=-1)
        patch_tokens = torch.nn.functional.normalize(latent[:, 1:, :].float(), dim=-1)
        cls_patch_sim_value = torch.einsum('bd,bnd->bn', cls_token, patch_tokens).mean().item()
        pp_sim = torch.einsum('bid,bjd->bij', patch_tokens, patch_tokens)
        pp_N = pp_sim.shape[1]
        pp_mask = ~torch.eye(pp_N, dtype=torch.bool, device=pp_sim.device).unsqueeze(0).expand_as(pp_sim)
        # RankMe — effective rank of patch outputs (Garrido et al., ICML 2023).
        rankme_value = statistics.rankme(latent[:, 1:, :]).item()
        metric_logger.meters['cls_patch_sim'].update(cls_patch_sim_value, n=batch_size)
        metric_logger.meters['patch_patch_sim'].update(pp_sim[pp_mask].mean().item(), n=batch_size)
        metric_logger.meters['rankme'].update(rankme_value, n=batch_size)

        # compute MSE and MAE only of the masked patches
        # (B, 1, C, T)
        # 0 is padding, 1 is actual value
        attn_mask_input_space = torch.nn.functional.interpolate(attn_mask.unsqueeze(1),
                                                                scale_factor=args.patch_size,
                                                                mode="nearest")

        # (B, 1, C, T)
        # 0 is keep, 1 is remove
        mask_input_space = torch.nn.functional.interpolate(mask.reshape(attn_mask.shape).unsqueeze(1), 
                                                            scale_factor=args.patch_size, 
                                                            mode="nearest")

        # (B, 1, C, T)
        combined_mask = attn_mask_input_space * mask_input_space

        # (B, 1, C, T)
        samples_diff = samples - samples_hat

        # evaluation only on the masked patches
        mse = ((samples_diff**2) * combined_mask).sum() / (combined_mask.sum() + 1e-9)
        mae = (abs(samples_diff) * combined_mask).sum() / (combined_mask.sum() + 1e-9)

        mse_value = mse.item()
        mae_value = mae.item()

        metric_logger.meters['mse'].update(mse_value, n=batch_size)
        metric_logger.meters['mae'].update(mae_value, n=batch_size)

    if args.save_embeddings and misc.is_main_process():
        embeddings = torch.cat(embeddings, dim=0).to(device="cpu", dtype=torch.float32).detach() # (B, D)
        
        embeddings_path = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        
        file_name = f"embeddings_{epoch}.pt"
        torch.save(embeddings, os.path.join(embeddings_path, file_name))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # tensorboard
    if log_writer is not None:
        log_writer.add_scalar('val/val_total_loss', test_stats["total_loss"], epoch)
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
        log_writer.add_scalar('val/val_ncc', test_stats["ncc"], epoch)
        log_writer.add_scalar('val/val_cos_sim', test_stats["cos_sim"], epoch)
        log_writer.add_scalar('val/val_cos_sim_embed', test_stats["cos_sim_embed"], epoch)
        log_writer.add_scalar('val/val_z_std', test_stats["z_std"], epoch)
        # evaluation only on the masked patches
        log_writer.add_scalar('val/val_mse', test_stats["mse"], epoch)
        log_writer.add_scalar('val/val_mae', test_stats["mae"], epoch)
        log_writer.add_scalar('val/val_cls_patch_sim', test_stats["cls_patch_sim"], epoch)
        log_writer.add_scalar('val/val_patch_patch_sim', test_stats["patch_patch_sim"], epoch)
        log_writer.add_scalar('val/val_rankme', test_stats["rankme"], epoch)

    # wandb
    if args.wandb == True:
        test_history['epoch'] = epoch
        test_history['val_total_loss'] = test_stats["total_loss"]
        test_history['val_loss'] = test_stats["loss"]
        test_history['val_ncc'] = test_stats["ncc"]
        test_history['val_cos_sim'] = test_stats["cos_sim"]
        test_history['val_cos_sim_embed'] = test_stats["cos_sim_embed"]
        test_history['val_z_std'] = test_stats["z_std"]
        # evaluation only on the masked patches
        test_history['val_mse'] = test_stats["mse"]
        test_history['val_mae'] = test_stats["mae"]
        test_history['val_cls_patch_sim'] = test_stats["cls_patch_sim"]
        test_history['val_patch_patch_sim'] = test_stats["patch_patch_sim"]
        test_history['val_rankme'] = test_stats["rankme"]

        if (epoch % 10) == 0:
            steps = 1
            idx = random.randint(0, len(samples)-1)

            # T_indie
            max_steps = int(attn_mask_input_space[idx, 0, 0, :].sum())

            # (1, C, T)
            x = samples[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat = samples_hat[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat_masked = (samples_hat[idx] * combined_mask[idx])[..., :max_steps:steps].detach().cpu().numpy()

            ncc_0 = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0])
            ncc_0_maskedOnly = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0], combined_mask[idx, 0, 0])

            mask_0 = (mask_input_space[idx, 0, 0, :max_steps:steps]==1).cpu().numpy()

            # samples of shape (Batch, 1, Channel, Time)
            max_channels = int(attn_mask_input_space[idx, 0, :, 0].sum())
            if max_channels > 1:
                ch_idx = random.randint(1, max_channels-1)
                ncc_1 = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx])
                ncc_1_maskedOnly = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx], combined_mask[idx, 0, ch_idx])
                mask_1 = (mask_input_space[idx, 0, ch_idx, :max_steps:steps]==1).cpu().numpy()
            else:
                ch_idx = 0
                ncc_1 = ncc_0
                ncc_1_maskedOnly = ncc_0_maskedOnly
                mask_1 = mask_0

            # Plot reconstructed time series
            plt.close('all')
            plt.figure(figsize=(8, 8))

            plt.subplot(811)
            plt.title(f"Input ({domain[idx]}, channel {0})")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')

            plt.subplot(812)
            plt.title(f"Input vs Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, 0, :].min(), x_hat[0, 0, :].min()), 
                             y2=max(x[0, 0, :].max(), x_hat[0, 0, :].max()), 
                             where=mask_0, color='gray', alpha=0.25)
            
            plt.subplot(813)
            plt.title(f"Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, 0, :].min(), 
                             y2=x_hat[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)

            plt.subplot(814)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_0_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, 0, :].min(), 
                             y2=x_hat_masked[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_0 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, 0, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

            plt.subplot(815)
            plt.title(f"Input ({domain[idx]}, channel {ch_idx})")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')

            plt.subplot(816)
            plt.title(f"Input vs Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, ch_idx, :].min(), x_hat[0, ch_idx, :].min()), 
                             y2=max(x[0, ch_idx, :].max(), x_hat[0, ch_idx, :].max()), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(817)
            plt.title(f"Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, ch_idx, :].min(), 
                             y2=x_hat[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(818)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_1_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, ch_idx, :].min(), 
                             y2=x_hat_masked[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_1 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, ch_idx, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

            plt.tight_layout()
            test_history["Val Reconstruction"] = wandb.Image(plt)

    return test_stats, test_history


# ---------------------------------------------------------------------------
# Online evaluations
#
# OTIS's token layout is [CLS, patches] and its forward signatures take an
# extra ``pos_embed_y`` (LongTensor of variate indices, padding_idx=0). The UEA
# and forecast helpers below build a simple per-batch ``pos_embed_y`` using
# indices 1..V' (domain_offset=0) — good enough for an online diagnostic.
# ---------------------------------------------------------------------------

# 23 UEA multivariate time series classification datasets: (name, variates, timesteps, classes)
UEA_DATASETS = [
    ("ArticularyWordRecognition", 9, 144, 25),
    ("AtrialFibrillation", 2, 640, 3),
    ("BasicMotions", 6, 100, 4),
    ("CharacterTrajectories", 3, 180, 20),
    ("Cricket", 6, 1197, 12),
    ("ERing", 4, 65, 6),
    ("Epilepsy", 3, 206, 4),
    ("EthanolConcentration", 3, 1751, 4),
    ("FaceDetection", 144, 62, 2),
    ("FingerMovements", 28, 50, 2),
    ("HandMovementDirection", 10, 400, 4),
    ("Handwriting", 3, 152, 26),
    ("Heartbeat", 61, 405, 2),
    ("JapaneseVowels", 12, 26, 9),
    ("LSST", 6, 36, 14),
    ("Libras", 2, 45, 15),
    ("NATOPS", 24, 51, 6),
    ("Phoneme", 1, 1024, 39),
    ("RacketSports", 6, 30, 4),
    ("SelfRegulationSCP1", 6, 896, 2),
    ("SelfRegulationSCP2", 7, 1152, 2),
    ("SpokenArabicDigits", 13, 93, 10),
    ("UWaveGestureLibrary", 3, 315, 8),
]


def _load_uea_dataset(data_base, name):
    train_data = torch.load(os.path.join(data_base, name, "train.pt"), map_location="cpu", weights_only=False)
    train_labels = torch.load(os.path.join(data_base, name, "train_labels.pt"), map_location="cpu", weights_only=False)
    test_data = torch.load(os.path.join(data_base, name, "test.pt"), map_location="cpu", weights_only=False)
    test_labels = torch.load(os.path.join(data_base, name, "test_labels.pt"), map_location="cpu", weights_only=False)

    X_train = torch.stack([s.unsqueeze(0) for _, s in train_data], dim=0).float()
    X_test = torch.stack([s.unsqueeze(0) for _, s in test_data], dim=0).float()
    y_train = train_labels.argmax(dim=-1).long() if train_labels.ndim == 2 else train_labels.long()
    y_test = test_labels.argmax(dim=-1).long() if test_labels.ndim == 2 else test_labels.long()
    return X_train, y_train, X_test, y_test


def _build_pos_embed_y(B: int, V: int, Tp: int, device, dtype=torch.long, max_idx: int = None):
    """Row-major variate indices ``1..V`` expanded to (B, V, T'), no domain offset.

    ``max_idx`` (e.g. ``model.pos_embed_y.num_embeddings - 1``) clamps the
    indices so out-of-distribution samples with more variates than the
    training set saw don't trigger a CUDA assert in ``nn.Embedding``.
    """
    idx = torch.arange(V, device=device, dtype=dtype) + 1
    if max_idx is not None:
        idx = idx.clamp(max=max_idx)
    return idx.view(1, V, 1).expand(B, V, Tp).contiguous()


@contextmanager
def _random_pos_embed_y(model, num_variates):
    """Temporarily swap ``model.pos_embed_y`` for a freshly randomly-initialised
    ``nn.Embedding(num_variates + 1, D/2)`` so online evaluation does not leak
    pre-training domain signal through the learned variate embedding. Row 0
    stays zeroed to preserve the padding-index convention; indices 1..V are
    drawn iid from N(0, 0.02**2).
    """
    if not hasattr(model, "pos_embed_y"):
        yield
        return

    orig = model.pos_embed_y
    new = torch.nn.Embedding(num_variates + 1, orig.embedding_dim, padding_idx=0)
    torch.nn.init.normal_(new.weight, std=0.02)
    with torch.no_grad():
        new.weight[0].zero_()
    new = new.to(orig.weight.device).to(orig.weight.dtype)
    model.pos_embed_y = new
    try:
        yield
    finally:
        model.pos_embed_y = orig


def _extract_embeddings(model, X, device, batch_size=64):
    """CLS and GAP (patch-mean) latents. Layout: [CLS, patches]."""
    patch_size = model.patch_size if hasattr(model, "patch_size") else (1, 24)
    pw = patch_size[1]
    max_patches_x = int(getattr(model, "max_num_patches_x", 10 ** 9))
    cls_embs, gap_embs = [], []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size].to(device, non_blocking=True)
        B, _, V, T = batch.shape
        # trim to a multiple of the time patch size AND clamp to what pos_embed_x supports
        Tp_cap = max_patches_x
        T = min((T // pw), Tp_cap) * pw
        if T == 0:
            continue
        batch = batch[..., :T]
        Tp = T // pw
        max_idx = int(model.pos_embed_y.num_embeddings) - 1
        pos_embed_y = _build_pos_embed_y(B, V, Tp, device, max_idx=max_idx)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # (B, 1+N, D) = [CLS, patches]
            emb = model.forward_encoder_all_patches(batch, pos_embed_y)
        cls_embs.append(emb[:, 0, :].float().cpu())
        gap_embs.append(emb[:, 1:, :].float().mean(dim=1).cpu())
    return torch.cat(cls_embs, dim=0), torch.cat(gap_embs, dim=0)


def _prototypical_acc(train_cls, train_gap, train_y, test_cls, test_gap, test_y,
                      nb_classes, k_shots, n_episodes, rng):
    cls_accs, gap_accs = [], []
    for _ in range(n_episodes):
        support_idx = []
        for c in range(nb_classes):
            class_idx = (train_y == c).nonzero(as_tuple=True)[0].tolist()
            if len(class_idx) <= k_shots:
                support_idx.extend(class_idx)
            else:
                support_idx.extend(rng.choice(class_idx, size=k_shots, replace=False).tolist())

        support_y = train_y[support_idx]

        for train_emb, test_emb, accs in (
            (train_cls, test_cls, cls_accs),
            (train_gap, test_gap, gap_accs),
        ):
            support_emb = train_emb[support_idx]
            prototypes = torch.zeros(nb_classes, support_emb.shape[1])
            for c in range(nb_classes):
                mask = support_y == c
                if mask.any():
                    prototypes[c] = support_emb[mask].mean(dim=0)

            prototypes_norm = torch.nn.functional.normalize(prototypes, dim=1)
            query_norm = torch.nn.functional.normalize(test_emb, dim=1)
            preds = (query_norm @ prototypes_norm.T).argmax(dim=1)

            accs.append(balanced_accuracy_score(test_y.numpy(), preds.numpy()))

    return float(np.mean(cls_accs)), float(np.mean(gap_accs))


@torch.no_grad()
def evaluate_online_kshot(model, device, data_base, k_shots=5, n_episodes=3, seed=0):
    """5-shot prototypical evaluation on the 23 UEA datasets."""
    model.eval()
    rng = np.random.RandomState(seed)
    online_history = {}
    cls_accs, gap_accs = [], []

    for name, V, _T, C in UEA_DATASETS:
        try:
            X_train, y_train, X_test, y_test = _load_uea_dataset(data_base, name)
        except FileNotFoundError:
            print(f"  [online kshot] SKIP {name} (data not found)")
            continue

        # Random (not pretrained) variate embeddings for non-axial-RoPE models,
        # sized to this dataset's V so no index clamping is needed.
        with _random_pos_embed_y(model, V):
            train_cls, train_gap = _extract_embeddings(model, X_train, device)
            test_cls, test_gap = _extract_embeddings(model, X_test, device)

        acc_cls, acc_gap = _prototypical_acc(
            train_cls, train_gap, y_train,
            test_cls, test_gap, y_test,
            C, k_shots, n_episodes, rng,
        )
        cls_accs.append(acc_cls)
        gap_accs.append(acc_gap)
        online_history[f'online/{name}_acc_balanced'] = acc_gap * 100

    if gap_accs:
        avg_gap = float(np.mean(gap_accs)) * 100
        online_history['online/kshot_avg_acc_balanced'] = avg_gap
        print(f"  [online kshot] avg balanced acc (gap): {avg_gap:.1f}% ({len(gap_accs)} datasets)")

    if cls_accs:
        avg_cls = float(np.mean(cls_accs)) * 100
        online_history['online/kshot_cls_avg_acc_balanced'] = avg_cls
        print(f"  [online kshot] avg balanced acc (cls): {avg_cls:.1f}% ({len(cls_accs)} datasets)")

    return online_history


@torch.no_grad()
def evaluate_online_forecast(model, device, data_path, args, mask_ratio=0.25,
                             time_steps=2400, batch_size=32):
    """Right-sided forecasting eval on a small dataset. Returns {} if file missing."""
    if not os.path.exists(data_path):
        print(f"  [online forecast] SKIP (data not found: {data_path})")
        return {}

    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    # samples are (domain_str, tensor(1, V, T_full)); remember the known domains
    known_domains = list(getattr(model, 'grid_height', {}).keys()) or []
    domain_strings = [d if d in known_domains else (known_domains[0] if known_domains else d)
                      for d, _ in raw]
    tensors_all = [t for _, t in raw]

    pw = args.patch_width
    Tp = time_steps // pw
    T = Tp * pw
    X = torch.stack([t[..., :T] for t in tensors_all], dim=0).float()

    # The baked-in pos_embed_x tables are sized to
    # max_num_patches_x = time_steps_train // patch_width. Evaluating on a
    # longer horizon requires rebuilding the sin-cos tables at Tp.
    base = model.module if hasattr(model, "module") else model
    saved_pos = None
    if Tp > int(getattr(base, "max_num_patches_x", Tp)):
        enc_pe = base.pos_embed_x
        enc_new = torch.from_numpy(
            get_1d_sincos_pos_embed(enc_pe.shape[-1], Tp, cls_token=True)
        ).float().unsqueeze(0).to(enc_pe.device, enc_pe.dtype)
        dec_pe = getattr(base, "decoder_pos_embed_x", None)
        if dec_pe is not None:
            dec_new = torch.from_numpy(
                get_1d_sincos_pos_embed(dec_pe.shape[-1], Tp, cls_token=True)
            ).float().unsqueeze(0).to(dec_pe.device, dec_pe.dtype)
        else:
            dec_new = None
        saved_pos = (enc_pe, dec_pe)
        base.pos_embed_x = torch.nn.Parameter(enc_new, requires_grad=False)
        if dec_pe is not None:
            base.decoder_pos_embed_x = torch.nn.Parameter(dec_new, requires_grad=False)

    saved = (model.probabilistic_masking, model.include_forecasting,
             model.forecasting_probability, model.forecasting_mask_ratio)
    # Force right-sided masking with deterministic horizon = mask_ratio
    model.probabilistic_masking = False
    model.include_forecasting = True
    model.forecasting_probability = 1.0
    model.forecasting_mask_ratio = mask_ratio
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_ncc_masked = 0.0
    total_ncc_full = 0.0
    n_batches = 0
    last_batch = None

    # V is constant across the forecast dataset (tensors are stacked).
    V_fc = int(X.shape[-2])
    try:
        with _random_pos_embed_y(base, V_fc):
            for i in range(0, len(X), batch_size):
                samples = X[i:i + batch_size].to(device, non_blocking=True)  # (B, 1, V, T)
                batch_domain = domain_strings[i:i + batch_size]
                B, _, V, T_ = samples.shape
                Tp = T_ // pw
                attn_mask = torch.ones(B, V, Tp, device=device)
                max_idx = int(model.pos_embed_y.num_embeddings) - 1
                pos_embed_y = _build_pos_embed_y(B, V, Tp, device, max_idx=max_idx)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, ncc, _, _, _, samples_hat, mask, _ = model(
                        samples, attn_mask, pos_embed_y, batch_domain, mask_ratio=mask_ratio
                    )

                # MSE/MAE on masked region
                attn_mask_is = torch.nn.functional.interpolate(
                    attn_mask.unsqueeze(1), scale_factor=args.patch_size, mode="nearest")
                mask_is = torch.nn.functional.interpolate(
                    mask.reshape(attn_mask.shape).unsqueeze(1), scale_factor=args.patch_size, mode="nearest")
                combined = attn_mask_is * mask_is
                diff = samples - samples_hat
                mse_val = ((diff ** 2) * combined).sum() / (combined.sum() + 1e-9)
                mae_val = (diff.abs() * combined).sum() / (combined.sum() + 1e-9)

                ncc_masked = statistics.ncc(samples, samples_hat, combined, keep_batch=True).mean().item()

                total_mse += mse_val.item()
                total_mae += mae_val.item()
                total_ncc_masked += ncc_masked
                total_ncc_full += ncc.item()
                n_batches += 1

                last_batch = (samples, samples_hat, combined, mask_is, attn_mask_is)
    finally:
        (model.probabilistic_masking, model.include_forecasting,
         model.forecasting_probability, model.forecasting_mask_ratio) = saved
        if saved_pos is not None:
            enc_pe_orig, dec_pe_orig = saved_pos
            base.pos_embed_x = enc_pe_orig
            if dec_pe_orig is not None:
                base.decoder_pos_embed_x = dec_pe_orig

    if n_batches == 0:
        return {}

    avg_mse = total_mse / n_batches
    avg_mae = total_mae / n_batches
    avg_ncc_masked = total_ncc_masked / n_batches
    avg_ncc_full = total_ncc_full / n_batches

    print(f"  [online forecast] mse={avg_mse:.4f}  mae={avg_mae:.4f}  "
          f"ncc(horizon)={avg_ncc_masked:.3f}  ncc(full)={avg_ncc_full:.3f}  ({len(X)} samples)")

    online_history = {
        'online/forecast_mse': avg_mse,
        'online/forecast_mae': avg_mae,
        'online/forecast_ncc': avg_ncc_masked,
        'online/forecast_ncc_full': avg_ncc_full,
    }

    if misc.is_main_process() and last_batch is not None:
        try:
            online_history['online/forecast_reconstruction'] = _plot_forecast_reconstruction(*last_batch, idx=0)
        except Exception as e:
            print(f"  [online forecast] failed to plot reconstruction: {e}")

    return online_history


def _plot_forecast_reconstruction(samples, samples_hat, combined_mask,
                                  mask_input_space, attn_mask_input_space, idx=0):
    """8-subplot forecast reconstruction figure, following OTIS's Reconstruction layout."""
    max_steps = int(attn_mask_input_space[idx, 0, 0, :].sum())

    x = samples[idx][..., :max_steps].detach().cpu().numpy()
    x_hat = samples_hat[idx][..., :max_steps].detach().cpu().numpy()
    x_hat_masked = (samples_hat[idx] * combined_mask[idx])[..., :max_steps].detach().cpu().numpy()

    ncc_0 = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0])
    ncc_0_m = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0], combined_mask[idx, 0, 0])
    mask_0 = (mask_input_space[idx, 0, 0, :max_steps] == 1).cpu().numpy()

    max_channels = int(attn_mask_input_space[idx, 0, :, 0].sum())
    if max_channels > 1:
        ch_idx = random.randint(1, max_channels - 1)
        ncc_1 = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx])
        ncc_1_m = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx], combined_mask[idx, 0, ch_idx])
        mask_1 = (mask_input_space[idx, 0, ch_idx, :max_steps] == 1).cpu().numpy()
    else:
        ch_idx = 0
        ncc_1 = ncc_0
        ncc_1_m = ncc_0_m
        mask_1 = mask_0

    t = range(0, x.shape[-1])
    plt.close('all')
    plt.figure(figsize=(8, 8))

    for panel_idx, (c, ncc_v, ncc_mv, m) in enumerate(
        [(0, ncc_0, ncc_0_m, mask_0), (ch_idx, ncc_1, ncc_1_m, mask_1)]
    ):
        off = 1 + 4 * panel_idx
        plt.subplot(8, 1, off)
        plt.title(f"Input (channel {c})")
        plt.plot(t, x[0, c, :], color='black')

        plt.subplot(8, 1, off + 1)
        plt.title(f"Input vs Reconstruction (NCC {ncc_v.item():.2f}, masked in gray)")
        plt.plot(t, x[0, c, :], color='black')
        plt.plot(t, x_hat[0, c, :], color='darkorange')
        plt.fill_between(t,
                         y1=min(x[0, c, :].min(), x_hat[0, c, :].min()),
                         y2=max(x[0, c, :].max(), x_hat[0, c, :].max()),
                         where=m, color='gray', alpha=0.25)

        plt.subplot(8, 1, off + 2)
        plt.title(f"Reconstruction (NCC {ncc_v.item():.2f}, masked in gray)")
        plt.plot(t, x_hat[0, c, :], color='darkorange')
        plt.fill_between(t, y1=x_hat[0, c, :].min(), y2=x_hat[0, c, :].max(),
                         where=m, color='gray', alpha=0.25)

        plt.subplot(8, 1, off + 3)
        plt.title(f"Reconstruction of masked region (NCC {ncc_mv.item():.2f})")
        plt.plot(t, x_hat_masked[0, c, :], color='darkorange')
        plt.fill_between(t, y1=x_hat_masked[0, c, :].min(), y2=x_hat_masked[0, c, :].max(),
                         where=m, color='gray', alpha=0.25)
        vis = np.where(m == False)
        plt.scatter(vis, x_hat_masked[0, c, :][vis], color='white', s=7, zorder=2)

    plt.tight_layout()
    return wandb.Image(plt)