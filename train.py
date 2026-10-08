"""
SGF_RWKV — Training Script.

Trains the SGF_RWKV model for medical image restoration across 5 modalities:
speckle (OCT / ultrasound), noise (cryo-EM / low-dose CT / MRI).

Loss = L1_recon + L_DSM (score-map denoising score matching).

Usage:
  python train.py
  python train.py --data /path/to/data --gpus 0,1 --bs 4
  python train.py --resume
"""

import os
import sys
import argparse

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from sgf_rwkv import SGF_RWKV, count_parameters
from data.dataset import (build_train_loader, build_val_loader,
                           BalancedModalitySampler, MODALITY_LIST)
from utils.common import set_seeds, mkdir
from utils.metrics import compute_all_metrics
from utils.checkpoint import CheckpointManager
from utils.logger import ExperimentLogger


# ============================================================================
# Score-map DSM auxiliary loss
# ============================================================================
_SCORE_MAP_MULTS = [1, 2, 4]
_SCORE_MAP_WEIGHT = 1.0


def _score_map_loss(score_maps, noise_residual):
    """MSE between each score map and the downsampled noise residual.

    Target sizes are derived from each score map's spatial dimensions,
    so the loss works for any input resolution.

    Parameters
    ----------
    score_maps : list of tensors  (B, 1, H_i, W_i)
    noise_residual : Tensor  (B, 1, H, W)  LQ - HQ
    """
    mse = nn.MSELoss(reduction="mean")
    total = 0.0
    for sm, mult in zip(score_maps, _SCORE_MAP_MULTS):
        target = noise_residual
        if mult > 1:
            # downsample the noise residual to match this score map's size
            target = F.interpolate(
                target, size=sm.shape[2:],
                mode="bilinear", align_corners=False,
            )
        total = total + mse(sm, target)
    return total


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(description="SGF_RWKV Training")
parser.add_argument("--data", default="/home/lab/Data/zwj/RWKV-dataset",
                    help="Path to dataset root")
parser.add_argument("--gpus", default="0",
                    help="CUDA_VISIBLE_DEVICES")
parser.add_argument("--bs", type=int, default=4,
                    help="Batch size")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate")
parser.add_argument("--iters", type=int, default=300000,
                    help="Total training iterations")
parser.add_argument("--val_every", type=int, default=1000,
                    help="Validation interval (iterations)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint")
parser.add_argument("--no-dsm", action="store_true",
                    help="Disable DSM auxiliary loss (L1 only)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

MODALITIES = MODALITY_LIST

# ============================================================================
# Setup
# ============================================================================
set_seeds(args.seed)
save_dir = os.path.join("experiments", "sgf_rwkv")
checkpoint_dir = os.path.join(save_dir, "checkpoints")
mkdir(save_dir)
mkdir(checkpoint_dir)
logger = ExperimentLogger(save_dir)

print("=" * 60)
print(f"  SGF_RWKV Training")
print(f"  GPUs:        {args.gpus}")
print(f"  Data:        {args.data}")
print(f"  DSM loss:    {not args.no_dsm}")
print(f"  Resume:      {args.resume}")
print("=" * 60)

# --- Model ------------------------------------------------------------------
model = SGF_RWKV(
    inp_channels=1, out_channels=1, dim=48,
    num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
    clip_model_path="./pretrained/ViT-B-32.pt",
).cuda()

n_params = count_parameters(model) / 1e6
print(f"[Model] SGF_RWKV  |  params = {n_params:.2f} M  |  "
      f"DSM = {not args.no_dsm}")

# --- Loss & Optimizer -------------------------------------------------------
criterion = nn.L1Loss().cuda()
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
)
scheduler = CosineAnnealingLR(optimizer, args.iters, eta_min=1e-6)

# --- Data -------------------------------------------------------------------
# Training uses per-modality dataloaders and a balanced sampler: each
# iteration picks one modality uniformly at random and draws a full
# batch from it.  This balances gradient contribution across modalities
# regardless of their individual dataset sizes.
train_loaders, n_train = build_train_loader(
    args.data, MODALITIES,
    batch_size=args.bs, patch_size=None, num_workers=4,
)
val_loader, n_val = build_val_loader(
    args.data, MODALITIES,
    use_num=20, num_workers=2,
)
train_iter = BalancedModalitySampler(train_loaders, seed=args.seed)
print(f"[Data]  train = {n_train} samples across {len(train_loaders)} modalities"
      f" (balanced sampling)  |  val = {n_val}")

# --- Checkpoint -------------------------------------------------------------
ckpt = CheckpointManager(save_dir)
start_iter = 0
if args.resume:
    start_iter, ckpt_extra = ckpt.resume(model, optimizer, scheduler)
    if "sampler" in ckpt_extra:
        train_iter.load_state_dict(ckpt_extra["sampler"])

use_dsm = not args.no_dsm

# ============================================================================
# Training loop
# ============================================================================
pbar = tqdm(total=args.iters, initial=start_iter, desc="Train")

for it in range(start_iter + 1, args.iters + 1):

    # ---- data ----
    lq, hq, modality, fname, mod_id = next(train_iter)
    lq = lq.float().cuda()
    hq = hq.float().cuda()

    # ---- forward ----
    model.train()
    optimizer.zero_grad()

    pred, score_maps = model(lq, list(modality))

    # L1 reconstruction loss
    loss = criterion(pred, hq)

    # score-map auxiliary loss (DSM)
    if use_dsm:
        noise_residual = lq - hq
        dsm_loss = _score_map_loss(score_maps, noise_residual)
        loss = loss + _SCORE_MAP_WEIGHT * dsm_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    # ---- logging ----
    lr_now = scheduler.get_last_lr()[0]
    logger.log_scalar("Train/Loss", loss.item(), it)
    logger.log_scalar("Train/LR", lr_now, it)
    if use_dsm:
        logger.log_scalar("Train/DSM_Loss", dsm_loss.item(), it)
    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")
    pbar.update()

    # ---- validation ----
    if it % args.val_every == 0:
        model.eval()
        psum, ssum, rsum, cnt = 0.0, 0.0, 0.0, 0

        for vlq, vhq, vmod, vfn, _ in tqdm(val_loader, desc="Val", leave=False):
            vlq_c = vlq.float().cuda()
            vhq_c = vhq.float()

            with torch.no_grad():
                vout, _ = model(vlq_c, [vmod[0]])

            vout_np = vout.cpu().squeeze().numpy()
            vhq_np = vhq_c.squeeze().numpy()

            p, s, r = compute_all_metrics(vout_np, vhq_np, data_range=1.0)
            psum += p
            ssum += s
            rsum += r
            cnt += 1

            if cnt <= 3:
                logger.log_images(f"Val/{vmod[0]}/output", vout, it)

        mean_psnr = psum / cnt
        mean_ssim = ssum / cnt
        mean_rmse = rsum / cnt
        metrics = {"psnr": mean_psnr, "ssim": mean_ssim, "rmse": mean_rmse}

        logger.log_val(it, metrics)
        logger.log_scalars({
            "Val/PSNR": mean_psnr,
            "Val/SSIM": mean_ssim,
            "Val/RMSE": mean_rmse,
        }, it)

        ckpt.save(model, optimizer, scheduler, it, metrics,
                  extra_state={"sampler": train_iter.state_dict()})
        ckpt.save_best_summary()


# ============================================================================
# Done
# ============================================================================
pbar.close()
summary = ckpt.best_summary()
print(
    f"\nDone.  "
    f"Best PSNR = {summary['best_psnr']:.4f}  "
    f"Best SSIM = {summary['best_ssim']:.4f}  "
    f"Best RMSE = {summary['best_rmse']:.4f}"
)
logger.close()
