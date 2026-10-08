"""
SGF_RWKV — Testing Script.

Evaluates the trained SGF_RWKV model on each modality independently,
saves per-sample metrics and horizontal [input | output | label] strips.

Usage:
  python test.py
  python test.py --data /path/to/data
  python test.py --ckpt experiments/sgf_rwkv/checkpoints/best_psnr.pth
  python test.py --modality "noise in low dose CT" --split val
  python test.py --save_score_map
"""

import os
import sys
import argparse

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from sgf_rwkv import SGF_RWKV
from data.dataset import build_test_loader, build_val_loader, MODALITY_LIST
from utils.common import set_seeds, mkdir
from utils.metrics import compute_all_metrics
from utils.checkpoint import CheckpointManager
from utils.logger import ExperimentLogger


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(description="SGF_RWKV Testing")
parser.add_argument("--data", default="/home/lab/Data_4T/zwj/RWKV-dataset",
                    help="Path to dataset root")
parser.add_argument("--ckpt", default=None,
                    help="Checkpoint path; default = experiments/sgf_rwkv/checkpoints/best_psnr.pth")
parser.add_argument("--gpu", default="0",
                    help="CUDA device")
parser.add_argument("--split", default="test", choices=["test", "val"],
                    help="Data split to evaluate on")
parser.add_argument("--modality", default=None,
                    help="Test only a single modality")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--save_score_map", action="store_true",
                    help="Save score-map visualisations")
parser.add_argument("--score_cmap", default="inferno",
                    choices=["inferno", "viridis", "plasma", "magma", "jet", "hot"],
                    help="Colormap for score-map heatmaps")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

MODALITIES = list(MODALITY_LIST)
if args.modality:
    MODALITIES = [args.modality]

set_seeds(args.seed)
save_dir = os.path.join("experiments", "sgf_rwkv")
ckpt_path = args.ckpt or os.path.join(
    save_dir, "checkpoints", "best_psnr.pth")

print("=" * 60)
print(f"  SGF_RWKV Test")
print(f"  Split:       {args.split}")
print(f"  Checkpoint:  {ckpt_path}")
print(f"  Modalities:  {MODALITIES}")
print("=" * 60)


def _normalize_panel(tensor_1chw):
    """Min-max normalise a (1, C, H, W) tensor to numpy (H, W) in [0, 1]."""
    t = tensor_1chw.squeeze().cpu()
    t_min, t_max = t.min(), t.max()
    return ((t - t_min) / (t_max - t_min + 1e-8)).numpy()


# ============================================================================
# Build model
# ============================================================================
model = SGF_RWKV(
    inp_channels=1, out_channels=1, dim=48,
    num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
    clip_model_path="./pretrained/ViT-B-32.pt",
).cuda()
model.eval()

ckpt_mgr = CheckpointManager(save_dir)
if os.path.exists(ckpt_path):
    ckpt_mgr.load_model_only(model, ckpt_path)
else:
    print(f"[WARN] Checkpoint not found: {ckpt_path} — using random weights!")

# ── Logger ─────────────────────────────────────────────────────────────
logger = ExperimentLogger(save_dir)
logger.write_summary_header()

# ── Data loader ────────────────────────────────────────────────────────
build_loader = build_test_loader if args.split == "test" else build_val_loader


# ============================================================================
# Per-modality evaluation
# ============================================================================
for mod_name in MODALITIES:
    loader, n_samples = build_loader(
        args.data, [mod_name], use_num=None, num_workers=2)
    if n_samples == 0:
        print(f"[Skip] {mod_name}: no samples found")
        continue

    logger.start_modality(mod_name)
    out_dir = os.path.join(save_dir, "test_results", mod_name)
    mkdir(out_dir)

    for vlq, vhq, vmod, vfn, _ in tqdm(loader, desc=mod_name):
        vlq_c = vlq.float().cuda()
        fname = vfn[0]

        with torch.no_grad():
            vout, smaps = model(vlq_c, [vmod[0]])

        lq_np  = vlq.squeeze().numpy()
        out_np = vout.cpu().squeeze().numpy()
        gt_np  = vhq.squeeze().numpy()

        # ---- score map visualisation ----
        if args.save_score_map:
            _, _, H_target, W_target = vlq_c.shape

            lq_norm = _normalize_panel(vlq_c)
            lq_rgb = (np.stack([lq_norm] * 3, axis=-1) * 255).astype(np.uint8)

            cmap = plt.get_cmap(args.score_cmap)
            sm_panels = [lq_rgb]
            for sm in smaps:
                sm_up = F.interpolate(
                    sm, size=(H_target, W_target),
                    mode='bilinear', align_corners=False)
                sm_norm = _normalize_panel(sm_up)
                sm_rgb = (cmap(sm_norm)[:, :, :3] * 255).astype(np.uint8)
                sm_panels.append(sm_rgb)

            score_strip = np.concatenate(sm_panels, axis=1)

            sm_dir = os.path.join(out_dir, "score_maps")
            mkdir(sm_dir)
            Image.fromarray(score_strip).save(
                os.path.join(sm_dir, f"{fname}_scores.png"))

            raw_dir = os.path.join(out_dir, "score_maps_raw")
            mkdir(raw_dir)
            for si, sm in enumerate(smaps):
                sm_up = F.interpolate(
                    sm, size=(H_target, W_target),
                    mode='bilinear', align_corners=False)
                np.save(os.path.join(raw_dir, f"{fname}_sm{si}.npy"),
                        sm_up.cpu().numpy())

        # metrics  (data_range=1.0 — images normalised to [0,1] by dataset)
        psnr, ssim, rmse = compute_all_metrics(
            out_np, gt_np, data_range=1.0)
        logger.add_sample(mod_name, fname, psnr, ssim, rmse)

        # horizontal concat: [ input | output | label ]
        strip = np.concatenate([lq_np, out_np, gt_np], axis=-1)
        h, w = strip.shape
        raw_path = os.path.join(out_dir, f"{fname}_{w}x{h}.raw")
        strip.astype(np.float32).tofile(raw_path)

    logger.finish_modality(mod_name)

    mean_p, mean_s, mean_r = logger.modality_means(mod_name)
    print(
        f"  {mod_name:<35s}  "
        f"PSNR = {mean_p:.4f}  "
        f"SSIM = {mean_s:.4f}  "
        f"RMSE = {mean_r:.4f}"
    )

logger.close()
print(f"\nDone.  Results in: {os.path.join(save_dir, 'test_results')}")
