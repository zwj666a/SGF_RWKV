"""
Experiment logger: TensorBoard scalars/images + per-modality metric .txt files.
"""

import os
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # TensorBoard
        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # per-modality accumulator (for test-time txt output)
        self._mod_accum = {}

        # summary buffer — collected during testing, written atomically in close()
        self._summary_header = ""
        self._summary_lines = []

        # plain-text training log
        self._log_file = os.path.join(save_dir, "training_log.txt")

    # ── TensorBoard ──────────────────────────────────────────────────
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_dict, step):
        for key, value in tag_dict.items():
            self.writer.add_scalar(key, value, step)

    def log_images(self, tag, images, step):
        """images: (B, C, H, W) tensor."""
        self.writer.add_images(tag, images, step)

    # ── Console + txt log ────────────────────────────────────────────
    def _write_line(self, msg):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(msg)
        with open(self._log_file, "a") as f:
            f.write(line + "\n")

    def log_val(self, iteration, metrics):
        """metrics: dict with psnr, ssim, rmse."""
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        self._write_line(f"[Val  {iteration:>6d}]  " + "  ".join(parts))

    # ── Per-modality test metrics ────────────────────────────────────
    def start_modality(self, modality_name):
        self._mod_accum[modality_name] = dict(
            psnr=[], ssim=[], rmse=[], names=[],
        )

    def add_sample(self, modality_name, file_name, psnr, ssim, rmse):
        acc = self._mod_accum[modality_name]
        acc["psnr"].append(psnr)
        acc["ssim"].append(ssim)
        acc["rmse"].append(rmse)
        acc["names"].append(file_name)

    def modality_means(self, modality_name):
        """Return mean (psnr, ssim, rmse) averages for a modality."""
        acc = self._mod_accum.get(modality_name)
        if acc is None or len(acc["names"]) == 0:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(acc["psnr"])),
            float(np.mean(acc["ssim"])),
            float(np.mean(acc["rmse"])),
        )

    def finish_modality(self, modality_name):
        """Write per-sample + summary .txt for a modality."""
        acc = self._mod_accum.get(modality_name)
        if acc is None:
            return
        n = len(acc["names"])
        if n == 0:
            return

        psnr_arr = np.array(acc["psnr"])
        ssim_arr = np.array(acc["ssim"])
        rmse_arr = np.array(acc["rmse"])

        lines = [f"=== {modality_name} (n={n}) ==="]
        lines.append(f"{'File':<30s}  {'PSNR':>8s}  {'SSIM':>8s}  {'RMSE':>8s}")
        lines.append("-" * 62)
        for i in range(n):
            lines.append(
                f"{acc['names'][i]:<30s}  "
                f"{psnr_arr[i]:8.4f}  {ssim_arr[i]:8.4f}  {rmse_arr[i]:8.4f}"
            )
        lines.append("-" * 62)
        lines.append(
            f"{'MEAN':<30s}  "
            f"{psnr_arr.mean():8.4f}  {ssim_arr.mean():8.4f}  {rmse_arr.mean():8.4f}"
        )
        lines.append("")

        # write per-modality file
        out_path = os.path.join(self.save_dir, "test_results", f"{modality_name}.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        # buffer summary line — written atomically in close()
        self._summary_lines.append(
            f"{modality_name:<35s}  "
            f"PSNR={psnr_arr.mean():.4f}  "
            f"SSIM={ssim_arr.mean():.4f}  "
            f"RMSE={rmse_arr.mean():.4f}"
        )

    def write_summary_header(self):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._summary_header = (
            f"Test run: {stamp}\n"
            f"{'Modality':<35s}  Metrics\n"
            f"{'-' * 70}\n"
        )

    def close(self):
        # write summary atomically — all modalities at once
        if self._summary_header or self._summary_lines:
            sum_path = os.path.join(self.save_dir, "test_results", "summary.txt")
            os.makedirs(os.path.dirname(sum_path), exist_ok=True)
            with open(sum_path, "w") as f:
                f.write(self._summary_header)
                for line in self._summary_lines:
                    f.write(line + "\n")
        self.writer.close()
