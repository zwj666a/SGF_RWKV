"""
Checkpoint manager: save, resume, best-model tracking.

Stores under:  experiments/<name>/checkpoints/
  latest.pth          — most recent step
  best_psnr.pth       — best PSNR so far
  best_ssim.pth       — best SSIM so far
  best_rmse.pth       — best RMSE so far

RNG state (torch, cuda, numpy, python random) is saved and restored so
that resumed training is bit-exact reproducible.
"""

import os
import random

import numpy as np
import torch


def _collect_rng_state():
    """Snapshot all relevant random states into a dict."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state):
    """Restore random states from a dict produced by _collect_rng_state."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # tracking
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_rmse = float("inf")

    # ── save ────────────────────────────────────────────────────────
    def save(self, model, optimizer, scheduler, iteration, metrics,
             extra_state=None):
        """Save a training checkpoint and update best-metric records.

        Parameters
        ----------
        metrics : dict with keys 'psnr', 'ssim', 'rmse'
        extra_state : dict or None
            Optional additional state to store in the checkpoint
            (e.g. sampler RNG, custom schedulers).
        """
        state = {
            "iteration":    iteration,
            "model_state":  model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_psnr":    self.best_psnr,
            "best_ssim":    self.best_ssim,
            "best_rmse":    self.best_rmse,
            "rng_state":    _collect_rng_state(),
            "extra":        extra_state or {},
        }

        # always save latest
        torch.save(state, os.path.join(self.ckpt_dir, "latest.pth"))

        # update bests
        psnr = metrics.get("psnr", 0.0)
        ssim = metrics.get("ssim", 0.0)
        rmse = metrics.get("rmse", float("inf"))

        if psnr >= self.best_psnr:
            self.best_psnr = psnr
            torch.save(state, os.path.join(self.ckpt_dir, "best_psnr.pth"))

        if ssim >= self.best_ssim:
            self.best_ssim = ssim
            torch.save(state, os.path.join(self.ckpt_dir, "best_ssim.pth"))

        if rmse <= self.best_rmse:
            self.best_rmse = rmse
            torch.save(state, os.path.join(self.ckpt_dir, "best_rmse.pth"))

    # ── resume ──────────────────────────────────────────────────────
    def resume(self, model, optimizer, scheduler, checkpoint_path=None):
        """Try to restore from checkpoint_path, or latest.pth.

        Returns ``(iteration, extra_state)`` where *extra_state* is the
        dict stored via the ``extra_state`` argument of :meth:`save`,
        or an empty dict if not present.  *iteration* is 0 when no
        checkpoint was found.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.ckpt_dir, "latest.pth")

        if not os.path.exists(checkpoint_path):
            print("[Checkpoint] No checkpoint found — starting from scratch.")
            return 0, {}

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        self.best_psnr = state.get("best_psnr", 0.0)
        self.best_ssim = state.get("best_ssim", 0.0)
        self.best_rmse = state.get("best_rmse", float("inf"))
        iteration = state.get("iteration", 0)
        extra = state.get("extra", {})

        # restore RNG state for bit-exact reproducibility
        if "rng_state" in state:
            _restore_rng_state(state["rng_state"])

        print(
            f"[Checkpoint] Resumed from step {iteration}  "
            f"(best PSNR={self.best_psnr:.4f}  SSIM={self.best_ssim:.4f}  "
            f"RMSE={self.best_rmse:.4f})"
        )
        return iteration, extra

    # ── load model only (for testing) ───────────────────────────────
    @staticmethod
    def _remap_legacy_keys(state_dict):
        """Remap legacy state-dict keys to match the current model.

        The module stores token-shift under the attribute ``shift``, but
        older checkpoints used ``DEMS_Shift`` or ``omni_shift``.
        """
        remapped = {}
        for k, v in state_dict.items():
            k = k.replace("omni_shift", "shift")
            k = k.replace("DEMS_Shift", "shift")
            remapped[k] = v
        return remapped

    def load_model_only(self, model, checkpoint_path):
        """Load weights into model, ignoring optimizer/scheduler state.

        Automatically remaps legacy key names (``omni_shift`` / ``DEMS_Shift``
        → ``shift``) for backward compatibility with older checkpoints.
        """
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw = state.get("model_state", state)
        raw = self._remap_legacy_keys(raw)

        missing, unexpected = model.load_state_dict(raw, strict=False)

        if missing:
            print(f"[Checkpoint] MISSING keys ({len(missing)}):")
            for k in missing[:20]:
                print(f"  - {k}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        if unexpected:
            print(f"[Checkpoint] UNEXPECTED keys ({len(unexpected)}):")
            for k in unexpected[:20]:
                print(f"  - {k}")
            if len(unexpected) > 20:
                print(f"  ... and {len(unexpected) - 20} more")

        matched = len(raw) - len(unexpected)
        print(f"[Checkpoint] Loaded model weights from {checkpoint_path}  "
              f"(matched={max(0, matched)}, missing={len(missing)}, unexpected={len(unexpected)})")

    # ── summary ─────────────────────────────────────────────────────
    def best_summary(self):
        return {
            "best_psnr": self.best_psnr,
            "best_ssim": self.best_ssim,
            "best_rmse": self.best_rmse,
        }

    def save_best_summary(self):
        """Write best-achieved values to best_metrics.txt."""
        lines = [
            f"Best PSNR: {self.best_psnr:.6f}",
            f"Best SSIM: {self.best_ssim:.6f}",
            f"Best RMSE: {self.best_rmse:.6f}",
        ]
        path = os.path.join(self.save_dir, "best_metrics.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines))
