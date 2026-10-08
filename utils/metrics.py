"""
Image quality metrics: PSNR, SSIM, RMSE.

Test-time evaluation on numpy arrays, with a differentiable SSIM
fallback that also serves as the training loss component.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def compute_all_metrics(pred, target, data_range=1.0):
    """Return (PSNR, SSIM, RMSE) for numpy arrays (H, W) or (1, H, W)."""
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()

    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

    rmse = float(np.sqrt(mse))

    try:
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(
            pred, target, data_range=data_range,
            gaussian_weights=True, win_size=11,
            K1=0.01, K2=0.03,
        )
    except ImportError:
        ssim = _manual_ssim(
            torch.from_numpy(pred), torch.from_numpy(target),
            data_range=data_range,
        )

    return psnr, ssim, rmse


def _gaussian_window(window_size, sigma, channel):
    gauss = torch.Tensor([
        math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def _manual_ssim(img1, img2, data_range=1.0, window_size=11, channel=1):
    """Differentiable SSIM using a gaussian window. Returns float."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window = _gaussian_window(window_size, 1.5, channel).to(img1.device)

    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()
