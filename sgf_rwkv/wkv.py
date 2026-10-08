"""
WKV CUDA kernel wrapper — efficient linear-attention operator for RWKV.

Compiled JIT from local .cu/.cpp sources on first import.  The kernel
is built for the GPU architecture detected at import time, plus a
fallback PTX path so the binary keeps working on newer GPUs.
"""

import os
import warnings

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load as _cpp_load

T_MAX = 512 * 512

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_arch_flags():
    """Build -gencode flags for the current GPU plus a PTX fallback.

    Returns a list of nvcc flag strings.  Falls back to sm_70 + compute_70
    PTX if no GPU is visible at import time.
    """
    flags = []
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        major, minor = cap[0], cap[1]
        arch = f"compute_{major}{minor}"
        sm = f"sm_{major}{minor}"
        # native architecture (fast path)
        flags.append(f"-gencode=arch={arch},code={sm}")
        # PTX fallback so the binary runs on newer GPUs too
        flags.append(f"-gencode=arch={arch},code={arch}")
    else:
        # no GPU visible at import — compile for Volta+ with PTX fallback
        warnings.warn(
            "No CUDA device found at import time; compiling WKV kernel "
            "for sm_70 with PTX fallback.  This may be slower than a "
            "native build.",
            stacklevel=3,
        )
        flags.append("-gencode=arch=compute_70,code=sm_70")
        flags.append("-gencode=arch=compute_70,code=compute_70")
    return flags


_wkv_cuda = _cpp_load(
    name="wkv_sgf",
    sources=[
        os.path.join(_MODEL_DIR, "cuda", "wkv_op.cpp"),
        os.path.join(_MODEL_DIR, "cuda", "wkv_cuda.cu"),
    ],
    verbose=False,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount=60",
        "--use_fast_math",
        "-O3",
        "-Xptxas=-O3",
        *_build_arch_flags(),
        f"-DTmax={T_MAX}",
    ],
)


class WKV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        assert T <= T_MAX and B * C % min(C, 1024) == 0

        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)

        w2 = w.float().contiguous()
        u2 = u.float().contiguous()
        k2 = k.float().contiguous()
        v2 = v.float().contiguous()

        y = torch.empty((B, T, C), device="cuda",
                        memory_format=torch.contiguous_format)
        _wkv_cuda.forward(B, T, C, w2, u2, k2, v2, y)

        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        assert T <= T_MAX and B * C % min(C, 1024) == 0

        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()

        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16

        _wkv_cuda.backward(
            B, T, C,
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
            gw, gu, gk, gv,
        )

        if half_mode:
            return (
                None, None, None,
                torch.sum(gw.half(), dim=0),
                torch.sum(gu.half(), dim=0),
                gk.half(),
                gv.half(),
            )
        elif bf_mode:
            return (
                None, None, None,
                torch.sum(gw.bfloat16(), dim=0),
                torch.sum(gu.bfloat16(), dim=0),
                gk.bfloat16(),
                gv.bfloat16(),
            )
        else:
            return (
                None, None, None,
                torch.sum(gw, dim=0),
                torch.sum(gu, dim=0),
                gk,
                gv,
            )


def wkv_forward(B, T, C, w, u, k, v):
    """Run the WKV linear-attention kernel.

    All input tensors must already live on the same CUDA device.
    """
    return WKV.apply(B, T, C, w, u, k, v)
