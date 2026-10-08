"""
SGF_RWKV — RWKV-based U-Net for Medical Image Restoration.

Core modules:
  - DilatedMultiScaleShift (DMS) — dilated multi-scale token shift
  - FrequencyEnhancementModule (FEM) — FFT magnitude enhancement
  - SemanticGuidedScoreMap (SGM) — CLIP cross-modal fusion
  - Bidirectional WKV scan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.fft

import clip

from .wkv import wkv_forward
from .clip_encoder import CLIPTextContextEncoder, ContextDecoder
from data.dataset import MODALITY_LIST, MODALITY_TO_ID


# ============================================================================
# DilatedMultiScaleShift (DMS) — dilated multi-scale token shift
# ============================================================================

class DilatedMultiScaleShift(nn.Module):
    """
    Dilated Multi-Scale token shift (DMS).

    Dilated depthwise-separable conv branches (d=1,2,3),
    fused by learnable weights with a residual connection.
    """

    def __init__(self, channels, dilations=(1, 2, 3)):
        super().__init__()

        def _make_branch(d):
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
            )

        self.branches = nn.ModuleList([_make_branch(d) for d in dilations])
        self.alpha = nn.Parameter(
            torch.randn(len(dilations)) * 0.1, requires_grad=True)

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        fused = sum(a * f for a, f in zip(self.alpha, feats))
        return x + fused


# ============================================================================
# FrequencyEnhancementModule (FEM) — FFT magnitude enhancement
# ============================================================================

class FrequencyEnhancementModule(nn.Module):
    """
    FFT → magnitude/phase decomposition → process magnitude with
    linear + GELU → reconstruct complex → iFFT → residual add.
    """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                     # B H W C

        xf = torch.fft.fft2(x, dim=(1, 2))
        mag = torch.abs(xf)
        phase = torch.angle(xf)

        mag = F.gelu(self.linear(mag))
        xf = mag * torch.exp(1j * phase)

        out = torch.fft.ifft2(xf, dim=(1, 2)).real     # B H W C
        x_out = x + self.norm(out)
        return x_out.permute(0, 3, 1, 2)               # B C H W


# ============================================================================
# SemanticGuidedScoreMap (SGM) — cross-modal semantic guidance
# ============================================================================

class SemanticGuidedScoreMap(nn.Module):
    """
    Semantic-Guided Score Map (SGM).

    Fuses degradation text embeddings with visual features via
    cross-attention (ContextDecoder). A frozen CLIP text encoder
    encodes degradation descriptions; learnable context tokens
    adapt CLIP to the medical domain.
    """

    def __init__(
        self,
        visual_dim=64,
        num_tasks=5,
        text_dim=512,
        token_embed_dim=512,
        context_length=10,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.context_length = context_length

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, visual_dim),
        )

        ctx_len = 42 - context_length
        self.context_decoder = ContextDecoder(
            context_length=ctx_len,
            transformer_width=visual_dim,
            transformer_heads=4,
            transformer_layers=3,
            visual_dim=visual_dim,
            text_dim=visual_dim,
            out_dim=visual_dim,
            dropout=0.0,
        )

        # tokenised text prompts (frozen, registered as buffer so that
        # .to(device) / state_dict / save & load work through PyTorch)
        self.register_buffer(
            "text_tokens",
            torch.cat([
                clip.tokenize(c, context_length=context_length)
                for c in MODALITY_LIST
            ]),
            persistent=False,
        )
        self.contexts = nn.Parameter(
            torch.randn(1, ctx_len, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)

        self.gamma = nn.Parameter(torch.ones(visual_dim) * 1e-4)

    def forward(self, vision_feat, text_encoder, names):
        B, C, H, W = vision_feat.shape
        vseq = vision_feat.reshape(B, C, -1).permute(0, 2, 1)      # B N C

        # text features from CLIP (text_tokens lives on the same device
        # as the module thanks to register_buffer)
        tfeat = text_encoder(self.text_tokens, self.contexts).expand(B, -1, -1)

        # cross-modal fusion
        tfeat = self.text_proj(tfeat)
        tdiff = self.context_decoder(tfeat, vseq)
        temb = F.normalize(tfeat + self.gamma * tdiff, dim=2, p=2)

        # score map via einsum
        vn = vseq.permute(0, 2, 1).reshape(B, -1, H, W)
        vn = F.normalize(vn, dim=1, p=2)
        smap = torch.einsum("bchw,bkc->bkhw", vn, temb)

        # select the channel corresponding to the given degradation name
        indices = [MODALITY_TO_ID.get(n, 0) for n in names]
        return smap[torch.arange(B), indices, :, :].unsqueeze(1)


# ============================================================================
# VRWKV SpatialMix — configurable spatial attention
# ============================================================================

class VRWKV_SpatialMix(nn.Module):
    """Bidirectional WKV spatial mixing with FEM and DMS token shift."""

    def __init__(self, n_embd, use_fem=True, key_norm=False):
        super().__init__()
        self.recurrence = 2  # row-wise then column-wise scan

        self.fem = FrequencyEnhancementModule(n_embd) if use_fem else None

        # depthwise conv + DMS token shift
        self.dwconv = nn.Conv2d(
            n_embd, n_embd, 3, padding=1, groups=n_embd, bias=False)
        self.shift = DilatedMultiScaleShift(n_embd)

        # attention projections
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd) if key_norm else None
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(
                torch.randn((self.recurrence, n_embd)))
            self.spatial_first = nn.Parameter(
                torch.randn((self.recurrence, n_embd)))

    def _proj(self, x, resolution):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        if self.fem is not None:
            x = self.fem(x)

        x = self.shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        sr = torch.sigmoid(self.receptance(x))
        k = self.key(x)
        v = self.value(x)
        return sr, k, v

    def _final(self, sr, v):
        if self.key_norm is not None:
            v = self.key_norm(v)
        return self.output(sr * v)

    def forward(self, x, resolution):
        B, T, C = x.size()
        h, w = resolution
        sr, k, v = self._proj(x, resolution)

        # bidirectional scan: row pass then column pass
        for j in range(self.recurrence):
            if j % 2 == 0:
                v = wkv_forward(
                    B, T, C,
                    self.spatial_decay[j] / T,
                    self.spatial_first[j] / T,
                    k, v,
                )
            else:
                k2 = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
                v2 = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v2 = wkv_forward(
                    B, T, C,
                    self.spatial_decay[j] / T,
                    self.spatial_first[j] / T,
                    k2, v2,
                )
                v = rearrange(v2, "b (w h) c -> b (h w) c", h=h, w=w)

        return self._final(sr, v)


# ============================================================================
# VRWKV ChannelMix
# ============================================================================

class VRWKV_ChannelMix(nn.Module):
    """Channel-wise feedforward with DMS token shift and squared-ReLU gating."""

    def __init__(self, n_embd, hidden_rate=4, key_norm=False):
        super().__init__()
        hidden = int(hidden_rate * n_embd)

        self.shift = DilatedMultiScaleShift(n_embd)

        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.key_norm = nn.LayerNorm(hidden) if key_norm else None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)

        kv = self.value(k)
        return torch.sigmoid(self.receptance(x)) * kv


# ============================================================================
# Block
# ============================================================================

class Block(nn.Module):
    """Residual block: spatial mix (WKV) + channel mix (MLP)."""

    def __init__(self, n_embd, use_fem=True, hidden_rate=4, key_norm=False):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(
            n_embd, use_fem=use_fem, key_norm=key_norm,
        )
        self.ffn = VRWKV_ChannelMix(
            n_embd, hidden_rate=hidden_rate, key_norm=key_norm,
        )

        self.gamma1 = nn.Parameter(torch.ones(n_embd))
        self.gamma2 = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)

        # SpatialMix
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        # ChannelMix
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x


# ============================================================================
# Downsample / Upsample
# ============================================================================

class Downsample(nn.Module):

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ============================================================================
# SGF_RWKV —  RWKV U-Net for Medical Image Restoration
# ============================================================================

class SGF_RWKV(nn.Module):
    """
    SGF_RWKV: RWKV U-Net with DilatedMultiScaleShift, FrequencyEnhancementModule,
    and SemanticGuidedScoreMap.

    Parameters
    ----------
    inp_channels, out_channels : int
        I/O channel count (default 1 for grayscale medical images).
    dim : int
        Base channel width (default 48).
    num_blocks : tuple of 4 ints
        Blocks per encoder level.
    num_refinement_blocks : int
        Refinement blocks after decoder.
    clip_model_path : str
        Path to pretrained CLIP ViT-B-32 weights.
    """

    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        clip_model_path="./pretrained/ViT-B-32.pt",
    ):
        super().__init__()

        d1 = dim
        d2 = dim * 2
        d3 = dim * 4
        d4 = dim * 8

        self.dim = dim

        # ── block factory ──────────────────────────────────────────
        def _blocks(ch, n):
            return nn.Sequential(*[
                Block(ch, use_fem=True)
                for _ in range(n)
            ])

        # ── encoder ────────────────────────────────────────────────
        self.patch_embed = nn.Conv2d(inp_channels, d1, 3, padding=1, bias=True)
        self.encoder_level1 = _blocks(d1, num_blocks[0])
        self.down1_2 = Downsample(d1)
        self.encoder_level2 = _blocks(d2, num_blocks[1])
        self.down2_3 = Downsample(d2)
        self.encoder_level3 = _blocks(d3, num_blocks[2])
        self.down3_4 = Downsample(d3)
        self.latent = _blocks(d4, num_blocks[3])

        # ── CLIP text encoder (frozen) ─────────────────────────────
        self.text_encoder = CLIPTextContextEncoder(
            context_length=42, embed_dim=512,
            transformer_width=512, transformer_heads=8,
            transformer_layers=12, pretrained=clip_model_path,
        )
        self.text_encoder.init_weights()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # ── score-map modules (3 levels) ───────────────────────────
        self.score_module_list = nn.ModuleList([
            SemanticGuidedScoreMap(visual_dim=d1),
            SemanticGuidedScoreMap(visual_dim=d2),
            SemanticGuidedScoreMap(visual_dim=d3),
        ])

        # ── decoder ────────────────────────────────────────────────
        self.up4_3 = Upsample(d4)
        self.reduce_chan_level3 = nn.Conv2d(
            d3 * 2 + d3, d3, 1, bias=True)
        self.decoder_level3 = _blocks(d3, num_blocks[2])

        self.up3_2 = Upsample(d3)
        self.reduce_chan_level2 = nn.Conv2d(
            d2 * 2 + d2, d2, 1, bias=True)
        self.decoder_level2 = _blocks(d2, num_blocks[1])

        self.up2_1 = Upsample(d2)
        self.reduce_chan_level1 = nn.Conv2d(
            d1 * 2 + d1, d2, 1, bias=True)
        self.decoder_level1 = _blocks(d2, num_blocks[0])

        # ── refinement + output ────────────────────────────────────
        self.refinement = _blocks(d2, num_refinement_blocks)
        self.output = nn.Conv2d(d2, out_channels, 3, padding=1, bias=True)

    def forward(self, inp_img, names=None):
        # encoder
        e1 = self.encoder_level1(self.patch_embed(inp_img))
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        lat = self.latent(self.down3_4(e3))

        # score maps
        sms = [
            m(feat, self.text_encoder, names)
            for m, feat in zip(self.score_module_list, [e1, e2, e3])
        ]

        # ── decoder with skip + score-map concatenation ─────────────
        def _cat(x, skip, idx):
            return torch.cat([
                x, skip,
                sms[idx].repeat(1, skip.shape[1], 1, 1),
            ], dim=1)

        d3 = self.decoder_level3(
            self.reduce_chan_level3(_cat(self.up4_3(lat), e3, 2)))
        d2 = self.decoder_level2(
            self.reduce_chan_level2(_cat(self.up3_2(d3), e2, 1)))
        d1 = self.reduce_chan_level1(_cat(self.up2_1(d2), e1, 0))

        out = self.output(self.refinement(self.decoder_level1(d1))) + inp_img
        return out, sms


# ============================================================================
# Utility
# ============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
