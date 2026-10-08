"""
CLIP text encoder and context decoder for SGF_RWKV cross-modal fusion.

Self-contained — only the modules actually used by SemanticGuidedScoreMap.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_


# ============================================================================
# Primitives
# ============================================================================

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ============================================================================
# Transformer building blocks
# ============================================================================

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int,
                 attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask, dpr[i])
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# ============================================================================
# Cross-attention decoder (used by SemanticGuidedScoreMap)
# ============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    """Cross-modal decoder: text features attend to visual features."""

    def __init__(self, transformer_width=256, transformer_heads=4,
                 transformer_layers=6, visual_dim=1024, text_dim=512,
                 out_dim=512, dropout=0.1, **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout)
            for _ in range(transformer_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = text
        for layer in self.decoder:
            x = layer(x, visual)
        return self.out_proj(x)


# ============================================================================
# CLIP text context encoder (used by SemanticGuidedScoreMap)
# ============================================================================

class CLIPTextContextEncoder(nn.Module):
    """CLIP text encoder with learnable context tokens for domain adaptation."""

    def __init__(self, context_length=22, vocab_size=49408,
                 transformer_width=512, transformer_heads=8,
                 transformer_layers=12, embed_dim=1024, out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.context_length = context_length
        self.embed_dim = embed_dim

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(
                pretrained, map_location='cpu').float().state_dict()
            state_dict = {}
            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                if k in ('positional_embedding', 'text_projection') \
                        or k.startswith('token_embedding') \
                        or k.startswith('ln_final'):
                    if k == 'positional_embedding' \
                            and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                    state_dict[k] = checkpoint[k]
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def forward(self, text, context):
        """text: (K, N1) tokenized text; context: (B, N2, C) learnable tokens."""
        x_text = self.token_embedding(text)        # K, N1, C
        K, N1, C = x_text.shape
        B, N2, _ = context.shape

        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

        x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
        context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

        x = torch.cat(
            [x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2
        ).reshape(B * K, N1 + N2, C)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)                    # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                    # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(B, K, self.embed_dim)
        return x
