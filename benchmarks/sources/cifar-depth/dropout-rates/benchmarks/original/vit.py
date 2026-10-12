"""ViT shared by the ablation notebook and original seed-extension runner.

provenance.json records the historical extraction, before patch_size became
an explicit constructor argument. The default retains the original model.
"""

import torch
import torch.nn as nn

PATCH_SIZE = 4


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=PATCH_SIZE, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        h = int(dim * ratio)
        self.fc1, self.fc2 = nn.Linear(dim, h), nn.Linear(h, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer block with separate dropout for attention and MLP.

    Args:
        dim: embedding dimension
        heads: number of attention heads
        ratio: MLP expansion ratio
        drop_attn: dropout rate for attention sublayer (post-attention residual)
        drop_mlp: dropout rate for MLP sublayer (post-MLP residual)
    """

    def __init__(self, dim, heads, ratio=4.0, drop_attn=0.0, drop_mlp=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.mlp = Attention(dim, heads), MLP(dim, ratio)
        self.drop_attn = nn.Dropout(drop_attn)
        self.drop_mlp = nn.Dropout(drop_mlp)

    def forward(self, x):
        x = x + self.drop_attn(self.attn(self.norm1(x)))
        x = x + self.drop_mlp(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """Vision Transformer with ablation support for dropout location.

    Args:
        depth: number of transformer blocks
        dim: embedding dimension
        heads: number of attention heads
        ratio: MLP expansion ratio
        h_layers: list of dropout rates per layer
        ablation_mode: 'both', 'attn_only', or 'mlp_only'
    """

    def __init__(
        self,
        depth=12,
        dim=384,
        heads=6,
        ratio=4.0,
        h_layers=None,
        ablation_mode="both",
        *,
        patch_size=PATCH_SIZE,
    ):
        super().__init__()
        self.patch = PatchEmbed(32, patch_size, dim)
        n = self.patch.num_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        h_layers = h_layers or [0.0] * depth

        # Create blocks with ablation-specific dropout
        blocks = []
        for i in range(depth):
            h = h_layers[i]
            if ablation_mode == "both":
                drop_attn, drop_mlp = h, h
            elif ablation_mode == "attn_only":
                drop_attn, drop_mlp = h, 0.0
            elif ablation_mode == "mlp_only":
                drop_attn, drop_mlp = 0.0, h
            else:
                raise ValueError(f"Unknown ablation_mode: {ablation_mode}")
            blocks.append(Block(dim, heads, ratio, drop_attn, drop_mlp))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])
