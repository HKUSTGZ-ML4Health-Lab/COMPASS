import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# --------------------------------------------------------
# --------------------------------------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# --------------------------------------------------------
# --------------------------------------------------------
class ViT3D(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_depth, image_height, image_width = pair(image_size)
        patch_depth, patch_height, patch_width = pair(patch_size)

        assert image_depth % patch_depth == 0 and image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_depth * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            # Input: (B, C, D, H, W) -> Output: (B, Dim, D/p, H/p, W/p)
            nn.Conv3d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c d h w -> b (d h w) c'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img shape: (Batch, Channels, Depth, Height, Width)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return x


# --------------------------------------------------------
# --------------------------------------------------------

def vit_tiny_3d(in_channels=1, image_size=(128, 128, 128), patch_size=(16, 16, 16), **kwargs):
    return ViT3D(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        channels=in_channels,
        dropout=0.1,
        emb_dropout=0.1,
        **kwargs
    )


def vit_small_3d(in_channels=1, image_size=(128, 128, 128), patch_size=(16, 16, 16), **kwargs):
    return ViT3D(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        channels=in_channels,
        dropout=0.1,
        emb_dropout=0.1,
        **kwargs
    )


def vit_base_3d(in_channels=1, image_size=(128, 128, 128), patch_size=(16, 16, 16), **kwargs):
    return ViT3D(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=in_channels,
        dropout=0.1,
        emb_dropout=0.1,
        **kwargs
    )

