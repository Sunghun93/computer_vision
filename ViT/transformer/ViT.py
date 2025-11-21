import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Qt 충돌 방지
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

img = Image.open(os.path.join(BASE_PATH, "data/Cat/1200px-Cat03.jpg"))
# img.show()

transforms = T.Compose([
    T.Resize((224, 224)),  # 16으로 나누어떨어지도록 수정 (224 ÷ 16 = 14)
    T.ToTensor()
])

x = transforms(img)
x = x[None, ...]
print(x.shape)  # torch.Size([1, 3, 224, 224])

patch_size = 16
# rearrange string 값 설명
# input_pattern -> output_pattern
# b: batch size
# c: channels
# h: height divided by patch size
# w: width divided by patch size
# s1: patch size (height)
# s2: patch size (width)
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print(patches.shape)  # torch.Size([1, 196, 768]) - (14*14=196 patches)

# 시각화를 위한 패치 재배열: (batch, num_patches_h, num_patches_w, channels, patch_h, patch_w)
patches_d = rearrange(x, 'b c (h s1) (w s2) -> b h w c s1 s2', s1=patch_size, s2=patch_size)
print(patches_d.shape)  # torch.Size([1, 14, 14, 3, 16, 16])

fig, axes = plt.subplots(nrows=14, ncols=14, figsize=(20, 20))
for i, j in itertools.product(range(14), repeat=2):
    # patches_d[0, i, j]는 (3, 16, 16) 형태, imshow를 위해 (16, 16, 3)로 변환
    patch_img = patches_d[0, i, j].permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    axes[i, j].imshow(patch_img)
    axes[i, j].axis('off')
    axes[i, j].set_title(f'Patch ({i},{j})', fontsize=6)

fig.tight_layout()
# plt.show() 대신 파일로 저장
plt.savefig(os.path.join(BASE_PATH, 'patches_visualization.png'), dpi=150, bbox_inches='tight')
print(f"이미지가 저장되었습니다: {os.path.join(BASE_PATH, 'patches_visualization.png')}")


# 입력 이미지: [B, 3, 224, 224]
#     ↓
# [Rearrange] 패치로 분할 및 평탄화
#     ↓
# [B, 196, 768] (196개 패치, 각 768차원)
#     ↓
# [Linear] Embedding projection
#     ↓
# [B, 196, 768]
#     ↓
# [CLS Token 추가] 앞에 결합
#     ↓
# [B, 197, 768] (1 CLS + 196 patches)
#     ↓
# [Positional Embedding 추가] 위치 정보
#     ↓
# [B, 197, 768] → Transformer Encoder로 전달
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()

        assert img_size / patch_size % 1 == 0, "Image size must be divisible by patch size."

        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_emb = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B, *_ = x.shape
        x = self.projection(x)  # (B, N_PATCHES, EMB_SIZE)
        cls_token = repeat(self.cls_token, '() p e -> b p e', b = B)

        x = torch.cat([cls_token, x], dim=1)  # (B, N_PATCHES + 1, EMB_SIZE)
        x += self.positional_emb  # (B, N_PATCHES + 1, EMB_SIZE)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.emb_size = emb_size

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

        self.projection = nn.Linear(emb_size, emb_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.scaling = (emb_size // num_heads) ** -0.5

    def forward(self, x, mask=None):
        rearrange_heads = 'batch seq_len (num_head h_dim) -> batch num_head seq_len h_dim'

        queries = rearrange(self.query(x), rearrange_heads, num_head=self.num_heads)
        keys = rearrange(self.key(x), rearrange_heads, num_head=self.num_heads)
        values = rearrange(self.value(x), rearrange_heads, num_head=self.num_heads)

        energies = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = -torch.finfo(energies.dtype).min
            energies.mask_fill(~mask, fill_value)

        attention = F.softmax(energies * self.scaling, dim=-1)
        attention = self.attn_dropout(attention)

        out = torch.einsum('bhas, bhsd -> bhad', attention, values)
        out = rearrange(out, 'batch num_head seq_length dim -> batch seq_length (num_head dim)')
        out = self.projection(out)

        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

feed_forward_block = lambda emb_size=768, expansion=4, drop_p=0: nn.Sequential(
    nn.Linear(emb_size, expansion * emb_size),
    nn.GELU(),
    nn.Dropout(drop_p),
    nn.Linear(expansion * emb_size, emb_size),
)

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0, **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    feed_forward_block(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout
                )
            )
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super(TransformerEncoder, self).__init__(
            *(TransformerEncoderBlock(**kwargs) for _ in range(depth))
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, num_classes=1000):
        super(ClassificationHead, self).__init__(
            Reduce('batch_size seq_len emb_dim -> batch_size emb_dim', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

class ViT(nn.Sequential):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, num_classes=1000, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )