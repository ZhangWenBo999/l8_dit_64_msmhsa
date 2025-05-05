import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from typing import Type
from timm.layers import use_fused_attn

class MSMHSAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            group_kernel_sizes=[3, 5, 7, 9],
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # 每层级独立的卷积和自注意力模块
        self.multi_layer_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
            for k in group_kernel_sizes
        ])
        self.levels = 4
        self.multi_layer_norms = nn.ModuleList([nn.GroupNorm(4, dim) for _ in range(self.levels)])
        self.sa_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        layer_outputs = []
        for conv, norm in zip(self.multi_layer_convs, self.multi_layer_norms):
            layer_out = self.sa_gate(norm(conv(x)))
            layer_outputs.append(layer_out)

        # 多层级融合特征
        fused_features = sum(layer_outputs) / self.levels

        x = fused_features.reshape(B, C, H * W).permute(0, 2, 1)


        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 1024, 1024)
    image = torch.rand(*image_size)
    # Model
    model = MSMHSAttention(1024) # MKLA
    out = model(image)
    print('input_size:', image.size())
    print('output_size:', out.size())