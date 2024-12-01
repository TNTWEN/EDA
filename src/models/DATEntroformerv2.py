import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.ops import quantize_ste
import math
from src.models.google import  conv, deconv, update_registered_buffers,CompressionModel,get_scale_table
from src.entropy_models import GaussianConditional
from CodeC.ans import BufferedRansEncoder, RansDecoder
from src.layers import GDN
import einops
from timm.models.layers import to_2tuple, trunc_normal_


#from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction
from natten.functional import natten2dav, natten2dqkrpb
from torch.nn.functional import pad

# class NeighborhoodAttention2D(nn.Module):
#     """
#     Neighborhood Attention 2D Module
#     """
#     def __init__(self,config, kernel_size,  attn_drop=0., proj_drop=0.,
#                  dilation=None):
#         super().__init__()
#         self.fp16_enabled = False
#         self.dim = config.dim
#         self.num_heads = config.num_heads
#         self.head_dim = config.dim // self.num_heads
#         self.scale = self.head_dim ** -0.5
#         assert kernel_size > 1 and kernel_size % 2 == 1, \
#             f"Kernel size must be an odd number greater than 1, got {kernel_size}."
#         assert kernel_size in [3, 5, 7, 9, 11, 13], \
#             f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
#         self.kernel_size = kernel_size
#         if type(dilation) is str:
#             self.dilation = None
#             self.window_size = None
#         else:
#             assert dilation is None or dilation >= 1, \
#                 f"Dilation must be greater than or equal to 1, got {dilation}."
#             self.dilation = dilation or 1
#             self.window_size = self.kernel_size * self.dilation

#         self.qkv = nn.Linear(self.dim, self.dim * 3)
#         self.rpb = nn.Parameter(torch.zeros(self.num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
#         trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(self.dim, self.dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
#         x = x.permute(0, 2, 3, 1)
#         B, Hp, Wp, C = x.shape
#         H, W = int(Hp), int(Wp)
#         pad_l = pad_t = pad_r = pad_b = 0
#         dilation = self.dilation
#         window_size = self.window_size
#         if window_size is None:
#             dilation = max(min(H, W) // self.kernel_size, 1)
#             window_size = dilation * self.kernel_size
#         if H < window_size or W < window_size:
#             pad_l = pad_t = 0
#             pad_r = max(0, window_size - W)
#             pad_b = max(0, window_size - H)
#             x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
#             _, H, W, _ = x.shape
#         qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = q * self.scale
#         # breakpoint()
#         attn = NATTEN2DQKRPBFunction.apply(q, k, self.rpb, self.kernel_size, dilation)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = NATTEN2DAVFunction.apply(attn, v, self.kernel_size, dilation)
#         x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
#         if pad_r or pad_b:
#             x = x[:, :Hp, :Wp, :]

#         return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2)







class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        config,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(self.num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2)















def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class UpPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, scale=1, padding=None, groups=1):
        super(UpPixelShuffle, self).__init__()
        padding = kernel_size//2 if padding is None else padding
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels*(scale**2),
                                kernel_size=kernel_size,
                                padding=padding,
                                padding_mode='zeros',
                                groups=groups)
        self.up = nn.PixelShuffle(scale)

    def forward(self,x):
        out = self.conv2d(x)
        out = self.up(out)
        return out
    

class Config():
    def __init__(
        self,
        dim=384,
        num_layers=6,
        num_heads=6,
        dim_head=64,
        relative_attention_num_buckets=5,
        n_groups=3,
        dropout_rate=0.,
        mlp_ratio=4,
        mask_ratio=0.,
        proj_drop=0.,
        attn_drop=0.,
    ):

        self.dim = dim
        self.dim_head = dim_head

        self.num_layers = num_layers

        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.n_groups = n_groups

        self.dropout_rate = dropout_rate
        self.mlp_ratio = mlp_ratio
        self.mask_ratio = mask_ratio
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop


    @property
    def hidden_size(self):
        return self.dim

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class Attention(nn.Module):
    def __init__(self, config,ksize,stride):



        super().__init__()
        self.dim = config.dim
        self.key_value_proj_dim = config.dim_head
        self.n_heads = config.num_heads
        self.scale = self.key_value_proj_dim ** -0.5 
        self.ksize = ksize
        self.stride = stride
        self.n_groups = config.n_groups
        self.n_group_channels = self.dim // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        assert self.relative_attention_num_buckets%2 == 1

        kk = self.ksize
        pad_size = kk // 2 if kk != self.stride  else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, self.stride , pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.dim, self.dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.dim, self.dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.dim, self.dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.dim, self.dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(config.proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(config.attn_drop, inplace=True)
        self.rpe_table = nn.Parameter(
            torch.zeros(self.n_heads, self.relative_attention_num_buckets, self.relative_attention_num_buckets)
            )
        trunc_normal_(self.rpe_table, std=0.01)



    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

        

    def forward(self,x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        pos = (offset + reference).clamp(-1., +1.)
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
        q = q.reshape(B * self.n_heads, self.key_value_proj_dim, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.key_value_proj_dim, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.key_value_proj_dim, n_sample)


        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)


        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        q_grid = self._get_q_grid(H, W, B, dtype, device)

        num_buckets = self.relative_attention_num_buckets
        num_buckets_half = num_buckets // 2

        q_grid = q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2)
        q_grid1 = q_grid.clone()
        q_grid1[...,1]=((q_grid[...,1]+1.0)/2.0)*(W+1.0)
        q_grid1[...,0]=((q_grid[...,0]+1.0)/2.0)*(H+1.0)   


        # pos1 = pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
        # pos1[...,1]=pos1.clone()[...,1].add_(1.0).div_(2.0).mul_(W+1.0)
        # pos1[...,0]=pos1.clone()[...,0].add_(1.0).div_(2.0).mul_(H+1.0)

        pos = pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
        pos1 = pos.clone()
        pos1[...,1]=((pos1[...,1]+1.0)/2.0)*(W+1.0)
        pos1[...,0]=((pos1[...,0]+1.0)/2.0)*(H+1.0)   

        displacement = q_grid1-pos1
        hamming_distance = torch.abs(displacement[...,0])+torch.abs(displacement[...,1])
        is_small = hamming_distance <= num_buckets_half
        result = torch.zeros_like(displacement)
        result[...,0] = torch.where(is_small,displacement[...,0],num_buckets_half)
        result[...,1] = torch.where(is_small,displacement[...,1],num_buckets_half)
        result[..., 1].div_(num_buckets - 1.0).mul_(2.0).sub_(1.0)
        result[..., 0].div_(num_buckets - 1.0).mul_(2.0).sub_(1.0)
    
        attn_bias = F.grid_sample(
            input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
            grid=result[..., (1, 0)],
            mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
        attn = attn + attn_bias


        # if mask is not None:
        #     mask_value = -torch.finfo(attn.dtype).max
        #     assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
        #     mask = rearrange(mask, 'b i j -> b () i j')
        #     attn.masked_fill_(~mask, mask_value)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormProxy(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim*mlp_ratio,1,1,0),
            # nn.GELU(),  # modified
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim*mlp_ratio, dim,1,1,0),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, config,ksize,stride):
        super().__init__()
        self.SelfAttention = Attention(config,ksize,stride)
        self.layer_norm = LayerNormProxy(config.dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,

    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states
        )
        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states


class NeighborhoodAttentionBlock(nn.Module):
    def __init__(self, config,kernel_size):
        super().__init__()
        self.SelfAttention = NeighborhoodAttention2D(config,kernel_size)
        self.layer_norm = LayerNormProxy(config.dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,

    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states
        )
        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states

class Block(nn.Module):
    def __init__(self, config,ksize,stride):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(AttentionBlock(config,ksize,stride))
        self.layer.append(PreNorm(config.dim, FeedForward(config.dim, config.mlp_ratio, config.dropout_rate)))

    def forward(
        self,
        hidden_states,
    ):
        hidden_states = self.layer[0](
            hidden_states,
        )

        # Apply Feed Forward layer.
        hidden_states = hidden_states + self.layer[-1](hidden_states)

        return hidden_states # hidden-states


class NeighborhoodBlock(nn.Module):
    def __init__(self, config,kernel_size):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(NeighborhoodAttentionBlock(config,kernel_size))
        self.layer.append(PreNorm(config.dim, FeedForward(config.dim, config.mlp_ratio, config.dropout_rate)))

    def forward(
        self,
        hidden_states,
    ):
        hidden_states = self.layer[0](
            hidden_states,
        )

        # Apply Feed Forward layer.
        hidden_states = hidden_states + self.layer[-1](hidden_states)

        return hidden_states # hidden-states


class LocalGlobal(nn.Module):
    def __init__(self, config,local_kernel_size,global_kernel_size,global_stride):
        super().__init__()
        self.local_block = NeighborhoodBlock(config,kernel_size=local_kernel_size)
        self.global_block = Block(config,global_kernel_size,global_stride)
        self.merge = nn.Sequential(
            nn.Conv2d(config.dim*2,config.dim,1,1,0),
            nn.LeakyReLU(0.2),
        )

    def forward(
        self,
        hidden_states,
        ):
        local_states = self.local_block(hidden_states)
        global_states = self.global_block(hidden_states)
        hidden_states = self.merge(torch.cat((local_states,global_states), dim=1) )

        return hidden_states

class TransHyperScale(nn.Module):

    #
    def __init__(self, cin=0, cout=0,scale=2, down=True):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.scale =scale

        self.dim = 384
        self.num_layers = 6
        self.num_heads = 6
        self.dim_head = 64
        self.proj_drop=0.
        self.attn_drop=0.
        self.dropout_rate=0.
        self.mlp_ratio=4
        self.mask_ratio=0.
        self.relative_attention_num_buckets=7
        self.down = down
        self.n_groups = 3
        self.ksizes = [7,5,3]
        self.strides= [4,2,1]

        self.config = Config(

            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            proj_drop=self.proj_drop,
            attn_drop=self.attn_drop,
            dropout_rate=self.dropout_rate,
            mlp_ratio=self.mlp_ratio,
            mask_ratio=self.mask_ratio,
            n_groups = self.n_groups

        )

        self.build()

    def build(self):
        # Head projection and out projection if needed
        self.to_patch_embedding = nn.Conv2d(self.cin, self.config.dim,1,1,0) if self.cin else nn.Identity()
        self.mlp_head = nn.Sequential(LayerNormProxy(self.config.dim), nn.Conv2d(self.config.dim, self.cout,1,1,0)) if self.cout else nn.Identity()
        num_each_stage = self.config.num_layers // 2 // (self.scale+1)
        # Down \ Up scale blocks. modified
        if(self.down):
            self.scale_blocks = clones(nn.Conv2d(self.config.dim, self.config.dim, 3, 2, 1, groups=1), self.scale)
        else:
            self.scale_blocks = clones(UpPixelShuffle(self.config.dim, self.config.dim, kernel_size=3, scale=2), self.scale)

        self.trans_blocks = nn.ModuleList()

        for i in range(self.scale):
            block_scale = nn.ModuleList()
            for _ in range(num_each_stage):
                block_scale.append(NeighborhoodBlock(self.config,kernel_size=7))
                #block_scale.append(Block(self.config,self.ksizes[i],self.strides[i]))
                #block_scale.append(LocalGlobal(self.config,local_kernel_size=7,global_kernel_size=self.ksizes[i],global_stride=self.strides[i]))
            #self.trans_blocks.append(block_scale)

        block_scale = nn.ModuleList()
        for _ in range(num_each_stage):
            block_scale.append(NeighborhoodBlock(self.config,kernel_size=7))
        self.trans_blocks.append(block_scale)
        block_scale = nn.ModuleList()
        for _ in range(num_each_stage):
            block_scale.append(NeighborhoodBlock(self.config,kernel_size=5))
        self.trans_blocks.append(block_scale)
            # next_num = self.config.relative_attention_num_buckets//2
            # next_num = next_num if next_num%2 == 1 else next_num + 1
            # self.config.relative_attention_num_buckets = max(next_num, 5)

        block_scale = nn.ModuleList()
        for _ in range(num_each_stage):
            #block_scale.append(Block(self.config,self.ksizes[-1],self.strides[-1]))
            #block_scale.append(Block(self.config,self.ksizes[-1],self.strides[-1]))
            block_scale.append(NeighborhoodBlock(self.config,kernel_size=3))

        self.trans_blocks.append(block_scale)



        
        if not self.down:
            self.trans_blocks = self.trans_blocks[::-1]

    def forward(self, x):
        batch_size, channels, height, width  = x.shape   # input_shape
        seq_length = height * width


        # Input Embedding
        inputs_embeds = self.to_patch_embedding(x)

        # Init state
        # encoder_decoder_position_bias = None
        hidden_states = inputs_embeds

            
        for i, scale_layer in enumerate(self.scale_blocks):
            for _, layer_module in enumerate(self.trans_blocks[i]):
                # Transformer block
                hidden_states = layer_module(
                    hidden_states,
                )


            hidden_states = scale_layer(hidden_states)
            if(self.down):
                height, width = height//2, width//2
            else:
                height, width = height*2, width*2



        for _, layer_module in enumerate(self.trans_blocks[-1]):
            # Transformer block
            hidden_states  = layer_module(
                hidden_states,

            )

        # Out projection
        out = self.mlp_head(hidden_states)
        return out







class TransDecoder(nn.Module):
    def __init__(self, cin=0, cout=0):
        super().__init__()
        self.cin = cin
        self.cout = cout

        self.dim = 384
        self.num_layers = 6
        self.num_heads = 6
        self.dim_head = 64
        self.proj_drop=0.
        self.attn_drop=0.
        self.dropout_rate=0.
        self.mlp_ratio=4
        self.mask_ratio=0.
        self.relative_attention_num_buckets=7
        self.n_groups = 3
        self.ksizes = 7
        self.strides= 4
        self.scan_mode = 'checkboard'

        self.config = Config(

            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            proj_drop=self.proj_drop,
            attn_drop=self.attn_drop,
            dropout_rate=self.dropout_rate,
            mlp_ratio=self.mlp_ratio,
            mask_ratio=self.mask_ratio,
            n_groups = self.n_groups

        )
        self.build()


    def build(self):
        self.to_patch_embedding = nn.Conv2d(self.cin, self.config.dim,1,1,0) if self.cin else nn.Identity()
        self.mlp_head = nn.Sequential(LayerNormProxy(self.config.dim), nn.Conv2d(self.config.dim, self.cout,1,1,0)) if self.cout else nn.Identity()

        self.blocks = nn.ModuleList()
        for _ in range(self.config.num_layers):
            self.blocks.append(NeighborhoodBlock(self.config,kernel_size=7))
            #self.blocks.append(Block(self.config,self.ksizes,self.strides))
            #self.blocks.append(LocalGlobal(self.config,local_kernel_size=7,global_kernel_size=self.ksizes,global_stride=self.strides))

    def forward(self, x):
        x = x.clone()
        batch_size, channels, height, width  = x.shape   # input_shape

        # Self-attention Mask & Token Mask

        mask, token_mask, input_mask, output_mask = self.get_mask(batch_size, height, width)

        mask, input_mask, output_mask = mask.to(x.device), input_mask.to(x.device), output_mask.to(x.device)
        token_mask = token_mask.to(x.device) if token_mask is not None else token_mask

        # Mask Input
        x.masked_fill_(~input_mask, 0.)
        # Input Embedding
        hidden_states = self.to_patch_embedding(x)



        for _, layer_module in enumerate(self.blocks):
            # Transformer block
            hidden_states = layer_module(
                hidden_states,

            )


        # Out projection
        out = self.mlp_head(hidden_states)
        # Mask output
        out.masked_fill_(~output_mask, 0.)
        return out            

    def get_mask(self, b, h, w):
        n = h*w

        token_mask = None
        mask_checkboard = torch.ones((h, w)).bool()
        mask_checkboard[0::2, 0::2] = 0
        mask_checkboard[1::2, 1::2] = 0
        input_mask = mask_checkboard.clone()
        output_mask = ~mask_checkboard.clone()
        mask = repeat(mask_checkboard.view(1,-1), '() n -> d n', d=n)
        mask = mask | torch.eye(n).bool()

        mask = repeat(mask.unsqueeze(0), '() d n -> b d n', b=b)
        token_mask = token_mask  # torch.ones_like(mask).bool()
        input_mask = repeat(input_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=self.cin)
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(output_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=channel)
        return mask, token_mask, input_mask, output_mask







class DATEntroformerv2(CompressionModel):
    def __init__(self, N=192, M=384, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)    

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.cit_he = TransHyperScale(cin=384, cout=192, scale=2, down=True)
        self.cit_hd = TransHyperScale(cin=192, scale=2, down=False)
        self.cit_ar = TransDecoder(cin=384)
        self.cit_pn = torch.nn.Sequential(
            nn.Conv2d(384*2, 384*4, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384*4, 384*1*2, 1, 1, 0),
        )
        self.gaussian_conditional = GaussianConditional(None)
        # Init modules
        self.g_a .apply(xavier_uniform_init)
        self.g_s.apply(xavier_uniform_init)
        self.cit_he.apply(vit2_init)
        self.cit_hd.apply(vit2_init)
        self.cit_ar.apply(vit2_init)
        self.cit_pn.apply(xavier_uniform_init) 

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def forward(self, x):
        y = self.g_a(x)
        z = self.cit_he(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.cit_hd(z_hat)

        _,_, yh, yw = y.shape
        _, _, _, mask = self.cit_ar.get_mask(1, yh, yw)

        mask = mask.to(y.device)
        ctx_params = torch.zeros_like(y).to(y.device)
        gaussian_params = self.cit_pn(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat1, means_hat1 = gaussian_params.chunk(2, 1)

        means_hat1=means_hat1.masked_fill(mask, 0.)
        scales_hat1=scales_hat1.masked_fill(mask, 0.)

        y1=quantize_ste(y - means_hat1) + means_hat1

        ctx_params = self.cit_ar(y1)
        gaussian_params = self.cit_pn(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat2, means_hat2 = gaussian_params.chunk(2, 1)

        means_hat2=means_hat2.masked_fill(~mask, 0.)
        scales_hat2=scales_hat2.masked_fill(~mask, 0.)
        means_hat,scales_hat = means_hat1+means_hat2,scales_hat1+scales_hat2
        y_hat=quantize_ste(y - means_hat) + means_hat
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)##
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }



    def compress(self,x):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        y_strings = []
        y = self.g_a(x)
        _,_, yh, yw = y.shape
        z = self.cit_he(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.cit_hd(z_hat)

        _, _, _, mask = self.cit_ar.get_mask(1, yh, yw)
        y_hat = torch.zeros_like(y).to(y.device)

        #ctx_params = self.cit_ar(y_hat)
        gaussian_params = self.cit_pn(
            torch.cat((params, y_hat), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y1_slice_h,y1_slice_w= torch.where(mask[0,0]== False)

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        for pos in range(y1_slice_h.size(0)):
            pos_h=y1_slice_h[pos]
            pos_w=y1_slice_w[pos]
            scale = scales_hat[:,:,pos_h,pos_w]
            y_q = y[:,:,pos_h,pos_w]
            mean = means_hat[:,:,pos_h,pos_w]
            indexes = self.gaussian_conditional.build_indexes(scale)
            y_q = self.gaussian_conditional.quantize(y_q, "symbols", mean)
            y_hat[:,:,pos_h,pos_w] = y_q+mean

            symbols_list.extend(y_q.squeeze().tolist())
            indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        y_strings.append(string)




        symbols_list = []
        indexes_list = []        

        ctx_params = self.cit_ar(y_hat)
        gaussian_params = self.cit_pn(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y2_slice_h,y2_slice_w= torch.where(mask[0,0]== True)



        for pos in range(y2_slice_h.size(0)):
            pos_h=y2_slice_h[pos]
            pos_w=y2_slice_w[pos]
            scale = scales_hat[:,:,pos_h,pos_w]
            y_q = y[:,:,pos_h,pos_w]
            mean = means_hat[:,:,pos_h,pos_w]
            indexes = self.gaussian_conditional.build_indexes(scale)
            y_q = self.gaussian_conditional.quantize(y_q, "symbols", mean)
            y_hat[:,:,pos_h,pos_w] = y_q+mean

            symbols_list.extend(y_q.squeeze().tolist())
            indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        y_strings.append(string)


        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}





    def decompress(self,strings,shape):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()



        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.cit_hd(z_hat)
        
        yh,yw = shape
        yh = yh*4
        yw = yw*4
        _, _, _, mask = self.cit_ar.get_mask(1, yh, yw)
        y_hat = torch.zeros(1,384,yh,yw).to(z_hat.device)

        #ctx_params = self.cit_ar(y_hat)
        gaussian_params = self.cit_pn(
            torch.cat((params, y_hat), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y1_slice_h,y1_slice_w= torch.where(mask[0,0]== False)
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        for pos in range(y1_slice_h.size(0)):
            pos_h=y1_slice_h[pos]
            pos_w=y1_slice_w[pos]
            scale = scales_hat[:,:,pos_h,pos_w]
            mean = means_hat[:,:,pos_h,pos_w]

            indexes = self.gaussian_conditional.build_indexes(scale)
            rv = decoder.decode_stream(
                indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1)
            rv = self.gaussian_conditional.dequantize(rv, mean)
            y_hat[:,:,pos_h,pos_w] = rv

        decoder.set_stream(strings[0][1])
        ctx_params = self.cit_ar(y_hat)
        gaussian_params = self.cit_pn(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y2_slice_h,y2_slice_w= torch.where(mask[0,0]== True)
        for pos in range(y2_slice_h.size(0)):
            pos_h=y2_slice_h[pos]
            pos_w=y2_slice_w[pos]
            scale = scales_hat[:,:,pos_h,pos_w]
            mean = means_hat[:,:,pos_h,pos_w]

            indexes = self.gaussian_conditional.build_indexes(scale)
            rv = decoder.decode_stream(
                indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1)
            rv = self.gaussian_conditional.dequantize(rv, mean)
            y_hat[:,:,pos_h,pos_w] = rv
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}





    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


def xavier_uniform_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)    
    else:
        pass  # print("Not Initial:", classname)





def _no_grad_trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()
        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def vit2_init(m, head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    https://github.com/rwightman/pytorch-image-models/blob/9a1bd358c7e998799eed88b29842e3c9e5483e34/timm/models/vision_transformer.py
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear):
        _no_grad_trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    else:
        pass  # print("Not Initial:", classname)
