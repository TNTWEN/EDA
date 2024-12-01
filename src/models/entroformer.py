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


class Config():
    def __init__(
        self,
        debug=False,
        dim=384,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=6,
        dim_head=64,
        relative_attention_num_buckets=5,
        dropout_rate=0.,
        initializer_factor=1.0,
        is_decoder=True,
        scale=True,
        mlp_ratio=4,
        mask_ratio=0.,
        manual_init_bias=True,
        rpe_mode='contextualproduct',  # 'bias', 'contextual'
        # rpe_dist='cross',  # 'hamming', product'
        attn_topk=-1,
    ):
        # self.vocab_size = vocab_size
        self.debug = debug
        self.dim = dim
        self.dim_head = dim_head
        # self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        # self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        # self.feed_forward_proj = feed_forward_proj
        # self.use_cache = use_cache
        # self.gradient_checkpointing = gradient_checkpointing
        self.scale = scale
        self.is_decoder = is_decoder
        self.mlp_ratio = mlp_ratio
        self.mask_ratio = mask_ratio
        self.manual_init_bias = manual_init_bias
        self.rpe_mode = rpe_mode
        # self.rpe_dist = rpe_dist
        self.attn_topk = attn_topk

    @property
    def hidden_size(self):
        return self.dim

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            # nn.GELU(),  # modified
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim*mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        assert self.relative_attention_num_buckets%2 == 1

        self.dim = config.dim
        self.key_value_proj_dim = config.dim_head
        self.n_heads = config.num_heads
        self.n_heads_rpe = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.scale = self.key_value_proj_dim ** -0.5 if config.scale else 1.
        self.config = config
        self.attn = None
        self.score = None
        self.build()
        # self.contextual_position = None  # for debug

    def build(self):
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.qkv = nn.Linear(self.dim, self.inner_dim*3, bias=False)
        self.o = nn.Linear(self.inner_dim, self.dim, bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets**2, self.key_value_proj_dim)
            # self.relative_buckets = None
            # self.query_shape, self.key_shape = None, None
        
    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        num_buckets = self.relative_attention_num_buckets
        num_buckets_half = num_buckets // 2

        # distance matrix
        context_position_v = torch.arange(query_length[0], dtype=torch.long)[:, None]
        memory_position_v = torch.arange(key_length[0], dtype=torch.long)[None, :]
        relative_position_v = memory_position_v - context_position_v  # shape (query_length[0], key_length[0])
        context_position_h = torch.arange(query_length[1], dtype=torch.long)[:, None]
        memory_position_h = torch.arange(key_length[1], dtype=torch.long)[None, :]
        relative_position_h = memory_position_h - context_position_h  # shape (query_length[1], key_length[1])

        # expand to 2D
        relative_position_v = relative_position_v.repeat(query_length[1],key_length[1]).view(query_length[1], query_length[0], key_length[1], key_length[0])
        relative_position_v = relative_position_v.permute(1,0,3,2).contiguous().view(query_length[0]*query_length[1], -1)
        relative_position_h = relative_position_h.repeat(query_length[0],key_length[0]).view(query_length[0]*query_length[1], -1)

        # L1 distance boundary
        hamming_distance = torch.abs(relative_position_h) + torch.abs(relative_position_v)
        is_small = hamming_distance <= num_buckets_half
        relative_postion_if_small = torch.full_like(relative_position_v, 0)

        # index and clamp
        relative_buckets = (relative_position_v + num_buckets_half)*num_buckets + (relative_position_h + num_buckets_half)
        relative_buckets = torch.where(is_small, relative_buckets, relative_postion_if_small)
        relative_buckets = relative_buckets.to(self.relative_attention_bias.weight.device)

        # 取 Position Embedding
        values = self.relative_attention_bias(relative_buckets)  # shape (query_length, key_length, inner_dim)
        return values

    def forward(
        self,
        hidden_states,
        query_shape_2d,
        key_shape_2d,
        mask=None,
        position_bias=None,
        topk=-1,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        int_seq_length = int(seq_length)
        real_seq_length = seq_length

        key_length = real_seq_length

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        # get query \ key \ value states
        # query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        # key_states = shape(self.k(hidden_states))
        # value_states = shape(self.v(hidden_states))

        qkv = self.qkv(hidden_states).reshape(batch_size, -1, 3)
        query_states, key_states, value_states = shape(qkv[...,0]), shape(qkv[...,1]), shape(qkv[...,2])

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        
        # Position Bias: (seq_length, key_length, dim_per_head)
        if position_bias is None:
            position_bias = self.compute_bias(query_shape_2d, key_shape_2d)

        # Contextual Mode.
        rearrange_q = rearrange(query_states, 'b n q d -> q (b n) d')
        contextual_position = torch.matmul(rearrange_q, position_bias.transpose(1, 2))
        contextual_position = rearrange(contextual_position, 'q (b n) k -> b n q k', b=batch_size)

        # Relative position encoding
        scores = scores + contextual_position

        # Scale scores matrix by temperature.
        # Mask scores if needed.
        scores = scores * self.scale
        if mask is not None:
            mask_value = -torch.finfo(scores.dtype).max
            assert mask.shape[-1] == scores.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i j -> b () i j')
            scores.masked_fill_(~mask, mask_value)

        # filter Topk value if needed
        if topk != -1:
            values_topk, _ = scores.topk(min(topk, real_seq_length), dim=-1, largest=True, sorted=True)
            thres = repeat(values_topk[...,-1:], 'b h i () -> b h i j', j=real_seq_length)
            topk_mask = scores >= thres
            # self.topk_mask = topk_mask
            scores.masked_fill_(~topk_mask, -torch.finfo(scores.dtype).max)

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        outputs = attn_output

        # For DEBUG and Visualization
        # self.score = scores if self.config.debug else None
        # self.attn = attn_weights if self.config.debug else None
        # self.contextual_position = contextual_position if self.config.debug else None

        return outputs, position_bias


class AttentionBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        shape_2d,
        attention_mask=None,
        position_bias=None,
        topk=-1,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            shape_2d,
            shape_2d,
            mask=attention_mask,
            position_bias=position_bias,
            topk=topk,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, has_cross=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_cross = has_cross
        self.layer = nn.ModuleList()
        self.layer.append(AttentionBlock(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(PreNorm(config.dim, FeedForward(config.dim, config.mlp_ratio, config.dropout_rate)))

    def forward(
        self,
        hidden_states,
        shape_2d,
        attention_mask=None,
        position_bias=None,
        topk=-1,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            shape_2d,
            attention_mask=attention_mask,
            position_bias=position_bias,
            topk=topk,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # Apply Feed Forward layer.
        hidden_states = hidden_states + self.layer[-1](hidden_states)

        outputs = (hidden_states,) + attention_outputs
        return outputs  # hidden-states, (self-attention position bias)

    def compute_bias(self, shape_2d):
        return self.layer[0].SelfAttention.compute_bias(shape_2d, shape_2d)





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



class TransDecoder(nn.Module):
    debug = False
    train_scan_mode = 'default'  # default, random
    test_scan_mode = 'default'
    dim = 384
    num_layers = 6
    num_heads = 6
    dim_head = 64
    dropout = 0.
    att_scale = True
    mlp_ratio = 4
    manual_init_bias = True
    is_decoder = True
    rpe_mode = 'contextualproduct'  # 'default'
    attn_topk = -1
    def __init__(self, cin=0, cout=0):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.rpe_shared = True

        self.mask_ratio = 0
        self.dim = 384
        self.num_layers = 6
        self.num_heads = 6
        self.dim_head = 64
        self.mlp_ratio = 4
        self.dropout = 0.
        self.position_num = 7
        self.attn_topk = 32
        self.att_scale = True

        self.config = Config(
                debug=self.debug,
                dim=self.dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dim_head=self.dim_head,
                relative_attention_num_buckets=self.position_num,
                dropout_rate=self.dropout,
                scale=self.att_scale,
                mlp_ratio=self.mlp_ratio,
                mask_ratio=self.mask_ratio,
                manual_init_bias=self.manual_init_bias,
                is_decoder=self.is_decoder,
                rpe_mode=self.rpe_mode,
                attn_topk=self.attn_topk,
              )
        self.build()

    def build(self):
        # Head projection and out projection if needed
        self.to_patch_embedding = nn.Linear(self.cin, self.config.dim) if self.cin else nn.Identity()
        if self.cout:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.config.dim), nn.Linear(self.config.dim, self.cout))
            self.sos_pred_token = nn.Parameter(torch.randn(1, 1, self.cout))
        else:
            self.mlp_head = nn.Identity()
            self.sos_pred_token = nn.Parameter(torch.randn(1, 1, self.config.dim))
        
        # Transformer blocks.
        if self.rpe_shared:
            self.blocks = nn.ModuleList(
                [Block(self.config, has_relative_attention_bias=bool(i == 0)) for i in range(self.config.num_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [Block(self.config, has_relative_attention_bias=True) for i in range(self.config.num_layers)]
            )

        # Token mask
        if self.mask_ratio > 0:
            self.sampler = torch.distributions.uniform.Uniform(0., 1.)

    def forward(self, x, manual_mask=None):
        x = x.clone()
        batch_size, channels, height, width  = x.shape   # input_shape

        # Self-attention Mask & Token Mask
        if manual_mask is None:
            mask, token_mask, input_mask, output_mask = self.get_mask(batch_size, height, width)
        else:
            mask, token_mask, input_mask, output_mask = manual_mask
        mask, input_mask, output_mask = mask.to(x.device), input_mask.to(x.device), output_mask.to(x.device)
        token_mask = token_mask.to(x.device) if token_mask is not None else token_mask

        # Mask Input
        x.masked_fill_(~input_mask, 0.)

        # Patch Embedding
        x = rearrange(x, 'b c h w -> b (h w) c')
        inputs_embeds = self.to_patch_embedding(x)

        # Init state
        position_bias = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if self.training and topk != -1:
            topk = np.random.randint(topk//2, topk*2)
           
        for _, layer_module in enumerate(self.blocks):
            # Transformer block
            layer_outputs = layer_module(
                hidden_states,
                shape_2d=[height, width],
                attention_mask=mask,
                position_bias=position_bias,
                topk=topk,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
            if self.rpe_shared:
                position_bias = layer_outputs[1]

        # Out projection
        out = self.mlp_head(hidden_states)
        # Shift token if needed.  # modified
        if(hasattr(self, 'sos_pred_token')):
            sos_pred_token = repeat(self.sos_pred_token, '() n d -> b n d', b = batch_size)
            out = torch.cat((sos_pred_token, out[:,:-1,:]), dim=1)
        # Reshape Output to 2D map
        out = rearrange(out, 'b (h w) c -> b c h w', h=height)
        # Mask output
        out.masked_fill_(~output_mask, 0.)
        return out

    def get_mask(self, b, h, w):
        n = h*w
        if self.training:
            if(self.train_scan_mode == 'random' and hasattr(self, 'sampler')):
                mask_random = (self.sampler.sample([n]) > self.mask_ratio).bool()
                input_mask = mask_random.clone().view(h,w)
                mask = repeat(mask_random.unsqueeze(0), '() n -> d n', d=n) & torch.tril(torch.ones((n, n))).bool() | torch.eye(n).bool()
                output_mask = torch.cat((torch.ones(1).bool(),mask_random.clone()[:-1]), 0).view(h,w)
            else:  # (self.train_scan_mode == 'default'):
                mask = torch.tril(torch.ones((n, n))).bool()
                token_mask = None  # torch.ones_like(mask).bool()
                input_mask = torch.ones(h, w).bool()
                output_mask = torch.ones(h, w).bool()
        else:
            if self.test_scan_mode == 'default':
                mask = torch.tril(torch.ones((n, n))).bool()
                token_mask = None  # torch.ones_like(mask).bool()
                input_mask = torch.ones(h, w).bool()
                output_mask = torch.ones(h, w).bool()
            else:
                raise ValueError("No such test scan mode.")

        mask = repeat(mask.unsqueeze(0), '() d n -> b d n', b=b)
        token_mask = None  # torch.ones_like(mask).bool()
        input_mask = repeat(input_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=self.cin)
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(output_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=channel)

        return mask, token_mask, input_mask, output_mask


class TransDecoder2(TransDecoder):
    train_scan_mode = 'default'  #  'random', 'default'
    test_scan_mode = 'checkboard'
    is_decoder = False
    def __init__(self, cin=0, cout=0):
        super().__init__(cin, cout)
        del self.sos_pred_token

    def forward(self, x, manual_mask=None):
        x = x.clone()
        batch_size, channels, height, width  = x.shape   # input_shape

        # Self-attention Mask & Token Mask
        if manual_mask is None:
            mask, token_mask, input_mask, output_mask = self.get_mask(batch_size, height, width)
        else:
            mask, token_mask, input_mask, output_mask = manual_mask
        mask, input_mask, output_mask = mask.to(x.device), input_mask.to(x.device), output_mask.to(x.device)
        token_mask = token_mask.to(x.device) if token_mask is not None else token_mask

        # Mask Input
        x.masked_fill_(~input_mask, 0.)
        # Input Embedding
        x = rearrange(x, 'b c h w -> b (h w) c')
        inputs_embeds = self.to_patch_embedding(x)

        # Init state
        position_bias = None
        # encoder_decoder_position_bias = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if self.training and topk != -1:
            topk = np.random.randint(topk//2, topk*2)

        for _, layer_module in enumerate(self.blocks):
            # Transformer block
            layer_outputs = layer_module(
                hidden_states,
                shape_2d=[height, width],
                attention_mask=mask,
                position_bias=position_bias,
                topk=topk,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
            if self.rpe_shared:
                position_bias = layer_outputs[1]

        # Out projection
        out = self.mlp_head(hidden_states)        
        # Reshape Output to 2D map
        out = rearrange(out, 'b (h w) c -> b c h w', h=height)
        # Mask output
        out.masked_fill_(~output_mask, 0.)
        return out            

    def get_mask(self, b, h, w):
        n = h*w
        if self.training:
            if(self.train_scan_mode == 'random' and hasattr(self, 'sampler')):
                #mask = torch.ones(n, n).bool()    # modified
                token_mask = None
                mask_random = (self.sampler.sample([n]) > self.mask_ratio).bool()
                input_mask = mask_random.clone().view(h,w)
                output_mask = ~mask_random.clone().view(h,w)
                mask = repeat(mask_random.unsqueeze(0), '() n -> d n', d=n)
                mask = mask | torch.eye(n).bool()
            else:
                #mask = torch.ones(n, n).bool()
                token_mask = None
                mask_checkboard = torch.ones((h, w)).bool()
                mask_checkboard[0::2, 0::2] = 0
                mask_checkboard[1::2, 1::2] = 0
                input_mask = mask_checkboard.clone()
                output_mask = ~mask_checkboard.clone()
                mask = repeat(mask_checkboard.view(1,-1), '() n -> d n', d=n)
                mask = mask | torch.eye(n).bool()


        else:
            if 'checkboard' in self.test_scan_mode:
                #mask = torch.ones(n, n).bool()
                token_mask = None
                mask_checkboard = torch.ones((h, w)).bool()
                if self.test_scan_mode == 'checkboard':
                    mask_checkboard[0::2, 0::2] = 0
                    mask_checkboard[1::2, 1::2] = 0
                else:
                    mask_checkboard[0::2, 1::2] = 0
                    mask_checkboard[1::2, 0::2] = 0
                input_mask = mask_checkboard.clone()
                output_mask = ~mask_checkboard.clone()
                mask = repeat(mask_checkboard.view(1,-1), '() n -> d n', d=n)
                mask = mask | torch.eye(n).bool()
            else:
                raise ValueError("No such test scan mode.")

        #print(input_mask)
        mask = repeat(mask.unsqueeze(0), '() d n -> b d n', b=b)
        token_mask = token_mask  # torch.ones_like(mask).bool()
        input_mask = repeat(input_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=self.cin)
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(output_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=channel)
        return mask, token_mask, input_mask, output_mask


class TransHyperScale(TransDecoder):
    is_decoder = False
    def __init__(self, cin=0, cout=0, scale=1, down=True):
        self.scale = scale
        self.down = down
        super().__init__(cin, cout)

    def build(self):
        # Head projection and out projection if needed
        self.to_patch_embedding = nn.Linear(self.cin, self.config.dim) if self.cin else nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.config.dim), nn.Linear(self.config.dim, self.cout)) if self.cout else nn.Identity()
        
        # Down \ Up scale blocks. modified
        if(self.down):
            self.scale_blocks = clones(nn.Conv2d(self.config.dim, self.config.dim, 3, 2, 1, groups=1), self.scale)
        else:
            self.scale_blocks = clones(UpPixelShuffle(self.config.dim, self.config.dim, kernel_size=3, scale=2), self.scale)

        self.trans_blocks = nn.ModuleList()
        num_each_stage = self.config.num_layers // 2 // (self.scale+1)
        for _ in range(self.scale+1):
            if self.rpe_shared:
                block_scale = nn.ModuleList(
                    [Block(self.config, has_relative_attention_bias=bool(i == 0)) for i in range(num_each_stage)]
                )
            else:
                block_scale = nn.ModuleList(
                    [Block(self.config, has_relative_attention_bias=True) for i in range(num_each_stage)]
                )
            self.trans_blocks.append(block_scale)
            #  if too large for hyperprior
            next_num = self.config.relative_attention_num_buckets//2
            next_num = next_num if next_num%2 == 1 else next_num + 1
            self.config.relative_attention_num_buckets = max(next_num, 5)
            #self.config.relative_attention_num_buckets = 7  # modified, if too large for hyperprior
        
        if not self.down:
            self.trans_blocks = self.trans_blocks[::-1]

    def forward(self, x):
        batch_size, channels, height, width  = x.shape   # input_shape
        seq_length = height * width

        # Self-attention Mask & Token Mask
        mask_list, _, _, _ = self.get_mask(batch_size, height, width)
        mask_list = [mask.to(x.device) for mask in mask_list]

        # Input Embedding
        x = rearrange(x, 'b c h w -> b (h w) c')
        inputs_embeds = self.to_patch_embedding(x)

        # Init state
        # encoder_decoder_position_bias = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if topk != -1:
            if self.training:
                topk = np.random.randint(topk//2, topk*2)
            topk_list = [topk//(2**i) for i in range(self.scale+1)]  # modified
            topk_list = np.clip(topk_list, a_min=2, a_max=None)
            if not self.down:
                topk_list = topk_list[::-1]
        else:
            topk_list = [-1 for i in range(self.scale+1)]
            
        for i, scale_layer in enumerate(self.scale_blocks):
            position_bias = None
            for _, layer_module in enumerate(self.trans_blocks[i]):
                # Transformer block
                layer_outputs = layer_module(
                    hidden_states,
                    shape_2d=[height, width],
                    attention_mask=mask_list[i],
                    position_bias=position_bias,
                    topk=int(topk_list[i]),
                )

                hidden_states = layer_outputs[0]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
                if self.rpe_shared:
                    position_bias = layer_outputs[1]

            hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h=height)
            hidden_states = scale_layer(hidden_states)
            if(self.down):
                height, width = height//2, width//2
            else:
                height, width = height*2, width*2
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')

        position_bias = None
        for _, layer_module in enumerate(self.trans_blocks[-1]):
            # Transformer block
            layer_outputs = layer_module(
                hidden_states,
                shape_2d=[height, width],
                attention_mask=mask_list[-1],
                position_bias=position_bias,
                topk=int(topk_list[-1]),
            )

            hidden_states = layer_outputs[0]
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
            if self.rpe_shared:
                position_bias = layer_outputs[1]

        # Out projection
        out = self.mlp_head(hidden_states)
        out = rearrange(out, 'b (h w) c -> b c h w', h=height)
        return out

    def get_mask(self, b, h, w):
        n = h*w

        # Local Mask
        mask_list = []
        ns, hs, ws = n, h, w
        for _ in range(self.scale+1):
            mask = torch.ones((hs, ws, hs, ws)).bool()
            mask = mask.view(ns,ns)
            if self.down:
                ns, hs, ws = ns//4, hs//2, ws//2
            else:
                ns, hs, ws = ns*4, hs*2, ws*2           
            mask_list.append(mask)

        token_mask = None  # torch.ones_like(mask).bool()
        input_mask = torch.ones(h, w).bool()
        output_mask = torch.ones(h, w).bool()

        mask_list = [repeat(mask.unsqueeze(0), '() d n -> b d n', b=b) for mask in mask_list]
        token_mask = None  # torch.ones_like(mask).bool()
        input_mask = repeat(input_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=self.cin)
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(output_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=channel)

        return mask_list, token_mask, input_mask, output_mask



class Entroformer(CompressionModel):
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
        self.cit_ar = TransDecoder2(cin=384)
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

    # def forward(self, x):
    #     y = self.g_a(x)
    #     z = self.cit_he(y)
    #     z_hat, z_likelihoods = self.entropy_bottleneck(z)
    #     params = self.cit_hd(z_hat)
    #     y_hat = self.gaussian_conditional.quantize(
    #         y, "noise" if self.training else "dequantize"
    #     )
    #     #y_hat2 = quantize_ste(y)  bad
    #     ctx_params = self.cit_ar(y_hat)
    #     gaussian_params = self.cit_pn(
    #         torch.cat((params, ctx_params), dim=1)
    #     )
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
    #     y_hat2 = quantize_ste(y - means_hat) + means_hat
    #     x_hat = self.g_s(y_hat2)##
    #     return {
    #         "x_hat": x_hat,
    #         "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
    #     }
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
