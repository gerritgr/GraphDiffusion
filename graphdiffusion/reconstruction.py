import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


import torch
import torch.nn as nn
import torch.nn.functional as F

from graphdiffusion.encoding import *


class VectorDenoiser(nn.Module):
    def __init__(
        self,
        node_feature_dim=1,
        hidden_dim=256,
        num_layers=8,
        dropout_rate=0.2,
        time_dim=32,
        encoding_obj=None,
    ):
        # important: config overwrites parameters
        super(VectorDenoiser, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.time_dim = time_dim

        # First fully connected layer
        self.fc1 = nn.Linear(self.node_feature_dim + self.time_dim, hidden_dim)

        # Creating hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.hidden_layers.append(layer)

        # Dropout layers
        self.dropout_layers = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_last = nn.Linear(hidden_dim, self.node_feature_dim)

        self.encoding_obj = encoding_obj or time_to_pos_emb  # SinusoidalPositionEmbeddingsMLP(dim=time_dim, use_mlp=False)

    def forward(self, data, t, condition=None, pipeline=None, *args, **kwargs):
        # Make sure data has form (batch_size, feature_dim)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert data.shape[1] == self.node_feature_dim

        # Add time to data tensor
        # Handle the case where t is a float
        if isinstance(t, float) or isinstance(t, int):
            t_tensor = torch.full((data.size(0), 1), t, dtype=data.dtype, device=data.device)
        # Handle the case where t is a tensor of size m
        elif isinstance(t, torch.Tensor) and t.ndim == 1 and t.size(0) == data.size(0):
            t_tensor = t.view(-1, 1)
        else:
            raise ValueError("t must be a float or a 1D tensor of size equal to the number of rows in x")
        # Concatenate t_tensor to x along the last dimension (columns)

        if self.time_dim > 1:
            t_tensor = self.encoding_obj(t_tensor, self.time_dim, add_original=True)
        data = torch.cat((data, t_tensor), dim=1)

        x = F.relu(self.fc1(data))  # Apply first layer with ReLU

        # Apply hidden layers with ReLU and Dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout_layers(x)

        x = self.fc_last(x)  # Apply output layer

        return x

    def save_model(self, model_path):
        """
        Save the model weights to the specified path.
        :param model_path: The file path where the model weights should be saved.
        """
        torch.save(self.state_dict(), model_path)


#   def load_model(self, pre_trained_path):
#       self.load_state_dict(torch.load(pre_trained_path, map_location=torch.device("cpu")))


################################
# Unet
################################


from inspect import isfunction
from functools import partial
from einops import rearrange
from torch import nn, einsum
import math


def correct_dims(x, channels=3, img_size=16):
    """
    Checks the shape of a given tensor and adds batch dimension if necessary.

    Args:
    x (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The output tensor with corrected shape: (batch_dim, CHANNELS, IMG_SIZE, IMG_SIZE)
    """
    if x.dim() == 2:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, channels, img_size, img_size)
    elif x.dim() == 3:
        batch_dim = 1
        x = x.reshape(batch_dim, channels, img_size, img_size)
    elif x.dim() == 4:
        return x
    else:
        assert False

    return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


# class SinusoidalPositionEmbeddings(nn.Module):
#    def __init__(self, dim, device=None):
#        super().__init__()
#        self.dim = dim
#        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")#
#
#    def forward(self, time):
#        half_dim = self.dim // 2
#        embeddings = math.log(10000) / (half_dim - 1)
#        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
#        embeddings = time[:, None] * embeddings[None, :]
#        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# TODO use pipeline encoder
class Unet(nn.Module):
    def __init__(self, dim=16, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, with_time_emb=True, resnet_block_groups=8, use_convnext=True, convnext_mult=2, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # determine dimensions
        self.channels = channels
        self.dim = dim

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            # self.time_mlp = nn.Sequential(
            #    SinusoidalPositionEmbeddings(dim, device=self.device),
            #    nn.Linear(dim, time_dim),
            #    nn.GELU(),
            #    nn.Linear(time_dim, time_dim),
            # )
            self.time_mlp = SinusoidalPositionEmbeddingsMLP(dim=time_dim)
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x, time, pipeline=None):
        img_shape = x.shape
        x = correct_dims(x, channels=self.channels, img_size=self.dim)
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, device=x.device, dtype=x.dtype).view(-1)
        img_shape_corrected = x.shape

        batch_dim = x.shape[0]
        # img_withnoise = x*1.0

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        # x = img_withnoise-x # here x is the img wo noise
        assert x.shape == img_shape_corrected
        x = x.reshape(*img_shape)
        return x


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    timestep = torch.randn(1, device=DEVICE)

    model = Unet(
        dim=16,
        channels=3,
        dim_mults=(
            1,
            2,
            4,
        ),
    ).to(DEVICE)
    img = torch.randn([1, 3, 16, 16]).to(DEVICE)
    out = model(img, timestep)
    print(out.shape)
    print(out)


class ImageReconstruction(nn.Module):
    def __init__(self, dim=16, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, with_time_emb=True, resnet_block_groups=8, use_convnext=True, convnext_mult=2, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # determine dimensions
        self.channels = channels
        self.dim = dim

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 2
            self.time_mlp = SinusoidalPositionEmbeddingsMLP(dim=time_dim)
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x, time, condition=None, pipeline=None):
        x = correct_dims(x, channels=self.channels, img_size=self.dim)
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, device=x.device, dtype=x.dtype).view(-1)
        img_shape_corrected = x.shape

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        assert x.shape == img_shape_corrected
        return x
