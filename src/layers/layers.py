
from typing import Any

import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Function
import torch.nn.functional as F
from torch import Tensor

from src.ops.parametrizers import NonNegativeParametrizer


__all__ = [
    "GDN",
    "AttentionBlock",
    "MaskedConv2d",
    "ResidualBlock",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv3x3",
    "subpel_conv3x3",
    "QReLU",
    "gaussian_blur",
    "meshgrid2d",
]



class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2**bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
            torch.exp(
                (-ctx.alpha**ctx.beta)
                * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
            )
            * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None






def gaussian_kernel1d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x


def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)
