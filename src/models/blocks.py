import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = padding // 2
        pad_right = padding - pad_left
        return self.conv(F.pad(x, (pad_left, pad_right), 'constant', 0))


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size, stride_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride_size
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = padding // 2
        pad_right = padding - pad_left
        return self.max_pool(F.pad(x, (pad_left, pad_right), 'constant', 0))


class MyMaxPool2dPadSame(nn.Module):
    def __init__(self, kernel_size, stride_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride_size
        self.max_pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        in_height = x.shape[-2]
        out_height = (in_height + self.stride - 1) // self.stride
        pad_h = max(0, (out_height - 1) * self.stride + self.kernel_size - in_height)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        in_width = x.shape[-1]
        out_width = (in_width + self.stride - 1) // self.stride
        pad_w = max(0, (out_width - 1) * self.stride + self.kernel_size - in_width)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return self.max_pool(F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0))


class Inception_Block_1d_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(self.num_kernels)
        ])
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = torch.stack([kernel(x) for kernel in self.kernels], dim=-1)
        return res.mean(-1)


class Inception_Block_2d_V1(nn.Module):
    def __init__(self, in_channels, out_channels, w_kernel, w_padding, num_kernels=6, init_weight=True):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(w_kernel, 2 * i + 1), padding=(w_padding, i))
            for i in range(self.num_kernels)
        ])
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = torch.stack([kernel(x) for kernel in self.kernels], dim=-1)
        return res.mean(-1)


class Inception_Block_2d_SEB(nn.Module):
    """Multi-kernel 2D convolution with channel-wise kernel attention used by WFCAT."""

    def __init__(self, in_channels, out_channels, w_kernel, w_padding,
                 num_kernels=4, reduction=16, init_weight=True):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(w_kernel, 2 * i + 1), padding=(w_padding, i))
            for i in range(self.num_kernels)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(out_channels // reduction, out_channels * num_kernels, bias=False),
            nn.Sigmoid(),
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = torch.stack([kernel(x) for kernel in self.kernels], dim=1)
        res_sum = res.sum(dim=1)
        batch, channels, _, _ = res_sum.size()
        weights = self.avg_pool(res_sum).view(batch, channels)
        weights = self.fc(weights).view(batch, self.num_kernels, channels, 1, 1)
        return (res * weights.expand_as(res)).sum(dim=1)
