import math

import torch.nn as nn

from models.blocks import Inception_Block_1d_V1, Inception_Block_2d_SEB


class WFCAT(nn.Module):
    """WFCAT model for IAT histogram features.

    Input shape for IAT features is [batch, 2 * iat_bins, seq_length]. The
    forward pass reshapes it to [batch, iat_bins, 2, seq_length], where the
    height dimension stores outgoing/incoming directions.
    """

    def __init__(self, num_classes: int = 100, iat_bins: int = 9,
                 num_kernels_1d: int = 4, num_kernels_2d: int = 4,
                 init_weights: bool = True):
        super().__init__()
        cfg = {'N': [128, 128, 'A', 256, 256, 'A', 512]}
        self.iat_bins = iat_bins
        self.first_layer_out_channel = 64
        self.first_layer = self.make_first_layers(
            in_channels=iat_bins,
            out_channel=self.first_layer_out_channel,
            num_kernels=num_kernels_2d,
        )
        self.features = self.make_layers(
            cfg['N'] + [num_classes],
            in_channels=self.first_layer_out_channel,
            num_kernels=num_kernels_1d,
        )
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), self.iat_bins, 2, -1)
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def make_layers(cfg, in_channels=64, num_kernels=4):
        layers = []
        for v in cfg:
            if v == 'A':
                layers += [nn.AvgPool1d(3), nn.Dropout(0.1)]
            else:
                conv1d = Inception_Block_1d_V1(in_channels, v, num_kernels)
                layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    def make_first_layers(in_channels=9, out_channel=64, num_kernels=4):
        layers = []
        layers += [Inception_Block_2d_SEB(in_channels, 32, 2, 0, num_kernels),
                   nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]
        layers += [nn.Conv2d(32, 32, kernel_size=(1, 6), stride=1, padding=(0, 1)),
                   nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]
        layers += [nn.AvgPool2d((1, 3)), nn.Dropout(0.1)]
        layers += [nn.Conv2d(32, out_channel, kernel_size=(1, 6), stride=1, padding=(0, 1)),
                   nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]
        layers += [nn.Conv2d(out_channel, out_channel, kernel_size=(1, 6), stride=1, padding=(0, 1)),
                   nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.GELU()]
        layers += [nn.AvgPool2d((1, 3)), nn.Dropout(0.1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
