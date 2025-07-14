import torch.nn as nn
from functools import reduce
from typing import Literal

import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

from model.layers import CausalConv1d, ResidualUnit, CausalConvTranspose1d

class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(EncoderBlock, self).__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=1
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=3
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=9
            ),
            CausalConv1d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=4, out_channels=C, kernel_size=7),
            EncoderBlock(out_channels=2*C, stride=strides[0]),
            EncoderBlock(out_channels=4*C, stride=strides[1]),
            EncoderBlock(out_channels=8*C, stride=strides[2]),
            EncoderBlock(out_channels=16*C, stride=strides[3]),
            CausalConv1d(in_channels=16*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(DecoderBlock, self).__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=1
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=3
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=9
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=16*C, kernel_size=7),
            DecoderBlock(out_channels=8*C, stride=strides[3]),
            DecoderBlock(out_channels=4*C, stride=strides[2]),
            DecoderBlock(out_channels=2*C, stride=strides[1]),
            DecoderBlock(out_channels=C, stride=strides[0]),
            CausalConv1d(in_channels=C, out_channels=4, kernel_size=7)
        )

    def forward(self, x):
        return self.layers(x)


class SoundRain(nn.Module):
    def __init__(self, n_q=4, codebook_size=1024, D=32, C=32, strides=(2, 4, 5, 8)):
        super(SoundRain, self).__init__()

        # The temporal resampling ratio between input waveform and embeddings.
        # Not used in here, but helpful for consumers.
        self.M = reduce(lambda a, b: a * b, strides)

        self.encoder = Encoder(C=C, D=D, strides=strides)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q,
            codebook_size=codebook_size,
            dim=D,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D, strides=strides)

    def forward(
            self,
            x,
            mode: Literal['end-to-end', 'encode', 'decode'] = 'end-to-end',
        ):
        # x: batch_size x 1 x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x 1 x (T / 1)

        if mode == 'end-to-end':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            o = self.decoder(quantized.permute((0,2,1)))
            return o
        
        if mode == 'encode':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            return quantized
        
        if mode == 'decode':
            o = self.decoder(x.permute((0,2,1)))
            return o
