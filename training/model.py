import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(c_in, c_out, k, p, d):
    return nn.Conv2d(
            c_in, c_out, kernel_size=k,
            stride=1, padding=p, dilation=d)


def conv3x3(c_in, c_out, d=1, p_hack=2):
    return conv(c_in, c_out, 3, p_hack, d)


def conv1x1(c_in, c_out, d=1):
    return conv(c_in, c_out, 1, 0, d)


def conv5x5(c_in, c_out, d=1):
    return conv(c_in, c_out, 5, 4, d)


class Squeeze(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], x.shape[1])


def ConvBlock(c_in, c_out):
    return nn.Sequential(
            nn.BatchNorm2d(c_in),
            conv3x3(c_in, c_out, p_hack=1), nn.LeakyReLU(0.2, True),
            conv3x3(c_out, c_out, p_hack=1), nn.LeakyReLU(0.2, True),
            conv1x1(c_out, c_out))


def DownConv(c_in, c_out):
    return nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(c_in, c_out))


class Upsample(nn.Module):
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


class UpBlock(nn.Module):
    def __init__(self, c_in):
        super().__init__()

        self.resize = nn.Sequential(
                conv3x3(c_in, c_in // 2, p_hack=1),
                Upsample())
                # nn.UpsamplingBilinear2d(scale_factor=2))

        self.layer = ConvBlock(c_in, c_in // 2)

    def forward(self, x, skip):
        return self.layer(torch.cat([self.resize(x), skip], 1))


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, n=3, k=32):
        super().__init__()

        pow_2 = [2 ** i for i in range(n)]

        self.up = torch.nn.ModuleList([UpBlock(k * (2 ** (n+1)))] + [UpBlock(2 * k * mult) for mult in reversed(pow_2)])
        self.last = conv1x1(k, out_channels)

    def forward(self, down, mid):
        up = [mid]

        for layer, skip in zip(self.up, reversed(down)):
            up.append(layer(up[-1], skip))

        return self.last(up[-1])


class Printer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


def get_num_channels_vizdoom(desired):
    channels = {
            'depth': 1,
            'label': 6,
            'flow': 2
            'normal': 3
            }
    result = 0

    for x in desired:
        result += channels[x]

    return result


class UnetVizDoom(nn.Module):
    def __init__(self, in_channels, desired, n=2, k=32):
        """
        desired is a dict
        {
        'flow': 2,
        'depth': 1,
        ...
        }
        """
        super().__init__()

        pow_2 = [2 ** i for i in range(n)]
        self.desired = [x for x in ['depth', 'label', 'flow', 'normal'] if x in desired]

        self.down = torch.nn.ModuleList(
                [conv3x3(in_channels, k, p_hack=1)] +
                [DownConv(k * mult, 2 * k * mult) for mult in pow_2])
        self.mid = DownConv(k * (2 ** n), k * (2 ** (n+1)))
        self.decoders = torch.nn.ModuleDict(
                {key: Decoder(k * (2 ** (n+1)), val, n) for key, val in desired.items()})

    def forward(self, x, preprocess, cat_image):
        if preprocess:
            x = x / 255.0 - 0.5

        down = [self.down[0](torch.nn.functional.pad(x, (2, 2, 2, 2)))]

        for layer in self.down[1:]:
            down.append(layer(down[-1]))

        mid = self.mid(down[-1])
        tmp = [self.decoders[x](down, mid) for x in self.desired]
        tmp = torch.cat(tmp, 1)
        tmp = tmp[:,:,2:-2,2:-2]

        if cat_image:
            tmp = torch.cat([x, tmp], 1)

        return tmp


class Unet(nn.Module):
    def __init__(self, in_channels, desired, n=3, k=32):
        """
        desired is a dict
        {
        'flow': 2,
        'depth': 1,
        ...
        }
        """
        super().__init__()

        pow_2 = [2 ** i for i in range(n)]

        self.down = torch.nn.ModuleList(
                [conv3x3(in_channels, k, p_hack=1)] +
                [DownConv(k * mult, 2 * k * mult) for mult in pow_2])
        self.mid = DownConv(k * (2 ** n), k * (2 ** (n+1)))
        self.decoders = torch.nn.ModuleDict(
                {key: Decoder(k * (2 ** (n+1)), val) for key, val in desired.items()})

    def forward(self, x, out=None, args=None, single_desired=None):
        down = [self.down[0](x)]

        for layer in self.down[1:]:
            down.append(layer(down[-1]))

        mid = self.mid(down[-1])

        if out is not None:
            out[:,       : 4] = x[:,:4]

            if single_desired is not None:
                out[:,4        :           ] = self.decoders[single_desired](down, mid)
            elif args is None or len(args.desired) > 1:
                out[:,4        : 4+5       ] = self.decoders['depth'](down, mid)
                out[:,4+5      : 4+5+10    ] = self.decoders['label'](down, mid)
                out[:,4+5+10   : 4+5+6+10  ] = self.decoders['flow'](down, mid)
                out[:,4+5+10+6 : 4+5+6+10+4] = self.decoders['material'](down, mid)
            else:
                out[:,4        :           ] = self.decoders[args.desired[0]](down, mid)

            return

        return {key: network(down, mid) for key, network in self.decoders.items()}


def NetworkInNetwork(in_channels, n_actions):
    return nn.Sequential(
            conv5x5(in_channels, 192), nn.LeakyReLU(0.2, True),
            conv1x1(192, 160), nn.LeakyReLU(0.2, True),
            conv1x1(160, 96), nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(3, 2), nn.Dropout(0.5),

            nn.BatchNorm2d(96),
            conv5x5(96, 192), nn.LeakyReLU(0.2, True),
            conv1x1(192, 192), nn.LeakyReLU(0.2, True),
            conv1x1(192, 192), nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(3, 2), nn.Dropout(0.5),

            nn.BatchNorm2d(192),
            conv3x3(192, 192), nn.LeakyReLU(0.2, True),
            conv1x1(192, 192), nn.LeakyReLU(0.2, True),
            conv1x1(192, n_actions), nn.AdaptiveAvgPool2d(1), Squeeze())
