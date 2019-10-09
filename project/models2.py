import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function


class SignFunction(Function):
    """
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """

    def __init__(self):
        super(SignFunction, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        # Apply quantization noise while only training
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Sign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SignFunction.apply(x, self.training)


class ConvLSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 hidden_kernel_size,
                 space_dim,
                 batch_size,
                 bias=True):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.hidden_kernel_size = hidden_kernel_size
        self.space_dim = space_dim
        self.batch_size = batch_size
        self.bias = bias
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate_channels = 4 * self.hidden_channels
        self.hidden = None

        self.convW = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            bias=self.bias,
            padding=self.padding
            )

        self.convU = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.gate_channels,
            kernel_size=self.hidden_kernel_size,
            stride=1,
            dilation=1,
            bias=self.bias,
            padding=(self.hidden_kernel_size // 2)
            )


    def reset_params(self):
        self.convW.reset_parameters()
        self.convU.reset_parameters()

    def init_hidden(self, args):
        self.hidden = (
            torch.zeros(self.batch_size, self.hidden_channels, self.space_dim, self.space_dim).to(args.device),
            torch.zeros(self.batch_size, self.hidden_channels, self.space_dim, self.space_dim).to(args.device)
        )

    def forward(self, x):
        hx, cx = self.hidden
        #print("x", x.shape)
        #print("hx", hx.shape)
        #print("convW(x)", self.convW(x).shape)
        #print("convU(hx)", self.convU(hx).shape)
        gates = self.convW(x) + self.convU(hx)
        fgate, igate, ogate, jgate = gates.chunk(4, 1)
        fgate = self.sigmoid(fgate)
        igate = self.sigmoid(igate)
        ogate = self.sigmoid(ogate)
        jgate = self.tanh(jgate)

        # print("fgate:", fgate.shape, " cx:", cx.shape)
        # print()
        ct = fgate * cx + igate * jgate
        ht = ogate * self.tanh(ct)
        self.hidden = (ht, ct)
        return ht, ct


class Encoder(nn.Module):
    def __init__(self, space_dim, batch_size):
        super(Encoder, self).__init__()
        self.space_dim = space_dim
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
            )
        self.ConvLSTM1 = ConvLSTM(
            in_channels=64,
            hidden_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            hidden_kernel_size=1,
            space_dim=int(space_dim/4),
            batch_size=batch_size,
            bias=False
            )
        self.ConvLSTM2 = ConvLSTM(
            in_channels=256,
            hidden_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            hidden_kernel_size=1,
            space_dim=int(space_dim/8),
            batch_size=batch_size,
            bias=False
            )
        self.ConvLSTM3 = ConvLSTM(
            in_channels=512,
            hidden_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            hidden_kernel_size=1,
            space_dim=int(space_dim/16),
            batch_size=batch_size,
            bias=False
            )

    def init_hidden(self, args):
        self.ConvLSTM1.init_hidden(args)
        self.ConvLSTM2.init_hidden(args)
        self.ConvLSTM3.init_hidden(args)

    def forward(self, x):
        out = self.conv(x)
        hidden1 = self.ConvLSTM1(out)
        hidden2 = self.ConvLSTM2(hidden1[0])
        hidden3 = self.ConvLSTM3(hidden2[0])
        return hidden3[0]


class Binarizer(nn.Module):
    def __init__(self, stochastic=False):
        super(Binarizer, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels=512,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
            )
        self.tanh = nn.Tanh()
        self.stochastic = stochastic
        if self.stochastic:
            self.sign = Sign()

    def forward(self, x):
        out = self.Conv(x)
        out = self.tanh(out)
        if self.stochastic:
            out = self.sign(out)
        else:
            out = out.sign()
        return out


class Decoder(nn.Module):
    def __init__(self, space_dim, batch_size):
        super(Decoder, self).__init__()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
            )
        self.ConvLSTM1 = ConvLSTM(
            in_channels=512,
            hidden_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            hidden_kernel_size=1,
            space_dim=int(space_dim/16),
            batch_size=batch_size,
            bias=False
            )
        self.ConvLSTM2 = ConvLSTM(
            in_channels=128,
            hidden_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            hidden_kernel_size=1,
            space_dim=int(space_dim/8),
            batch_size=batch_size,
            bias=False
            )
        self.ConvLSTM3 = ConvLSTM(
            in_channels=128,
            hidden_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            hidden_kernel_size=3,
            space_dim=int(space_dim/4),
            batch_size=batch_size,
            bias=False
            )
        self.ConvLSTM4 = ConvLSTM(
            in_channels=64,
            hidden_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            hidden_kernel_size=3,
            space_dim=int(space_dim/2),
            batch_size=batch_size,
            bias=False
            )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
            )

    def init_hidden(self, args):
        self.ConvLSTM1.init_hidden(args)
        self.ConvLSTM2.init_hidden(args)
        self.ConvLSTM3.init_hidden(args)
        self.ConvLSTM4.init_hidden(args)

    def forward(self, x):
        out = self.conv1(x)
        hidden1 = self.ConvLSTM1(out)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        hidden2 = self.ConvLSTM2(x)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        hidden3 = self.ConvLSTM3(x)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        hidden4 = self.ConvLSTM4(x)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        out = self.tanh(self.conv2(x))
        out = (out+1)/2
        #out = self.sigmoid(self.conv2(x))
        return out