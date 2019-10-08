import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvLSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 hidden_kernel_size,
                 bias=True):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.hidden_kernel_size = hidden_kernel_size
        self.bias = bias
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate_channels = 4 * self.hidden_channels

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

    def forward(self, x, hidden):
        hx, cx = hidden
        # print("x", x.shape)
        # print("hx", hx.shape)
        # print("convW(x)", self.convW(x).shape)
        # print("convU(hx)", self.convU(hx).shape)
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
        return ht, ct


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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
            bias=False
            )

    def forward(self, x, hidden1, hidden2, hidden3):
        out = self.conv(x)
        hidden1 = self.ConvLSTM1(out, hidden1)
        hidden2 = self.ConvLSTM2(hidden1[0], hidden2)
        hidden3 = self.ConvLSTM3(hidden2[0], hidden3)
        return hidden3[0], hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self, in_channels=512, out_channels=32):
        super(Binarizer, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
            )
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.Conv(x)
        out = self.tanh(out)
        return out.sign()


class Decoder(nn.Module):
    def __init__(self):
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

    def forward(self, x, hidden1, hidden2, hidden3, hidden4):
        out = self.conv1(x)
        hidden1 = self.ConvLSTM1(out, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        hidden2 = self.ConvLSTM2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        hidden3 = self.ConvLSTM3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        hidden4 = self.ConvLSTM4(x, hidden4)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        #out = self.tanh(self.conv2(x))
        #out = (out+1)/2
        out = self.sigmoid(self.conv2(x))
        return out, hidden1, hidden2, hidden3, hidden4
