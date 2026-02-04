import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Interp23Tap(nn.Module):
    def __init__(self, ratio=4):
        super(Interp23Tap, self).__init__()
        assert (2 ** (round(np.log2(ratio)))) == ratio, 'Error: Only resize factors power of 2'
        self.ratio = ratio

        # 定义CDF23系数
        CDF23 = np.asarray([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942,
                            0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
        CDF23 = [element * 2 for element in CDF23]
        BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)
        BaseCoeff = np.expand_dims(BaseCoeff, axis=(0, 1))

        # 转换为torch tensor
        self.BaseCoeff = torch.from_numpy(BaseCoeff.astype(np.float32))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, channels, height, width = x.shape
        img = x

        # 将BaseCoeff扩展到匹配的通道数并移动到相同设备
        BaseCoeff_expanded = self.BaseCoeff.repeat(channels, 1, 1, 1).to(x.device)

        for z in range(int(np.log2(self.ratio))):
            # 创建上采样图像
            new_height = (2 ** (z + 1)) * height
            new_width = (2 ** (z + 1)) * width

            I1LRU = torch.zeros(batch_size, channels, new_height, new_width,
                                dtype=img.dtype, device=img.device)

            if z == 0:
                I1LRU[:, :, 1::2, 1::2] = img
            else:
                I1LRU[:, :, ::2, ::2] = img

            conv = nn.Conv2d(in_channels=channels, out_channels=channels, padding=(11, 0),
                             kernel_size=BaseCoeff_expanded.shape, groups=channels, bias=False, padding_mode='circular')

            conv.weight.data = BaseCoeff_expanded
            conv.weight.requires_grad = False

            t = conv(torch.transpose(I1LRU, 2, 3))
            img = conv(torch.transpose(t, 2, 3))

        if squeeze_output:
            img = img.squeeze(0)

        return img

class TorchInterp23(nn.Module):
    def __init__(self, start_point=1):
        super(TorchInterp23, self).__init__()
        self.start_point = start_point

        # 预计算卷积核
        basecoeff = np.array([[-4.63495665e-03, -3.63442646e-03, 3.84904063e-18,
                               5.76678319e-03, 1.08358664e-02, 1.01980790e-02,
                               -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
                               -2.84531643e-02, 1.85181518e-17, 4.42450253e-02,
                               7.71733386e-02, 6.70554910e-02, -2.85299239e-17,
                               -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
                               3.61741232e-17, 2.87940558e-01, 6.25431459e-01,
                               8.97067600e-01, 1.00107877e+00, 8.97067600e-01,
                               6.25431459e-01, 2.87940558e-01, 3.61741232e-17,
                               -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
                               -2.85299239e-17, 6.70554910e-02, 7.71733386e-02,
                               4.42450253e-02, 1.85181518e-17, -2.84531643e-02,
                               -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
                               1.01980790e-02, 1.08358664e-02, 5.76678319e-03,
                               3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])

        coeff = np.dot(basecoeff.T, basecoeff)
        coeff = torch.FloatTensor(coeff)

        coeff = coeff.unsqueeze(0).unsqueeze(0)  # (1, 1, 45, 45)

        self.register_buffer('coeff', coeff)

    def unpool(self, x):
        batch_size, channels, height, width = x.shape

        new_height = height * 4
        new_width = width * 4
        out = torch.zeros(batch_size, channels, new_height, new_width,
                          device=x.device, dtype=x.dtype)

        if self.start_point == 1:
            out[:, :, 1::4, 1::4] = x
        elif self.start_point == 0:
            out[:, :, 0::4, 0::4] = x

        return out

    def tf_inter23(self, x):
        groups = x.shape[1]

        kernel = self.coeff.repeat(groups, 1, 1, 1)

        padding = self.coeff.shape[-1] // 2
        out = F.conv2d(x, kernel, padding=padding, groups=groups)

        return out

    def forward(self, x):

        temp = self.unpool(x)
        temp = self.tf_inter23(temp)

        return temp

class HPFilter(nn.Module):
    def __init__(self):
        super(HPFilter, self).__init__()
        kernel = torch.ones(1, 1, 5, 5) / 25.0
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        # Apply average pooling as low-pass filter
        low_pass = F.conv2d(x, self.kernel.repeat(x.size(1), 1, 1, 1),
                            padding=2, groups=x.size(1))
        # High-pass = original - low-pass
        high_pass = x - low_pass
        return high_pass

class Resize(nn.Module):
    def __init__(self, target_size):
        super(Resize, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        return F.interpolate(x, size=self.target_size, mode='bicubic', align_corners=True)


class Duplicate(nn.Module):
    def __init__(self, target_channels):
        super(Duplicate, self).__init__()
        self.target_channels = target_channels

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        return x.repeat(1, self.target_channels, 1, 1)


class ConvBlock1(nn.Module):
    def __init__(self, in_channels=32, nf=32, block_name='1'):
        super(ConvBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return out + residual

# Main networks
class Pannet(nn.Module):
    def __init__(self, msi_channels=4, pan_channels=1):
        super(Pannet, self).__init__()

        self.hp_filter = HPFilter()
        self.interp23 = TorchInterp23()

        # Mixed input processing
        self.mixed_conv = nn.Conv2d(msi_channels + pan_channels, 32, 3, padding=1)
        self.relu = nn.ReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ConvBlock1(32, 32, '1'),
            ConvBlock1(32, 32, '2'),
            ConvBlock1(32, 32, '3'),
            ConvBlock1(32, 32, '4')
        )

        # Final convolution
        self.final_conv = nn.Conv2d(32, msi_channels, 3, padding=1)

    def forward(self, msi, pan):
        # msi: (batch, channels, 16, 16)
        # pan: (batch, channels, 64, 64)

        h_msi = self.hp_filter(msi)
        h_pan = self.hp_filter(pan)

        re_h_msi = self.interp23(h_msi)
        re_msi = self.interp23(msi)

        mixed = torch.cat([re_h_msi, h_pan], dim=1)
        mixed1 = self.relu(self.mixed_conv(mixed))

        x = self.res_blocks(mixed1)
        x = self.final_conv(x)

        last = x + re_msi
        return last


class FusNet(nn.Module):
    def __init__(self, msi_channels=4, pan_channels=1):
        super(FusNet, self).__init__()

        self.interp23 = TorchInterp23()
        self.duplicate = Duplicate(msi_channels)

        self.sub_conv = nn.Conv2d(msi_channels, 32, 3, padding=1)

        self.res_blocks = nn.Sequential(
            ConvBlock1(32, 32, '1'),
            ConvBlock1(32, 32, '2'),
            ConvBlock1(32, 32, '3'),
            ConvBlock1(32, 32, '4')
        )

        self.final_conv = nn.Conv2d(32, msi_channels, 3, padding=1)

    def forward(self, msi, pan):
        msi_inputs1 = self.interp23(msi)
        pan_inputs1 = self.duplicate(pan)

        sub = pan_inputs1 - msi_inputs1
        sub1 = self.sub_conv(sub)

        c1 = self.res_blocks(sub1)
        c4 = self.final_conv(c1)

        c6 = msi_inputs1 + c4
        return c6


class PNNNet(nn.Module):
    def __init__(self, msi_channels=4, pan_channels=1):
        super(PNNNet, self).__init__()

        self.interp23 = TorchInterp23()

        self.conv1 = nn.Conv2d(msi_channels + pan_channels, 64, 9, padding=4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, msi_channels, 5, padding=2)
        self.relu3 = nn.ReLU()

    def forward(self, msi, pan):
        msi_inputs1 = self.interp23(msi)
        mixed = torch.cat([msi_inputs1, pan], dim=1)

        mixed1 = self.relu1(self.conv1(mixed))
        mixed1 = self.relu2(self.conv2(mixed1))
        output = self.relu3(self.conv3(mixed1))

        return output


# Usage example:
if __name__ == "__main__":
    # Test the networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pannet
    pannet = Pannet(msi_channels=4, pan_channels=1).to(device)
    msi = torch.randn(2, 4, 16, 16).to(device)
    pan = torch.randn(2, 1, 64, 64).to(device)
    output = pannet(msi, pan)
    print(f"Pannet output shape: {output.shape}")

    # FusNet
    fusnet = FusNet(msi_channels=4, pan_channels=1).to(device)
    msi = torch.randn(2, 4, 16, 16).to(device)
    pan = torch.randn(2, 1, 64, 64).to(device)
    output = fusnet(msi, pan)
    print(f"FusNet output shape: {output.shape}")

    # PNNNet
    pnnnet = PNNNet(msi_channels=4, pan_channels=1).to(device)
    msi = torch.randn(2, 4, 16, 16).to(device)
    pan = torch.randn(2, 1, 64, 64).to(device)
    output = pnnnet(msi, pan)
    print(f"PNNNet output shape: {output.shape}")