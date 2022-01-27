# coding: utf-8
# Author：WangTianRui
# Date ：2021-08-18 10:58
import math
from conv_stft import *


def complexC2F(x):
    real, imag = torch.chunk(x, 2, 1)
    result = torch.cat([real, imag], dim=2)
    if result.size(1) == 1:
        return result.squeeze(1)
    return result


def complexF2C(x):
    real, imag = torch.chunk(x, 2, 1)
    return torch.stack([real, imag], 1)


class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias=True):
        super(CausalConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.left_pad = kernel_size[1] - 1
        padding = (kernel_size[0] // 2, self.left_pad)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        return self.conv(x)[..., :T]


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding):
        super(CausalTransConvBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                                             stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        T = x.size(-1)
        conv_out = self.trans_conv(x)[..., :T]
        return conv_out


class CausalPool1d(nn.Module):
    def __init__(self, ker, str):
        super(CausalPool1d, self).__init__()
        self.smooth = nn.AvgPool1d(kernel_size=ker, stride=str, padding=0)
        self.left_pad = ker - 1

    def forward(self, x):
        x = F.pad(x, [self.left_pad, 0], value=1e-8)
        return self.smooth(x)


def mag_phase(x):
    """
    :param x: B,2*C,F,T or B,2*F,T
    :return:
    """
    real, imag = torch.chunk(x, 2, dim=1)
    mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    real_phase = real / (mags.sqrt() + 1e-8)
    imag_phase = imag / (mags.sqrt() + 1e-8)
    phase = torch.atan(
        imag_phase / (real_phase + 1e-8)
    )
    phase_adjust = (real_phase < 0).to(torch.int) * torch.sign(imag_phase) * math.pi
    phase = phase + phase_adjust
    return mags, phase


class HarmonicIntegral(nn.Module):
    def __init__(self, corr_path, loc_path, harmonic_num=1):
        super(HarmonicIntegral, self).__init__()
        self.harmonic_smooth = CausalPool1d(ker=3, str=1)
        self.harmonic_num = harmonic_num
        if corr_path is not None:
            hi_integral_matrix = torch.tensor(np.load(corr_path), dtype=torch.float).unsqueeze(0).unsqueeze(0)
            harmonic_loc = torch.tensor(np.load(loc_path), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        else:
            # for loading param
            hi_integral_matrix = torch.randn(1, 1, 4200, 257)
            harmonic_loc = torch.randn(1, 1, 4200, 257)
        self.register_buffer("integral_m", hi_integral_matrix)
        self.register_buffer("harmonic_loc", harmonic_loc)
        self.integral_m[self.integral_m != self.integral_m] = 0  # deal nan
        self.harmonic_loc[self.harmonic_loc != self.harmonic_loc] = 0

    def forward(self, x, freq_dim=257):
        """
        :param x: B,2*C,F,T
        :param freq_dim:
        :return:
        """
        integral_m = None
        harmonic_loc = None
        if freq_dim == 256:
            integral_m = self.integral_m[:, :, :, 1:]  # In line with DCRN
            harmonic_loc = self.harmonic_loc[:, :, :, 1:]
        elif freq_dim == 257:
            integral_m = self.integral_m
            harmonic_loc = self.harmonic_loc

        mag, _ = mag_phase(x)  # B,C,F,T
        mag = mag.log()
        harmonic_nominee = torch.matmul(integral_m, mag)
        value, position = torch.topk(harmonic_nominee[:, :, :], k=self.harmonic_num, dim=-2)
        choosed_harmonic = torch.zeros(mag.size(0), mag.size(1), freq_dim, mag.size(-1)).to(
            mag.device)
        for i in range(self.harmonic_num):
            choose = self.harmonic_smooth(position.to(torch.float)[:, :, i, :]).to(torch.long)
            choosed_harmonic += harmonic_loc[:, :, choose, :][0][0].permute(0, 1, 3, 2)
        choosed_harmonic = (choosed_harmonic > 0).to(torch.float)
        return choosed_harmonic


class CEM(nn.Module):
    def __init__(self, time_ker=2, freq_ker=5, kernel_num=(16, 32, 64, 128, 128, 128), rnn_hidden=128, fft_len=512):
        super(CEM, self).__init__()
        self.fft_len = fft_len
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        kernel_num = (2,) + kernel_num
        for idx in range(len(kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    CausalConv(kernel_num[idx], kernel_num[idx + 1], kernel_size=(freq_ker, time_ker), stride=(2, 1)),
                    nn.BatchNorm2d(kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )

        hidden_dim = self.fft_len // (2 ** (len(kernel_num))) * kernel_num[-1]
        self.enhance = nn.LSTM(input_size=hidden_dim, hidden_size=rnn_hidden, num_layers=1, batch_first=False)
        self.transform = nn.Linear(rnn_hidden, hidden_dim)
        for idx in range(len(kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            kernel_num[idx] * 2, kernel_num[idx - 1],
                            kernel_size=(freq_ker, time_ker), stride=(2, 1), padding=(freq_ker // 2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            kernel_num[idx] * 2, 22,
                            kernel_size=(freq_ker, time_ker), stride=(2, 1), padding=(freq_ker // 2, 0),
                            output_padding=(1, 0)
                        )
                    )
                )
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()
        self.linear_a = nn.Linear(in_features=10, out_features=2)
        self.linear_b = nn.Linear(in_features=10, out_features=2)

    def forward(self, x, noisy_mag, noisy_phase):
        out = self.compress(noisy_mag[:, 1:, :], x)
        # out = x
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_out.append(out)

        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        out = torch.reshape(out, [T, B, C * D])
        out, _ = self.enhance(out)
        out = self.transform(out)
        out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)

        mask_real = out[:, 0]
        mask_imag = out[:, 1]

        mask_real = F.pad(mask_real, [0, 0, 1, 0], value=1e-8)
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0], value=1e-8)
        mask_mag = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mag + 1e-8)
        imag_phase = mask_imag / (mask_mag + 1e-8)
        mask_phase = torch.atan(
            imag_phase / (real_phase + 1e-8)
        )
        phase_adjust = (real_phase < 0).to(torch.int) * torch.sign(imag_phase) * math.pi
        mask_phase = mask_phase + phase_adjust
        mask_mag = torch.tanh(mask_mag)
        est_mags = mask_mag * noisy_mag
        est_phase = noisy_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)
        out_spec = torch.cat([real, imag], 1)

        region_a = F.pad(out[:, 2:12], [0, 0, 1, 0], value=1e-8).permute(0, 2, 3, 1)
        region_a = self.linear_a(region_a)
        region_b = F.pad(out[:, 12:], [0, 0, 1, 0], value=1e-8).permute(0, 2, 3, 1)
        region_b = self.linear_b(region_b)

        return out_spec, region_a, region_b

    def compress(self, mag, complex):
        scaler = torch.unsqueeze(mag ** 0.23 / (mag + 1e-8), 1)  # B,F,T
        complex = complex * scaler  # B,2,F,T
        return complex


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias, out_activate):
        super(ResidualBlock, self).__init__()
        self.convblock = nn.Sequential(
            CausalConv(in_ch, out_ch, kernel_size, stride, bias),
            nn.PReLU(),
            CausalConv(out_ch, out_ch, kernel_size, stride, bias),
        )
        self.out_activate = out_activate

    def forward(self, x):
        out = self.convblock(x)
        out = self.out_activate(out + x)
        return out


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(1, 1)):
        super(GatedConv2d, self).__init__()
        gate_ch = 1
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + gate_ch),
            CausalConv(in_channels + gate_ch, in_channels + gate_ch, (1, 1), (1, 1), bias=True),
            nn.PReLU(),
            CausalConv(in_channels + gate_ch, 1, (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out_conv = CausalConv(in_ch=in_channels, out_ch=out_channels, kernel_size=kernel_size, stride=stride,
                                   bias=False)

    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
        feature = input_features * alphas
        return self.out_conv(feature)


class GHCM(nn.Module):
    def __init__(self, inch=1, chs=(8, 16, 8)):
        super(GHCM, self).__init__()
        self.chs = (inch,) + chs + (1,)
        self.activate = [nn.PReLU() for _ in range(len(self.chs) - 2)]
        self.activate.append(nn.BatchNorm2d(1))
        self.body = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        for i in range(len(self.chs) - 1):
            self.body.append(
                nn.Sequential(
                    ResidualBlock(in_ch=self.chs[i + 1], out_ch=self.chs[i + 1], kernel_size=(5, 2), stride=(1, 1),
                                  bias=False, out_activate=self.activate[i])
                )
            )
            self.gate_convs.append(
                GatedConv2d(in_channels=self.chs[i], out_channels=self.chs[i + 1], kernel_size=(5, 2))
            )

    def forward(self, gate, in_feature, origin_spec):
        """
        :param gate: B,2,F,T
        :param in_feature: B,F,T
        :param origin_spec: B,2*F,T
        :return:
        """
        inp_mag = self.mag(complexC2F(in_feature)).unsqueeze(1)
        out = inp_mag
        for index in range(len(self.body)):
            out = self.gate_convs[index](input_features=out, gating_features=gate)
            out = self.body[index](out)
        result = self.bias_apply(x_origin=origin_spec, in_feature=in_feature, mask_out=out)
        return result

    def bias_apply(self, x_origin, in_feature, mask_out):
        """
        :param mask_out: B,1,F,T
        :param in_feature: B,F,T
        :param x_origin: B,2*F,T
        :return:
        """
        mask_out = mask_out.squeeze(1)
        real, imag = torch.chunk(x_origin, 2, 1)
        real = real[:, 1:, :]
        imag = imag[:, 1:, :]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        real_phase = real / (spec_mags + 1e-8)
        imag_phase = imag / (spec_mags + 1e-8)
        s1_mag = self.mag(complexC2F(in_feature))
        bias_mag = torch.sigmoid(mask_out) * s1_mag
        est_mags = bias_mag + spec_mags
        real = F.pad(est_mags * real_phase, [0, 0, 1, 0], value=0)
        imag = F.pad(est_mags * imag_phase, [0, 0, 1, 0], value=0)
        return torch.cat([real, imag], 1)

    def mag(self, x):
        """
        :param x:B,2*F,T
        :return:
        """
        return torch.stack(torch.chunk(x, 2, dim=-2), dim=-1).pow(2).sum(dim=-1).sqrt()


class HGCN(nn.Module):
    def __init__(self, win_len=512, hop_len=128, fft_len=512, win_type='hanning',
                 harmonic_num=1, gsrm_chs=(8, 16, 8),
                 corr_path="./harmonic_integral/harmonic_integrate_matrix.npy",
                 loc_path="./harmonic_integral/harmonic_loc.npy",
                 cem_conf=None, train_flag=False):
        super(HGCN, self).__init__()
        self.train_flag = train_flag
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.win_len = win_len
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, win_type, 'complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, win_type, 'complex')
        self.hi = HarmonicIntegral(corr_path=corr_path, loc_path=loc_path, harmonic_num=harmonic_num)
        self.cem = CEM(**cem_conf, fft_len=fft_len)
        self.vad_smooth = CausalPool1d(ker=5, str=1)
        self.ghcm = GHCM(inch=1, chs=gsrm_chs)

    def forward(self, x):
        stft = self.stft(x)
        real, imag = torch.chunk(stft, 2, 1)
        noisy_mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        noisy_phase = torch.atan(imag / (real + 1e-8))
        phase_adjust = (real < 0).to(torch.int) * torch.sign(imag) * math.pi
        noisy_phase = noisy_phase + phase_adjust
        noisy_complex = torch.stack([real, imag], dim=1)  # B,2,256
        noisy_complex = noisy_complex[:, :, 1:]

        results = []
        spec1, region_a, region_b = self.cem(noisy_complex, noisy_mag=noisy_mag, noisy_phase=noisy_phase)
        results.append(spec1)
        regions = [region_a, region_b]

        with torch.no_grad():
            harmonic_loc = self.hi(complexF2C(spec1), freq_dim=257)[:, :, 1:, :]
            region_a = torch.argmax(region_a, -1).to(torch.float)
            region_b = torch.argmax(region_b, -1).to(torch.float)
            region_a = region_a[:, 1:, :].unsqueeze(1)
            vad = self.vad(region_b, threshold=24)  # changed
            voiced_region = self.vioced_region(region_b).unsqueeze(1)
            gate = region_a * vad * harmonic_loc * voiced_region
        spec1 = spec1.detach()
        # spec1 = spec1
        in_feature = complexF2C(spec1)[:, :, 1:, :]
        spec2 = self.ghcm(gate=gate, in_feature=in_feature, origin_spec=spec1)
        results.append(spec2)
        if self.train_flag:
            return results, regions
        else:
            return self.istft(results[-1]).squeeze(1)

    def vad(self, region, threshold):
        vad = (self.vad_smooth(torch.sum(region, dim=1, keepdim=True)) > threshold).to(torch.float).unsqueeze(1)
        return vad

    def vioced_region(self, region):
        if region.size(-2) % 2 != 0:
            region = F.pad(region, [0, 0, 0, 1], value=0)
        low, high = torch.chunk(region, 2, -2)  # B,F//2,T
        return (torch.sum(high, dim=-2, keepdim=True) < torch.sum(low, dim=-2, keepdim=True)).to(torch.float)


if __name__ == '__main__':
    import soundfile as sf

    dcrn_conf_ = {"time_ker": 2, "freq_ker": 5, "kernel_num": (16, 32, 64, 128, 128, 128), "rnn_hidden": 128}
    hgcn = HGCN(cem_conf=dcrn_conf_, train_flag=False)
    test_inp = torch.tensor(
        [sf.read(
            r"./wavs\fileid10_cleanBAC009S0657W0284_noiseuI44_PzWnCA_snr5_level-19.wav",
            dtype="float32")[0]]
    )
    print(test_inp.size())
    results = hgcn(test_inp)
    print(results.size())
