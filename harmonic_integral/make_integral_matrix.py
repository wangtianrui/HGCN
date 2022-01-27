# coding: utf-8
# Author：WangTianRui
# Date ：2021-07-13 15:09
import sys
import torch, os, math

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import librosa as lib
import drawer
from conv_stft import *


def mag(x):
    """
    :param x:B,2*F,T
    :return:
    """
    return torch.stack(torch.chunk(x, 2, dim=-2), dim=-1).pow(2).sum(dim=-1).sqrt().log()


stft_512 = ConvSTFT(512, 128, 512, "hanning", 'complex')
istft_512 = ConviSTFT(512, 128, 512, "hanning", 'complex')


def load_test(path, loc_path):
    smooth = nn.AvgPool1d(kernel_size=1, stride=1, padding=0)
    corr_factor = torch.tensor(np.load(path), dtype=torch.float).unsqueeze(0)
    loc = torch.tensor(np.load(loc_path), dtype=torch.float).unsqueeze(0)
    corr = corr_factor
    corr[corr != corr] = 0
    drawer.plot_mesh(corr[0])
    noisy_path = r"../wavs/fileid10_cleanBAC009S0657W0284_noiseuI44_PzWnCA_snr5_level-19.wav"
    clean_path = r"../wavs/fileid10_BAC009S0657W0284.wav"
    noisy, _ = lib.load(noisy_path, sr=16000)
    clean, _ = lib.load(clean_path, sr=16000)
    noisy_stft = stft_512(torch.tensor([noisy]))
    noisy_mag = mag(noisy_stft)
    clean_stft = stft_512(torch.tensor([clean]))
    clean_mag = mag(clean_stft)
    drawer.plot_mesh(noisy_mag[0], "noisy_mag")
    drawer.plot_mesh(clean_mag[0], "clean_mag")
    harmonic_noisy = torch.matmul(corr, noisy_mag)
    # harmonic_clean = torch.matmul(corr, clean_mag)
    drawer.plot_mesh(harmonic_noisy[0].data, "harmonic_nominee_noisy")
    # drawer.plot_mesh(harmonic_clean[0].data, "harmonic_nominee_clean")
    value, position = torch.topk(harmonic_noisy, k=5, dim=1)
    choosed_harmonic = torch.zeros(1, 257, noisy_stft.size(-1))
    for i in range(1):
        choose = smooth(position.to(torch.float)[:, i, :].unsqueeze(1)).flatten().to(torch.long)
        choosed_harmonic += loc[:, choose, :].permute(0, 2, 1)
    choosed_result = (choosed_harmonic > 0).to(torch.float)
    drawer.plot_mesh(choosed_result[0].data, "harmonic position predicted by noisy")


def make_integral_matrix():
    factor = np.zeros((4200, 257))
    harmonic_loc = np.zeros((4200, 257))
    for f in range(600, 4200):
        last_loc = 0
        for k in range(1, int(80000 // f) + 1):
            compress_freq_loc = int(f * k / 80000 * 256)
            value = 1 / np.sqrt(k)
            factor[f, compress_freq_loc] += value
            harmonic_loc[f, compress_freq_loc] = 1.0
            # 谷结构建模
            if compress_freq_loc - last_loc > 1:
                if (last_loc + compress_freq_loc) % 2 != 0:
                    first_loc = int((last_loc + compress_freq_loc) // 2)
                    second_loc = first_loc + 1
                    factor[f, first_loc] += -0.5 * value
                    factor[f, second_loc] += -0.5 * value
                else:
                    loc = int((last_loc + compress_freq_loc) // 2)
                    factor[f, loc] += -1 * value
            # elif compress_freq_loc - last_loc == 1:
            else:
                factor[f, compress_freq_loc] = factor[f, compress_freq_loc] - value * 0.5
                factor[f, last_loc] = factor[f, last_loc] - value * 0.5
            last_loc = compress_freq_loc
    drawer.plot_mesh(factor)
    drawer.plot_mesh(harmonic_loc)
    np.save("harmonic_integrate_matrix", factor)
    np.save("harmonic_loc", harmonic_loc)
    return factor


if __name__ == '__main__':
    make_integral_matrix()
    load_test(r"./harmonic_integrate_matrix.npy", r"./harmonic_loc.npy")
