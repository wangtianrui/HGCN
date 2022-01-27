# coding: utf-8
# Author：WangTianRui
# Date ：2021/7/19 20:23
import librosa as lib
import torch, os, math, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from tqdm import tqdm
import drawer
from conv_stft import *


def load_wav_mag(wav_path, stft):
    clean, _ = lib.load(wav_path, sr=16000)
    temp = stft(torch.tensor([clean]))
    real, imag = torch.chunk(temp, 2, dim=1)
    mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    return mag[0].numpy().astype(np.float16)  # F,T


def get_label():
    stft_512 = ConvSTFT(512, 128, 512, "hanning", 'complex')
    all_log_mag_means = []
    root = r"E:\DNS-Challenge\make_data\clean"
    if not os.path.exists("mu.npy"):
        for i in os.walk(root):
            for wav_path in tqdm(i[2][:2500]):
                wav_path = os.path.join(root, wav_path)
                mag = load_wav_mag(wav_path, stft_512)
                mag = np.log(mag)
                mag[np.isinf(mag)] = 0
                all_log_mag_means.append(np.mean(mag, axis=-1))
        all_log_mag_means = np.array(all_log_mag_means) # B,F
        np.save(r"512+128", all_log_mag_means)
        mu = np.mean(all_log_mag_means, axis=0)
        np.save(r"mu", mu)
        sigma = np.var(all_log_mag_means, axis=0)
        np.save(r"sigma", sigma)


def for_test():
    stft_512 = ConvSTFT(512, 128, 512, "hanning", 'complex')
    clean_log_mag = np.log(load_wav_mag("../wavs/fileid10_BAC009S0657W0284.wav", stft_512))

    mu = np.expand_dims(np.load("mu.npy"), -1)
    sigma = np.expand_dims(np.load("sigma.npy"), -1)

    label_for_a = (clean_log_mag > mu).astype(np.float16)
    label_for_b = (clean_log_mag > (mu + (4 / 3) * sigma)).astype(np.float16)
    drawer.plot_mesh(label_for_a, "label_for_a")
    drawer.plot_mesh(label_for_b, "label_for_b")


if __name__ == '__main__':
    # get_label()
    for_test()
