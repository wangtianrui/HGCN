# coding: utf-8
# Author：WangTianRui
# Date ：2020/10/23 19:31
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_mesh(img, title="", save_home=""):
    img = img
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        return
    plt.show()


def plot_spec_mesh(img, title="", save_home=""):
    img = np.log(abs(img))
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
    plt.show()


def plot_scatter(array, title="1"):
    xs = np.arange(len(array))
    plt.scatter(xs, array)
    plt.title(title)
    plt.show()


def plot(array, title):
    plt.plot(array)
    plt.title(title)
    plt.show()
