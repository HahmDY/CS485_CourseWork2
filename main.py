import os
import cv2 as cv
import numpy as np
import random
from vectorQuantization import VectorQuantization


def main():
    root = "./RF_code/Caltech_101/101_ObjectCategories"

    vq = VectorQuantization(root)
    vq.fit_codebook()
    vq.save_codebook()
    vq.construct_train_histograms()

    # load codebook
    vq_new = VectorQuantization(root)
    vq_new.load_codebook()
    vq_new.construct_train_histograms()

    # # visualize
    # cls = "water_lilly"  # ['water_lilly', 'trilobite', 'wild_cat', 'wrench', 'wheelchair', 'yin_yang', 'umbrella', 'watch', 'windsor_chair', 'tick'
    # idx = 2  # 0~14
    # datahandler.visualization(True, cls, idx)  # train : True, test : False

    # # save result
    # np.save("histogram_tr.npy", histogram_tr)
    # np.save("label_tr.npy", label_tr)
    # np.save("histogram_te.npy", histogram_te)
    # np.save("label_te.npy", label_te)
    # print(histogram_tr.shape, label_tr.shape, histogram_te.shape, label_te.shape)


if __name__ == "__main__":
    main()
