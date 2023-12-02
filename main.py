import os
import cv2 as cv
import numpy as np
import random
from utils import DataHandler


def main():
    root = "./RF_code/Caltech_101/101_ObjectCategories"
    img_sel = [15, 15]
    vocab_size = 64

    datahandler = DataHandler(root)
    datahandler.sift(img_sel)
    histogram_tr, label_tr, histogram_te, label_te = datahandler.kmeans_codebook(
        vocab_size
    )

    # visualize
    cls = "water_lilly"  # ['water_lilly', 'trilobite', 'wild_cat', 'wrench', 'wheelchair', 'yin_yang', 'umbrella', 'watch', 'windsor_chair', 'tick'
    idx = 2  # 0~14
    datahandler.visualization(True, cls, idx)  # train : True, test : False

    # save result
    np.save("histogram_tr.npy", histogram_tr)
    np.save("label_tr.npy", label_tr)
    np.save("histogram_te.npy", histogram_te)
    np.save("label_te.npy", label_te)
    print(histogram_tr.shape, label_tr.shape, histogram_te.shape, label_te.shape)


if __name__ == "__main__":
    main()
