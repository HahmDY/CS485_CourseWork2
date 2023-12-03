import os
import cv2 as cv
import numpy as np
import random
from vectorQuantization import VectorQuantization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    root = "./RF_code/Caltech_101/101_ObjectCategories"
    codebook_path = "./"
    vq = VectorQuantization(root)
    # vq.fit_codebook()
    # vq.save_codebook()
    # vq.construct_train_histograms()

    # load codebook
    vq_new = VectorQuantization(root)
    vq_new.load_codebook("./codebook.npy")
    vq_new.construct_train_histograms()
    vq_new.construct_test_histograms()

    # visualize
    cls = "water_lilly"  # ['water_lilly', 'trilobite', 'wild_cat', 'wrench', 'wheelchair', 'yin_yang', 'umbrella', 'watch', 'windsor_chair', 'tick'
    idx = 5  # 0~14
    vq_new.visualization("train", cls, idx)

    # train RF
    train_X = vq_new.get_test_histograms()
    train_y = vq_new.get_test_labels()

    print(train_X.shape)
    print(train_y.shape)

    print("Start Random Forest Training")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    rf.fit(train_X, train_y)

    # test RF
    test_X = vq_new.get_test_histograms()
    test_y = vq_new.get_test_labels()

    print("Test set shape")
    print(test_X.shape)
    print(test_y.shape)

    print("Start Random Forest Testing")
    pred_y = rf.predict(test_X)

    # accuracy, confusion matrix
    print("Accuracy: ", np.mean(pred_y == test_y))
    print("Confusion Matrix: ")
    cm = confusion_matrix(test_y, pred_y)

    # plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=vq_new.get_class_list(),
        yticklabels=vq_new.get_class_list(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("confusion_matrix_Q2.png")


if __name__ == "__main__":
    main()
