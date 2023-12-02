import os
import cv2 as cv
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shutil


class DataHandler:
    def __init__(self, root):
        self.root = root
        self.class_list = None
        self.img_list = None

        self.vocab_size = 0
        self.img_idx_train = None
        self.img_idx_test = None

        self.descs_dic_train = None  # VISUAL Descriptors
        self.descs_train = None
        self.descs_train_label = None
        self.descs_dic_test = None
        self.descs_test = None
        self.descs_test_label = None

        self.histogram_train = None
        self.histogram_te = None

    def sift(self, img_sel=[15, 15]):
        ### initialize
        self.img_sel = img_sel
        class_list = [
            d
            for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ]
        self.class_list = class_list

        img_list = {}  # The total list of images
        descs_dic_train = (
            {}
        )  # The dictionary of descriptors of train set. key: (class, img_idx), value: descriptors.
        descs_train = None  # total desc of train set
        descs_train_label = []  # total label of train set
        img_idx_train = (
            {}
        )  # The dictionary of image index of train set. key: class, value: image index
        descs_dic_test = (
            {}
        )  # The dictionary of descriptors of test set. key: (class, img_idx), value: descriptors.
        descs_test = None  # total desc of test set
        descs_test_label = []  # total label of test set
        img_idx_test = (
            {}
        )  # The dictionary of image index of test set. key: class, value: image index

        ### apply sift
        print("SIFT...")
        for c in self.class_list:
            sub_folder_name = os.path.join(self.root, c)
            img_list[c] = [
                img for img in os.listdir(sub_folder_name) if img.endswith(".jpg")
            ]
            img_idx = random.sample(range(len(img_list[c])), sum(self.img_sel))
            img_idx_train[c] = img_idx[: self.img_sel[0]]
            img_idx_test[c] = img_idx[self.img_sel[0] : sum(self.img_sel)]

            ##### train set
            for i in img_idx_train[c]:
                img_path = os.path.join(sub_folder_name, img_list[c][i])
                image = cv.imread(img_path)
                if image.shape[2] == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                sift = cv.SIFT_create()
                # detectAndCompute Documentation: https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
                _, desc = sift.detectAndCompute(
                    image, None
                )  # return type: keypoints, descriptors

                descs_dic_train[
                    c, i
                ] = desc  # key: (class, img_idx), value: descriptors.
                if descs_train is not None:
                    descs_train = np.concatenate((descs_train, desc), axis=0)
                else:
                    descs_train = desc
                descs_train_label.append(c)
            ##### test set
            for i in img_idx_test[c]:
                img_path = os.path.join(sub_folder_name, img_list[c][i])
                image = cv.imread(img_path)
                if image.shape[2] == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                sift = cv.SIFT_create()
                _, desc = sift.detectAndCompute(image, None)
                descs_dic_test[c, i] = desc
                if descs_test is not None:
                    descs_test = np.concatenate((descs_test, desc), axis=0)
                else:
                    descs_test = desc
                descs_test_label.append(c)

        self.img_list = img_list
        self.img_idx_train = img_idx_train
        self.img_idx_test = img_idx_test
        self.descs_dic_train = descs_dic_train
        self.descs_train = descs_train
        self.descs_train_label = descs_train_label
        self.descs_dic_test = descs_dic_test
        self.descs_test = descs_test
        self.descs_test_label = descs_test_label

    def kmeans_codebook(self, vocab_size=64):
        """
        Creates a codebook using KMeans clustering and the SIFT descriptors of the training set.

        Args:
            vocab_size (int): The size of the codebook.

        Returns:
            histogram_train (np.ndarray): The histogram of the training set.
            label_tr (np.ndarray): The label of the training set.
            histogram_te (np.ndarray): The histogram of the test set.
            label_te (np.ndarray): The label of the test set.

        """
        ### load variables
        self.vocab_size = vocab_size
        img_list = self.img_list
        img_idx_train = self.img_idx_train
        img_idx_test = self.img_idx_test
        descs_train = self.descs_train
        descs_dic_train = self.descs_dic_train
        descs_test = self.descs_test
        descs_dic_test = self.descs_dic_test

        ### voabulary construction
        print("Clustering...")
        kmeans = KMeans(n_clusters=self.vocab_size, random_state=0, n_init=5).fit(
            descs_train
        )
        # Error here
        """
        놀랍게도 main.py로 실행하니깐 오류가 안 남...

            File ~/anaconda3/envs/ml4cv/lib/python3.11/site-packages/sklearn/base.py:1151, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
            1144     estimator._validate_params()
            1146 with config_context(
            1147     skip_parameter_validation=(
            1148         prefer_skip_nested_validation or global_skip_validation
            1149     )
            1150 ):
            -> 1151     return fit_method(estimator, *args, **kwargs)

            File ~/anaconda3/envs/ml4cv/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1526, in KMeans.fit(self, X, y, sample_weight)
            1523     print("Initialization complete")
            ...
            --> 646 config = get_config().split()
                647 if config[0] == b"OpenBLAS":
                648     return config[1].decode("utf-8")

            AttributeError: 'NoneType' object has no attribute 'split'
        """
        vocab = kmeans.cluster_centers_
        print(
            "Shape of vocab: ", vocab.shape, "(vocab_size, 128)"
        )  # 128 here is the length of the descriptor.
        squares_centers = np.sum(vocab**2, axis=1)

        ### histogram construction of train set
        print("Contructing histogram for train set...")
        histogram_train = np.zeros(
            (len(self.class_list) * self.img_sel[0], self.vocab_size)
        )  # (num_class * num_train, vocab_size)
        for c in self.class_list:
            for i in img_idx_train[c]:
                squares_desc_train = np.sum(descs_dic_train[c, i] ** 2, axis=1)
                dist = np.sqrt(
                    squares_desc_train[:, np.newaxis]
                    + squares_centers[np.newaxis, :]
                    - 2 * np.dot(descs_dic_train[c, i], vocab.T)
                )  # (len(kpt) of a image, len(centers))
                assignments = np.argmin(dist, axis=1)  # len(kpt)
                hist, _ = np.histogram(
                    assignments, bins=np.arange(0, self.vocab_size + 1)
                )  # len(vocab)
                histogram_train[
                    self.class_list.index(c) * len(img_idx_train[c])
                    + img_idx_train[c].index(i),
                    :,
                ] += hist
        histogram_train = (
            histogram_train / np.sum(histogram_train, axis=1)[:, np.newaxis]
        )  # (#data, #centers)
        print(
            "Shape of histogram_train: ",
            histogram_train.shape,
            "= (# of data, # of words)",
        )

        ### histogram construction of test set
        print("Contructing histogram for test set...")
        histogram_te = np.zeros(
            (len(self.class_list) * self.img_sel[1], self.vocab_size)
        )
        for c in self.class_list:
            for i in img_idx_test[c]:
                squares_desc_te = np.sum(descs_dic_test[c, i] ** 2, axis=1)
                dist = np.sqrt(
                    squares_desc_te[:, np.newaxis]
                    + squares_centers[np.newaxis, :]
                    - 2 * np.dot(descs_dic_test[c, i], vocab.T)
                )  # (len(kpt), len(centers))
                assignments = np.argmin(dist, axis=1)  # len(kpt)
                hist, _ = np.histogram(
                    assignments, bins=np.arange(1, self.vocab_size + 2)
                )  # len(vocab)
                histogram_te[
                    self.class_list.index(c) * len(img_idx_test[c])
                    + img_idx_test[c].index(i),
                    :,
                ] += hist
        histogram_te = (
            histogram_te / np.sum(histogram_te, axis=1)[:, np.newaxis]
        )  # (#data, #centers)
        print(
            "Shape of histogram_te: ", histogram_te.shape, "= (# of data, # of words)"
        )

        ### label
        label_tr = np.zeros(len(self.class_list) * self.img_sel[0])
        label_te = np.zeros(len(self.class_list) * self.img_sel[1])
        for i in range(1, 11):
            label_tr[(i - 1) * self.img_sel[0] : i * self.img_sel[0]] = i
            label_te[(i - 1) * self.img_sel[1] : i * self.img_sel[1]] = i

        ### save variables
        self.histogram_train = histogram_train
        self.histogram_te = histogram_te

        return histogram_train, label_tr, histogram_te, label_te

    def RF_codebook(self, vocab_size=64):
        ### load variables
        self.vocab_size = vocab_size
        img_list = self.img_list
        img_idx_train = self.img_idx_train
        img_idx_test = self.img_idx_test
        descs_train = self.descs_train
        descs_train_label = self.descs_train_label
        descs_dic_train = self.descs_dic_train
        descs_test = self.descs_test
        descs_dic_test = self.descs_dic_test
        descs_test_label = self.descs_test_label

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(descs_train, descs_train_label)

    def visualization(self, train=True, cls="water_lilly", idx=0):
        # Setting up the figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 5)
        )  # You can adjust the figure size

        # get index of image and histogram
        if train:
            img_idx = self.img_idx_train[cls][idx]
            hist_idx = self.class_list.index(cls) * self.img_sel[0] + img_idx
            histogram = self.histogram_train
        else:
            img_idx = self.img_idx_test[cls][idx]
            hist_idx = self.class_list.index(cls) * self.img_sel[1] + img_idx
            histogram = self.histogram_te

        # Plot the image
        sub_folder_name = os.path.join(self.root, cls)
        img_path = os.path.join(sub_folder_name, self.img_list[cls][img_idx])
        title = self.img_list[cls][img_idx]
        image = cv.imread(img_path)
        ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        ax1.set_title(title)
        ax1.axis("off")  # To hide axis ticks and labels

        # Plot the histogram
        ax2.bar(range(self.vocab_size), histogram[hist_idx])
        ax2.set_title(f"Histogram of {title}")
        ax2.set_xlabel("Visual Word Index")
        ax2.set_ylabel("Frequency")

        # Display the entire figure
        plt.show()

    def CNN_data_split(self):
        root_dir = self.root
        train_dir = "/Users/dongyoonhahm/KAIST/CS485/cw2/RF_code/train"
        val_dir = "/Users/dongyoonhahm/KAIST/CS485/cw2/RF_code/val"
        test_dir = "/Users/dongyoonhahm/KAIST/CS485/cw2/RF_code/test"

        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            images = [img for img in os.listdir(class_path) if img.endswith(".jpg")]
            random.shuffle(images)
            train_images = images[: self.img_sel[0]]
            val_images = images[self.img_sel[0] : self.img_sel[0] + self.img_sel[1]]
            test_images = images[
                self.img_sel[0]
                + self.img_sel[1] : self.img_sel[0]
                + self.img_sel[1]
                + self.img_sel[2]
            ]
            train_class_path = os.path.join(train_dir, class_folder)
            val_class_path = os.path.join(val_dir, class_folder)
            test_class_path = os.path.join(test_dir, class_folder)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(val_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)

            for img in train_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(train_class_path, img)
                shutil.copy(src_path, dst_path)

            for img in val_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(val_class_path, img)
                shutil.copy(src_path, dst_path)

            for img in test_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(test_class_path, img)
                shutil.copy(src_path, dst_path)
