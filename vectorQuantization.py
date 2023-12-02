import os
import cv2 as cv
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from time import time


class VectorQuantization:
    def __init__(self, datadir, train_num=15, test_num=15):
        # self.root = root
        # self.class_list = None
        # self.img_list = None

        # self.vocab_size = 0
        # self.img_idx_train = None
        # self.img_idx_test = None

        self.vocab = None  # VISUAL WORDS. shape (vocab_size, 128)

        # self.descs_dic_train = None  # VISUAL Descriptors
        # self.descs_train = None
        # self.descs_train_label = None
        # self.descs_dic_test = None
        # self.descs_test = None
        # self.descs_test_label = None

        # self.histogram_train = None
        # self.histogram_te = None

        ############# dataset #############
        self.data_dir = None  # data directory

        self.class_list = None  # list of class names
        self.image_dataset = None  # dictionary of image names. {key: class name, value: list of image names.} e.g. {'water_lilly': ['image_0001.jpg', 'image_0002.jpg', ...], ...}
        self.train_images = None  # dictionary of train images. {key: class name, value: list of image names.} e.g. {'water_lilly': ['image_0001.jpg', 'image_0002.jpg', ...], ...}
        self.test_images = None  # dictionary of test images. {key: class name, value: list of image names.} e.g. {'water_lilly': ['image_0001.jpg', 'image_0002.jpg', ...], ...}
        self.img_idx_train = None  # dictionary of image index of train set. {key: class name, value: image index.} e.g. {'water_lilly': [0, 1, 2, ...], ...}
        self.img_idx_test = None  # dictionary of image index of test set. {key: class name, value: image index.} e.g. {'water_lilly': [0, 1, 2, ...], ...}

        self.train_images_list = None  # list of train images. e.g. ['101_ObjectCategories/water_lilly/image_0001.jpg', '101_ObjectCategories/water_lilly/image_0002.jpg', ...]
        self.test_images_list = None  # list of test images. e.g. ['101_ObjectCategories/water_lilly/image_0001.jpg', '101_ObjectCategories/water_lilly/image_0002.jpg', ...]
        self.train_labels_list = None
        self.test_labels_list = None

        self.train_num = train_num
        self.test_num = test_num
        ############# codebook #############
        self.vocab_size = 0
        self.vocab = None  # VISUAL WORDS. shape (vocab_size, 128)

        ############# SIFT #############
        self.sift = cv.SIFT_create()
        ############# histogram #############

        ####################################

        self.load_data(datadir)

    def load_data(self, data_dir):
        """
        1. Load the dataset from the given directory.
        2. Load the class names from the directory names.
        3. Load the image names from the file names.
        4. Construct image dictionary, with {key: class name, value: list of image names.} e.g. {'water_lilly': ['image_0001.jpg', 'image_0002.jpg', ...], ...}
        5. split the dataset into train and test set. (default: 15 images for train, 15 images for test). As a result,
        you will have two dictionaries for train and test set.
        """

        if not os.path.isdir(data_dir):
            print("Invalid data directory.")
            return

        print("Start loading data...")
        class_list = []
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            class_list.append(class_folder)

        print(f"Found {len(class_list)} classes: {class_list}")

        image_dataset = {}
        for class_folder in class_list:
            class_path = os.path.join(data_dir, class_folder)
            image_dataset[class_folder] = [
                image for image in os.listdir(class_path) if image.endswith(".jpg")
            ]
            print(
                f"Class: {class_folder}, # of images found: {len(image_dataset[class_folder])}"
            )

        print(
            "Split the dataset into train and test set. (default: 15 images for train, 15 images for test)."
        )
        img_idx_train = {}
        img_idx_test = {}
        for class_folder in class_list:
            permute = np.random.permutation(len(image_dataset[class_folder]))
            img_idx_train[class_folder] = permute[: self.train_num]
            img_idx_test[class_folder] = permute[
                self.train_num : self.train_num + self.test_num
            ]

        train_images = {}
        test_images = {}
        for class_folder in class_list:
            train_images[class_folder] = [
                image_dataset[class_folder][idx] for idx in img_idx_train[class_folder]
            ]
            test_images[class_folder] = [
                image_dataset[class_folder][idx] for idx in img_idx_test[class_folder]
            ]

        # Listify the dataset
        # Images are stored as list of image paths
        train_images_list = []
        test_images_list = []
        train_labels_list = []
        test_labels_list = []

        for class_folder in class_list:
            for image_name in train_images[class_folder]:
                image_path = os.path.join(data_dir, class_folder, image_name)
                train_images_list.append(image_path)
                train_labels_list.append(class_folder)

            for image_name in test_images[class_folder]:
                image_path = os.path.join(data_dir, class_folder, image_name)
                test_images_list.append(image_path)
                test_labels_list.append(class_folder)

        # Convert to numpy array
        train_images_list = np.array(train_images_list)
        test_images_list = np.array(test_images_list)
        train_labels_list = np.array(train_labels_list)
        test_labels_list = np.array(test_labels_list)

        # save as attributes
        self.train_images_list = train_images_list
        self.test_images_list = test_images_list
        self.train_labels_list = train_labels_list
        self.test_labels_list = test_labels_list

        self.data_dir = data_dir
        self.class_list = class_list
        self.image_dataset = image_dataset
        self.train_images = train_images
        self.test_images = test_images
        self.img_idx_train = img_idx_train
        self.img_idx_test = img_idx_test

    def load_codebook(self, codebook_path):
        if not os.path.exists(codebook_path):
            print("Invalid codebook path.")
            return

        self.vocab = np.load(codebook_path)

    def save_codebook(self, export_path="./codebook.npy"):
        if self.vocab is None:
            print(
                "Codebook is not constructed. Please run construct_kmeans_codebook()."
            )
            return

        np.save(export_path, self.vocab)

    def fit_codebook(self, vocab_size=200, total_descriptors=100000):
        # check if the data is loaded
        if self.class_list is None:
            print("Data is not loaded. Please run load_data() first.")
            return

        # compute SIFT descriptors. Due to the memory issues, limit the total number of descriptors to total_descriptor (100,000 default).
        print("Start computing SIFT descriptors...")

        descriptors = []
        num_desc = 0

        tqdm_bar = tqdm(
            total=len(self.train_images_list),
            desc="Computing SIFT descriptors. The number of images loaded:",
        )

        for class_folder in self.class_list:
            for image_name in self.train_images[class_folder]:
                image_path = os.path.join(self.data_dir, class_folder, image_name)
                image = cv.imread(image_path)
                if image.shape[2] == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                _, desc = self.sift.detectAndCompute(image, None)
                descriptors.append(desc)
                num_desc += desc.shape[0]
                tqdm_bar.update(1)
                if num_desc >= total_descriptors:
                    print(f"Total number of descriptors: {num_desc}")
                    break
        tqdm_bar.close()

        descriptors = np.concatenate(descriptors, axis=0)
        print("Shape of descriptors: ", descriptors.shape)

        # construct codebook
        self.vocab_size = vocab_size
        self.construct_kmeans_codebook(vocab_size, descriptors)

    def construct_train_histograms(self):
        """
        Construct histogram of train set.
        """

        if self.vocab is None:
            print(
                "Codebook is not constructed. Please run construct_kmeans_codebook()."
            )
            return

        print("Constructing histogram of train set...")
        self.histogram_train = []
        for image in self.train_images_list:
            image_path = os.path.join(self.data_dir, image)
            descriptor = self.get_descriptor(image_path)
            histogram = self.construct_histogram(descriptor)
            self.histogram_train.append(histogram)

        self.histogram_train = np.concatenate(self.histogram_train, axis=0)
        print("Constructed histogram of train set. Shape: ", self.histogram_train.shape)

    def encode_image(self, image_paths):
        """
        Encode an image to a histogram.

        Args:
            image_paths (list): List of image paths.

        Returns:
            histogram (np.ndarray): The histogram of the image. shape of (vocab_size,).

        """
        hists = []
        for image_path in image_paths:
            descriptor = self.get_descriptor(image_path)
            histogram = self.construct_histogram(descriptor)
            hists.append(histogram)

        hists = np.concatenate(hists, axis=0)
        return hists

    def get_descriptor(self, image_path):
        if os.path.exists(image_path):
            image = cv.imread(image_path)
            if image.shape[2] == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, desc = self.sift.detectAndCompute(image, None)
            return desc

    def construct_kmeans_codebook(self, vocab_size=200, decriptors=None):
        """
        Creates a codebook using KMeans clustering and the SIFT descriptors of the training set.

        Args:
            vocab_size (int): The size of the codebook.
            descriptors (np.array): The descriptors of an image. shape of (num_of_desc, 128).
        """

        print("Start constructing k-means codebook...")
        self.vocab_size = vocab_size
        kmeans = KMeans(n_clusters=self.vocab_size, random_state=0, n_init=5)
        start = time()
        kmeans.fit(decriptors)
        end = time()
        self.vocab = kmeans.cluster_centers_
        print("Shape of vocab: ", self.vocab.shape, "(vocab_size, 128)")
        print(
            "The codebook is constructed. You can use it with 'self.vocab' attribute."
        )
        print(f"Time taken constructing k-means codebook: {end-start:.2f}s")

    def assign_kmeans_codeword(self, descriptors):
        """
        Assigns the nearest codeword for each descriptor.

        Args:
            descriptors (np.array): The descriptor of an image. shape of (num_of_desc, 128). The return shape is also determined by the shape of desc.

        Returns:
            codeword (np.array): The codeword of the descriptor. shape of (num_of_desc, vocab_size)
        """

        if self.vocab is None:
            print(
                "Codebook is not constructed. Please run construct_kmeans_codebook()."
            )
            return

        if descriptors.shape[1] != 128:
            print("Invalid shape of descriptor.")
            return
        if descriptors.ndim == 2:
            distances_to_codewords = np.linalg.norm(descriptors - self.vocab, axis=1)
            codeword = np.zeros((descriptors.shape[0], self.vocab_size))
            for i in range(descriptors.shape[0]):
                codeword[i, np.argmin(distances_to_codewords[i])] = 1

        return codeword

    def construct_histogram(self, descriptors):
        """
        Constructs the histogram of an image.

        Args:
            descriptors (np.array): The descriptors of an image. shape of (num_of_desc, 128).

        Returns:
            histogram (np.array): The histogram of the image. shape of (vocab_size,).
        """

        if self.vocab is None:
            print(
                "Codebook is not constructed. Please run construct_kmeans_codebook()."
            )
            return

        codewords = self.assign_kmeans_codeword(
            descriptors
        )  # (num_of_desc, vocab_size)

        histogram = np.sum(codewords, axis=1)  # (num_images,vocab_size,)

        return histogram

    def bow_dataset(self):
        """

        Construct histogram of train and test set.
        """

        if self.vocab is None:
            print(
                "Codebook is not constructed. Please run construct_kmeans_codebook()."
            )
            return

        print("Constructing histogram of train set...")
        # Need to construct train_set as (num_images, num_of_desc, 128)

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
