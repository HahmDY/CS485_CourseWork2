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
        self.train_histograms = (
            None  # histogram of train set. shape of (num_train, vocab_size)
        )
        self.test_histograms = (
            None  # histogram of test set. shape of (num_test, vocab_size)
        )

        self.train_histograms_dict = (
            {}
        )  # Dictionary of histogram of train set. {key: class name, value: histogram of train set. shape of (num_train_per_class, vocab_size)}
        self.test_histograms_dict = (
            {}
        )  # Dictionary of histogram of test set. {key: class name, value: histogram of test set. shape of (num_test_per_class, vocab_size)}
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

    def export_train_histograms(self, export_path="./histograms/train_histograms.npy"):
        if self.train_histograms is None:
            print("Train histograms are not constructed.")
            return

        if os.path.exists(export_path):
            print("File already exists.")
            ans = input("Do you want to overwrite? (y/n)")
            if ans != "y":
                return
        if not os.path.exists(export_path):
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

        np.save(export_path, self.train_histograms)

    def export_test_histograms(self, export_path="./histograms/test_histograms.npy"):
        if self.test_histograms is None:
            print("Test histograms are not constructed.")
            return

        if os.path.exists(export_path):
            print("File already exists.")
            ans = input("Do you want to overwrite? (y/n)")
            if ans != "y":
                return
        if not os.path.exists(export_path):
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

        np.save(export_path, self.test_histograms)

    def load_train_histograms(self, export_path="./histograms/train_histograms.npy"):
        if not os.path.exists(export_path):
            print("Invalid histogram path.")
            return

        self.train_histograms = np.load(export_path)
        self.vocab_size = self.train_histograms.shape[1]

    def load_test_histograms(self, export_path="./histograms/test_histograms.npy"):
        if not os.path.exists(export_path):
            print("Invalid histogram path.")
            return

        self.test_histograms = np.load(export_path)
        self.vocab_size = self.test_histograms.shape[1]

    def load_codebook(self, codebook_path):
        if not os.path.exists(codebook_path):
            print("Invalid codebook path.")
            return

        self.vocab = np.load(codebook_path)
        self.vocab_size = self.vocab.shape[0]

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
        self.vocab_size = vocab_size

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
        self.train_histograms = []
        self.train_histograms_dict = {}

        for i, image in enumerate(self.train_images_list):
            image_path = os.path.join(image)
            descriptor = self.get_descriptor(image_path)
            histogram = self.construct_histogram(descriptor)

            cls = self.train_labels_list[i]
            if cls not in self.train_histograms_dict:
                self.train_histograms_dict[cls] = []
            self.train_histograms_dict[cls].append(histogram)
            self.train_histograms.append(histogram)

        self.train_histograms = np.stack(self.train_histograms, axis=0)
        print(
            "Constructed histogram of train set. Shape: ", self.train_histograms.shape
        )
        return self.train_histograms

    def construct_test_histograms(self):
        """
        Construct histogram of test set.
        """

        if self.vocab is None:
            print(
                "Codebook is not constructed. Please run construct_kmeans_codebook()."
            )
            return

        print("Constructing histogram of test set...")
        self.test_histograms = []
        self.histogram_test_dict = {}
        for i, image in enumerate(self.test_images_list):
            descriptor = self.get_descriptor(image)
            histogram = self.construct_histogram(descriptor)

            cls = self.test_labels_list[i]
            if cls not in self.histogram_test_dict:
                self.histogram_test_dict[cls] = []
            self.histogram_test_dict[cls].append(histogram)
            self.test_histograms.append(histogram)

        self.test_histograms = np.stack(self.test_histograms, axis=0)
        print("Constructed histogram of test set. Shape: ", self.test_histograms.shape)
        return self.test_histograms

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
        """
        Get the SIFT descriptor of an image.

        Args:
            image_path (str): The path of the image.

        Returns:
            desc (np.array): The descriptor of the image. shape of (num_of_desc, 128).
        """
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
            distances_to_codewords = np.linalg.norm(
                descriptors[:, np.newaxis, :] - self.vocab, axis=2
            )
            codeword = np.zeros((descriptors.shape[0], self.vocab_size))
            for i in range(descriptors.shape[0]):
                codeword[i, np.argmin(distances_to_codewords[i, :])] = 1

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

        histogram = np.sum(codewords, axis=0)  # (num_images,vocab_size,)

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

    def visualization(self, dataset="train", cls="water_lilly", idx=0):
        """
        Visualize the image and its histogram.

        Args:
            dataset (str, optional): the dataset to visualize. "train" or "test". Defaults to "train".
            cls (str, optional): the class name. Defaults to "water_lilly".
            idx (int, optional): the index of the image. Defaults to 0.
        """
        # Setting up the figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 5)
        )  # You can adjust the figure size

        # get index of image and histogram
        if dataset == "train":
            img_path = os.path.join(self.data_dir, cls, self.train_images[cls][idx])
            histogram = self.train_histograms_dict[cls][idx]
        elif dataset == "test":
            img_path = os.path.join(self.data_dir, cls, self.test_images[cls][idx])
            histogram = self.test_histograms_dict[cls][idx]
        else:
            print("Invalid dataset.")
            return

        # Plot the image
        title = img_path.split("/")[-1]
        image = cv.imread(img_path)
        ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        ax1.set_title(title)
        ax1.axis("off")  # To hide axis ticks and labels

        # Plot the histogram
        ax2.bar(range(self.vocab_size), histogram)
        ax2.set_title(f"Histogram of {title}")
        ax2.set_xlabel("Visual Word Index")
        ax2.set_ylabel("Frequency")

        # Display the entire figure
        plt.show()

    def get_train_histograms(self):
        return self.train_histograms

    def get_test_histograms(self):
        return self.test_histograms

    def get_train_labels(self):
        return self.train_labels_list

    def get_test_labels(self):
        return self.test_labels_list

    def get_train_images(self):
        return self.train_images_list

    def get_test_images(self):
        return self.test_images_list

    def get_train_images_dict(self):
        return self.train_images

    def get_test_images_dict(self):
        return self.test_images

    def get_train_histograms_dict(self):
        return self.train_histograms_dict

    def get_test_histograms_dict(self):
        return self.test_histograms_dict

    def get_class_list(self):
        return self.class_list

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
