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

# SIFT_Thresholds
# contrastThreshold = 0.02
# edgeThreshold = 10


class VectorQuantization:
    def __init__(
        self,
        datadir,
        vocab_size,
        train_num=15,
        test_num=15,
        use_RF_codebook=False,
        vq_n_estimators=10,
        rf_keyword_args={},
        contrastThreshold=0.04,
        edgeThreshold=10,
    ):
        ############# dataset #############
        self.data_dir = datadir  # data directory

        self.class_list = None  # list of class names
        self.image_dataset = None  # dictionary of image names. {key: class name, value: list of image names.} e.g. {'water_lilly': ['image_0001.jpg', 'image_0002.jpg', ...], ...}
        self.train_images_dict = None  # dictionary of train images. {key: class name, value: list of image names.} e.g. {'water_lilly': ['image_0001.jpg', 'image_0002.jpg', ...], ...}
        self.test_images_dict = None  # dictionary of test images. {key: class name, value: list of image names.} e.g. {'water_lilly': ['image_0001.jpg', 'image_0002.jpg', ...], ...}
        self.img_idx_train = None  # dictionary of image index of train set. {key: class name, value: image index.} e.g. {'water_lilly': [0, 1, 2, ...], ...}
        self.img_idx_test = None  # dictionary of image index of test set. {key: class name, value: image index.} e.g. {'water_lilly': [0, 1, 2, ...], ...}

        self.train_images_list = None  # list of train images. e.g. ['101_ObjectCategories/water_lilly/image_0001.jpg', '101_ObjectCategories/water_lilly/image_0002.jpg', ...]
        self.test_images_list = None  # list of test images. e.g. ['101_ObjectCategories/water_lilly/image_0001.jpg', '101_ObjectCategories/water_lilly/image_0002.jpg', ...]
        self.train_labels_list = None
        self.test_labels_list = None

        self.train_num = train_num
        self.test_num = test_num
        ############# codebook #############
        self.vocab_size = vocab_size
        self.vocab = None  # VISUAL WORDS. shape (vocab_size, 128)

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
        ############SIFT################
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        ############# RF codebook #############
        self.use_RF_codebook = use_RF_codebook  # whether to use RF codebook or not
        self.vq_forest = None  # Random Forest for codebook
        self.vq_n_estimators = vq_n_estimators  # The number of trees in the forest.
        self.vq_forest_max_depth = None  # The maximum depth of the tree.
        self.rf_keyword_args = rf_keyword_args  # keyword arguments for Random Forest
        ######################################
        self.load_data(datadir)

    def save_as_file(self, export_path="vectorQuantization.pkl"):
        """
        Save the class as a pickle file.

        Args:
            export_path (str, optional): The path to save the file. Defaults to "vectorQuantization.pkl".
        """
        import pickle

        if os.path.exists(export_path):
            print("File already exists.")
            ans = input("Do you want to overwrite? (y/n)")
            if ans != "y":
                return

        if not os.path.exists(export_path):
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

        with open(export_path, "wb") as f:
            pickle.dump(self, f)

        print(f"Saved as {export_path}")

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

        train_images_dict = {}
        test_images_dict = {}
        for class_folder in class_list:
            train_images_dict[class_folder] = [
                os.path.join(
                    self.data_dir, class_folder, image_dataset[class_folder][idx]
                )
                for idx in img_idx_train[class_folder]
            ]
            test_images_dict[class_folder] = [
                os.path.join(
                    self.data_dir, class_folder, image_dataset[class_folder][idx]
                )
                for idx in img_idx_test[class_folder]
            ]

        # Listify the dataset
        # Images are stored as list of image paths
        train_images_list = []
        test_images_list = []
        train_labels_list = []
        test_labels_list = []

        for class_folder in class_list:
            for image_path in train_images_dict[class_folder]:
                train_images_list.append(image_path)
                train_labels_list.append(class_folder)

            for image_path in test_images_dict[class_folder]:
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
        self.train_images_dict = train_images_dict
        self.test_images_dict = test_images_dict
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
        if os.path.exists(export_path):
            print("File already exists.")
            ans = input("Do you want to overwrite? (y/n)")
            if ans != "y":
                return
        if not os.path.exists(export_path):
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

        np.save(export_path, self.vocab)

    def fit_codebook(self, total_descriptors=100000):
        # check if the data is loaded
        if self.class_list is None:
            print("Data is not loaded. Please run load_data() first.")
            return

        # compute SIFT descriptors. Due to the memory issues, limit the total number of descriptors to total_descriptor (100,000 default).
        print("Start computing SIFT descriptors...")

        descriptors = []
        descriptor_labels = []  # needed if using RF codebook
        num_desc = 0

        tqdm_bar = tqdm(
            total=len(self.train_images_list),
            desc="Computing SIFT descriptors. The number of images loaded:",
        )

        # if self.use_RF_codebook:
        #     print("Using Random Forest codebook. Use more dense descriptors.")
        #     sift = cv.SIFT_create(
        #         contrastThreshold=0.01, edgeThreshold=5
        #     )  # default: contrastThreshold=0.04, edgeThreshold=10
        # else:
        sift = cv.SIFT_create(
            contrastThreshold=self.contrastThreshold, edgeThreshold=self.edgeThreshold
        )
        for class_folder in self.class_list:
            for image_path in self.train_images_dict[class_folder]:
                image = cv.imread(image_path)
                if image.shape[2] == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                _, desc = sift.detectAndCompute(image, None)
                if desc is None:
                    # print(" No descriptor is found.")
                    # Actually this should not happen.
                    return False
                descriptors.append(desc)
                if self.use_RF_codebook:
                    descriptor_labels.append(
                        np.full(desc.shape[0], self.class_list.index(class_folder))
                    )

                num_desc += desc.shape[0]
                tqdm_bar.update(1)
                if num_desc >= total_descriptors:
                    print(f"Total number of descriptors: {num_desc}")
                    break
        tqdm_bar.close()
        if descriptors == []:
            print("No descriptors are found.")
            return False  # fail
        descriptors = np.concatenate(descriptors, axis=0)
        print("Shape of descriptors: ", descriptors.shape)
        if self.use_RF_codebook:
            descriptor_labels = np.concatenate(descriptor_labels, axis=0)
            print("Shape of descriptor_labels: ", descriptor_labels.shape)

        # construct codebook

        if self.use_RF_codebook:
            print("Constructing codebook using Random Forest...")
            self.construct_RF_codebook(descriptors, descriptor_labels)
        else:
            self.construct_kmeans_codebook(descriptors)

        return True  # success

    def construct_train_histograms(self):
        """
        Construct histogram of train set.
        """
        if self.use_RF_codebook:
            if self.vq_forest is None:
                print(
                    "Codebook is not constructed. Please run construct_RF_codebook()."
                )
                return
            else:
                self.construct_train_histograms_RF()

        else:
            if self.vocab is None:
                print(
                    "Codebook is not constructed. Please run construct_kmeans_codebook()."
                )
                return
            else:
                self.construct_train_histograms_kmeans()

    def construct_train_histograms_kmeans(self):
        print("Constructing histogram of train set...")
        self.train_histograms = []
        self.train_histograms_dict = {}

        for i, image in enumerate(self.train_images_list):
            image_path = os.path.join(image)
            descriptor = self.get_descriptor(image_path)
            histogram = self.construct_histogram_kmeans(descriptor)

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

    def construct_train_histograms_RF(self):
        print("Constructing histogram of train set...")
        self.train_histograms = []
        self.train_histograms_dict = {}

        image_ind__wo_descriptor = []
        for i, image in enumerate(self.train_images_list):
            image_path = os.path.join(image)
            descriptor = self.get_descriptor(image_path)
            if descriptor is None:
                print("No descriptor is found.")
                # print("descriptor shape: ", descriptor.shape)
                image_ind__wo_descriptor.append(image_path)

                continue
            histogram = self.construct_histogram_RF(descriptor)

            cls = self.train_labels_list[i]
            if cls not in self.train_histograms_dict:
                self.train_histograms_dict[cls] = []
            self.train_histograms_dict[cls].append(histogram)
            self.train_histograms.append(histogram)

        self.train_histograms = np.stack(self.train_histograms, axis=0)
        print(
            "Constructed histogram of train set. Shape: ", self.train_histograms.shape
        )
        for i in image_ind__wo_descriptor:
            self.train_images_list = np.delete(
                self.train_images_list, np.where(self.train_images_list == i)
            )
            self.train_labels_list = np.delete(
                self.train_labels_list, np.where(self.train_labels_list == i)
            )
        return self.train_histograms

    def construct_histogram_RF(self, descriptor):
        """
        Constructs the histogram of an image using the Random Forest codebook.

        Args:
            descriptors (np.array): The descriptors of an image. shape of (num_of_desc, 128).

        Returns:
            histogram (np.array): The histogram of the image. shape of (vocab_size,).
        """
        if self.vq_forest is None:
            print("Codebook is not constructed. Please run construct_RF_codebook().")
            return

        codewords = self.vq_forest.apply(descriptor)  # (num_of_desc, vocab_size)
        # one hot encoding for each tree
        codewords_one_hot = np.zeros(
            (codewords.shape[0], self.vq_n_estimators * (2**self.vq_forest_max_depth))
        )

        for i in range(descriptor.shape[0]):
            starting_of_j_th_tree = 0
            for j in range(self.vq_n_estimators):
                codewords_one_hot[
                    i,
                    starting_of_j_th_tree
                    + codewords[i, j]
                    - (2**self.vq_forest_max_depth - 1),
                ] = 1
                starting_of_j_th_tree += 2**self.vq_forest_max_depth

        histogram = np.sum(codewords_one_hot, axis=0)  # (,vocab_size,)

        return histogram

    def construct_test_histograms(self):
        """
        Construct histogram of test set.
        """

        if self.use_RF_codebook:
            if self.vq_forest is None:
                print(
                    "Codebook is not constructed. Please run construct_RF_codebook()."
                )
                return
            else:
                self.construct_test_histograms_RF()

        else:
            if self.vocab is None:
                print(
                    "Codebook is not constructed. Please run construct_kmeans_codebook()."
                )
                return
            else:
                self.construct_test_histograms_kmeans()

    def construct_test_histograms_kmeans(self):
        """
        Constructs the histogram of an image using the k-means codebook.

        Returns:
            histogram (np.array): The histogram of the image. shape of (vocab_size,).
        """
        print("Constructing histogram of test set...")
        self.test_histograms = []
        self.test_histograms_dict = {}

        for i, image in enumerate(self.test_images_list):
            image_path = os.path.join(image)
            descriptor = self.get_descriptor(image_path)
            histogram = self.construct_histogram_kmeans(descriptor)

            cls = self.test_labels_list[i]
            if cls not in self.test_histograms_dict:
                self.test_histograms_dict[cls] = []
            self.test_histograms_dict[cls].append(histogram)
            self.test_histograms.append(histogram)

        self.test_histograms = np.stack(self.test_histograms, axis=0)
        print("Constructed histogram of test set. Shape: ", self.test_histograms.shape)
        return self.test_histograms

    def construct_test_histograms_RF(self):
        """
        Constructs the histogram of an image using the Random Forest codebook.


        """
        print("Constructing histogram of test set...")
        self.test_histograms = []
        self.test_histograms_dict = {}

        image_ind__wo_descriptor = []
        for i, image in enumerate(self.test_images_list):
            image_path = os.path.join(image)
            descriptor = self.get_descriptor(image_path)
            if descriptor is None:
                print("No descriptor is found.")
                # print("descriptor shape: ", descriptor.shape)
                image_ind__wo_descriptor.append(image_path)
                continue
            histogram = self.construct_histogram_RF(descriptor)

            cls = self.test_labels_list[i]
            if cls not in self.test_histograms_dict:
                self.test_histograms_dict[cls] = []
            self.test_histograms_dict[cls].append(histogram)
            self.test_histograms.append(histogram)

        for i in image_ind__wo_descriptor:
            self.test_images_list = np.delete(
                self.test_images_list, np.where(self.test_images_list == i)
            )
            self.test_labels_list = np.delete(
                self.test_labels_list, np.where(self.test_labels_list == i)
            )

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
        sift = cv.SIFT_create(
            contrastThreshold=self.contrastThreshold, edgeThreshold=self.edgeThreshold
        )
        for image_path in image_paths:
            descriptor = self.get_descriptor(image_path, sift=sift)
            histogram = self.construct_histogram_kmeans(descriptor)
            hists.append(histogram)

        hists = np.concatenate(hists, axis=0)
        return hists

    def get_descriptor(self, image_path, sift=None):
        """
        Get the SIFT descriptor of an image.

        Args:
            image_path (str): The path of the image.

        Returns:
            desc (np.array): The descriptor of the image. shape of (num_of_desc, 128).
        """
        if sift is None:
            sift = cv.SIFT_create(
                contrastThreshold=self.contrastThreshold,
                edgeThreshold=self.edgeThreshold,
            )

        if os.path.exists(image_path):
            image = cv.imread(image_path)
            if image.shape[2] == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, desc = sift.detectAndCompute(image, None)
            return desc

    def construct_RF_codebook(self, descriptors=None, descriptors_label=None):
        """
        Creates a codebook using Random Forest and the SIFT descriptors of the training set.

        Args:
            vocab_size (int): The size of the codebook.
            descriptors (np.array): The descriptors of an image. shape of (num_of_desc, 128).
            descriptors_label (np.array): The labels of the descriptors. shape of (num_of_desc,).


        """

        print("Start constructing Random Forest codebook...")

        self.vq_forest_max_depth = int(np.log2(self.vocab_size / self.vq_n_estimators))
        print(
            f"Max depth of each tree: {self.vq_forest_max_depth}, # of trees: {self.vq_n_estimators}, total number of leaves: {2**self.vq_forest_max_depth * self.vq_n_estimators}"
        )

        start = time()

        self.vq_forest = RandomForestClassifier(
            n_estimators=self.vq_n_estimators,
            max_depth=self.vq_forest_max_depth,
            random_state=0,
            **self.rf_keyword_args,
        )
        self.vq_forest.fit(descriptors, descriptors_label)
        end = time()

        self.vq_forest_num_leaves = []
        for i in range(self.vq_n_estimators):
            leaves_i = self.vq_forest.estimators_[i].get_n_leaves()
            self.vq_forest_num_leaves.append(leaves_i)

        print(f"Time taken constructing Random Forest codebook: {end-start:.2f}s")
        print(f"Total number of codewords: {sum(self.vq_forest_num_leaves)}")
        print("The RF codebook is constructed. You can now encode image with it.")

    def construct_kmeans_codebook(self, decriptors=None):
        """
        Creates a codebook using KMeans clustering and the SIFT descriptors of the training set.

        Args:
            vocab_size (int): The size of the codebook.
            descriptors (np.array): The descriptors of an image. shape of (num_of_desc, 128).
        """

        print("Start constructing k-means codebook...")

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

    def construct_histogram_kmeans(self, descriptors):
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
            img_path = self.train_images_dict[cls][idx]

            histogram = self.train_histograms_dict[cls][idx]
        elif dataset == "test":
            img_path = self.test_images_dict[cls][idx]
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
        return self.train_images_dict

    def get_test_images_dict(self):
        return self.test_images_dict

    def get_train_histograms_dict(self):
        return self.train_histograms_dict

    def get_test_histograms_dict(self):
        return self.test_histograms_dict

    def get_class_list(self):
        return self.class_list


def load_from_file(vq_path="vectorQuantization.pkl"):
    """
    Load the class from a pickle file.

    Args:
        vq_path (str, optional): The path to load the file. Defaults to "vectorQuantization.pkl".

    Returns:
        vq (VectorQuantization): The class.
    """
    import pickle

    if not os.path.exists(vq_path):
        print("Invalid path.")
        return

    with open(vq_path, "rb") as f:
        vq = pickle.load(f)

    print(f"Loaded from {vq_path}")
    return vq
