import os
import cv2 as cv
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class DataHandler():
    
    def __init__(self, img_sel, vocab_size):
        self.root = "/Users/dongyoonhahm/KAIST/CS485/cw2/RF_code/Caltech_101/101_ObjectCategories"
        self.img_sel = img_sel
        self.class_list = None
        self.img_list = None
        
        self.n_centers = vocab_size
        self.img_idx_tr = None
        self.histogram_tr = None
        self.img_idx_te = None
        self.histogram_te = None
    
    def load_data(self):
        class_list = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        self.class_list = class_list
        img_list = {}
        
        descs_dic_tr = {}
        descs_tr = None # total desc of train set
        img_idx_tr = {}
        
        descs_dic_te = {}
        descs_te = None # total desc of test set
        img_idx_te = {}
        
        ### apply sift
        print("SIFT...")
        for c in self.class_list:
            sub_folder_name = os.path.join(self.root, c)
            img_list[c] = [img for img in os.listdir(sub_folder_name) if img.endswith('.jpg')]
            img_idx = random.sample(range(len(img_list[c])), sum(self.img_sel))
            img_idx_tr[c] = img_idx[:self.img_sel[0]]
            img_idx_te[c] = img_idx[self.img_sel[0]:sum(self.img_sel)]
            
            ##### train set
            for i in img_idx_tr[c]:
                img_path = os.path.join(sub_folder_name, img_list[c][i])
                image = cv.imread(img_path)
                if image.shape[2] == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                sift = cv.SIFT_create()
                _, desc = sift.detectAndCompute(image, None)
                descs_dic_tr[c, i] = desc
                if descs_tr is not None:
                    descs_tr = np.concatenate((descs_tr, desc), axis=0)
                else:
                    descs_tr = desc
            ##### test set
            for i in img_idx_te[c]:
                img_path = os.path.join(sub_folder_name, img_list[c][i])
                image = cv.imread(img_path)
                if image.shape[2] == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                sift = cv.SIFT_create()
                _, desc = sift.detectAndCompute(image, None)
                descs_dic_te[c, i] = desc
                if descs_te is not None:
                    descs_te = np.concatenate((descs_te, desc), axis=0)
                else:
                    descs_te = desc
        
        ### voabulary construction
        print("Clustering...")  
        kmeans = KMeans(n_clusters=self.n_centers, random_state=0, n_init=5).fit(descs_tr)
        vocab = kmeans.cluster_centers_
        print("Shape of vocab: ", vocab.shape)
        squares_centers = np.sum(vocab**2, axis=1)
        
        print("Contructing histogram for train set...")  
        ##### train set
        histogram_tr = np.zeros((len(self.class_list)*self.img_sel[0], self.n_centers))
        for c in self.class_list:
            for i in img_idx_tr[c]:
                squares_desc_tr = np.sum(descs_dic_tr[c, i]**2, axis=1)
                dist = np.sqrt(squares_desc_tr[:, np.newaxis] + squares_centers[np.newaxis, :] - 2*np.dot(descs_dic_tr[c, i], vocab.T)) #(len(kpt) of a image, len(centers))
                assignments = np.argmin(dist, axis=1) # len(kpt)
                hist, _ = np.histogram(assignments, bins=np.arange(0, self.n_centers+1)) # len(vocab)
                histogram_tr[self.class_list.index(c)*len(img_idx_tr[c]) + img_idx_tr[c].index(i), :] += hist
        histogram_tr = histogram_tr / np.sum(histogram_tr, axis=1)[:, np.newaxis] # (#data, #centers)
        print("Shape of histogram_tr: ", histogram_tr.shape, "= (# of data, # of words)")
        
        ##### test set
        print("Contructing histogram for test set...")  
        histogram_te = np.zeros((len(self.class_list)*self.img_sel[1], self.n_centers))
        for c in self.class_list:
            for i in img_idx_te[c]:
                squares_desc_te = np.sum(descs_dic_te[c, i]**2, axis=1)
                dist = np.sqrt(squares_desc_te[:, np.newaxis] + squares_centers[np.newaxis, :] - 2*np.dot(descs_dic_te[c, i], vocab.T)) #(len(kpt), len(centers))
                assignments = np.argmin(dist, axis=1) # len(kpt)
                hist, _ = np.histogram(assignments, bins=np.arange(1, self.n_centers+2)) # len(vocab)
                histogram_te[self.class_list.index(c)*len(img_idx_te[c]) + img_idx_te[c].index(i), :] += hist
        histogram_te = histogram_te / np.sum(histogram_te, axis=1)[:, np.newaxis] # (#data, #centers)
        print("Shape of histogram_te: ", histogram_te.shape, "= (# of data, # of words)")
        
        ### label
        label_tr = np.zeros(len(self.class_list)*self.img_sel[0])
        label_te = np.zeros(len(self.class_list)*self.img_sel[1])        
        for i in range(1,11):
            label_tr[(i-1) * self.img_sel[0]: i * self.img_sel[0]] = i
            label_te[(i-1) * self.img_sel[1]: i * self.img_sel[1]] = i
            
        ### save variables
        self.img_list = img_list
        self.img_idx_tr = img_idx_tr
        self.histogram_tr = histogram_tr
        self.img_idx_te = img_idx_te
        self.histogram_te = histogram_te
            
        return histogram_tr, label_tr, histogram_te, label_te
    
    def visualization(self, train=True, cls='water_lilly', idx=0):
        # Setting up the figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # You can adjust the figure size

        # get index of image and histogram
        if train:
            img_idx = self.img_idx_tr[cls][idx]
            hist_idx = self.class_list.index(cls) * self.img_sel[0] + img_idx
            histogram = self.histogram_tr
        else:
            img_idx = self.img_idx_te[cls][idx]
            hist_idx = self.class_list.index(cls) * self.img_sel[1] + img_idx
            histogram = self.histogram_te
            
        # Plot the image
        sub_folder_name = os.path.join(self.root, cls)
        img_path = os.path.join(sub_folder_name, self.img_list[cls][img_idx])
        title = self.img_list[cls][img_idx]
        image = cv.imread(img_path)
        ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        ax1.set_title(title)
        ax1.axis('off')  # To hide axis ticks and labels

        # Plot the histogram 
        ax2.bar(range(self.n_centers), histogram[hist_idx])
        ax2.set_title(f"Histogram of {title}")
        ax2.set_xlabel('Visual Word Index')
        ax2.set_ylabel('Frequency')

        # Display the entire figure
        plt.show()