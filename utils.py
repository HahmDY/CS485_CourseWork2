import os
import cv2 as cv
import numpy as np
import random
from sklearn.cluster import KMeans

img_sel = [15, 15] # train, test set size
n_centers = 4096 # k-means

class DataHandler():
    
    def __init__(self):
        self.root = "/Users/dongyoonhahm/KAIST/CS485/cw2/RF_code/Caltech_101/101_ObjectCategories"
    
    def load_data(self):
        class_list = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        descs_dic_tr = {}
        descs_tr = None
        descs_dic_te = {}
        descs_te = None
        
        ### apply sift
        for c in class_list:
            sub_folder_name = os.path.join(self.root, c)
            img_list = [img for img in os.listdir(sub_folder_name) if img.endswith('.jpg')]
            img_idx = random.sample(range(len(img_list)), sum(img_sel))
            img_idx_tr = img_idx[:img_sel[0]]
            img_idx_te = img_idx[img_sel[0]:sum(img_sel)]
            
            ##### train set
            for i in img_idx_tr:
                img_path = os.path.join(sub_folder_name, img_list[i])
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
            for i in img_idx_te:
                img_path = os.path.join(sub_folder_name, img_list[i])
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
        kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(descs_tr)
        centers = kmeans.cluster_centers_
        squares_centers = np.sum(centers**2, axis=1)
        
        ##### test set
        histogram_tr = np.zeros((len(class_list)*img_sel[0], n_centers))
        for c in class_list:
            for i in img_idx_tr:
                squares_desc_tr = np.sum(descs_dic_tr[c, i]**2, axis=1)
                dist = np.sqrt(squares_desc_tr[:, np.newaxis] + squares_centers[np.newaxis, :] - 2*np.dot(descs_dic_tr[c, i], centers.T)) #(len(kpt), len(centers))
                assignments = np.argmin(dist, axis=1)
                hist, _ = np.histogram(assignments, bins=np.arange(1, n_centers+2))
                histogram_tr[class_list.index(c)*len(img_idx_tr) + img_idx_tr.index(i), :] += hist
        histogram_tr = histogram_tr / np.sum(histogram_tr, axis=1)[:, np.newaxis]
        
        ##### test set
        histogram_te = np.zeros((len(class_list)*img_sel[1], n_centers))
        for c in class_list:
            for i in img_idx_tr:
                squares_desc_te = np.sum(descs_dic_te[c, i]**2, axis=1)
                dist = np.sqrt(squares_desc_te[:, np.newaxis] + squares_centers[np.newaxis, :] - 2*np.dot(descs_dic_tr[c, i], centers.T)) #(len(kpt), len(centers))
                assignments = np.argmin(dist, axis=1)
                hist, _ = np.histogram(assignments, bins=np.arange(1, n_centers+2))
                histogram_te[class_list.index(c)*len(img_idx_te) + img_idx_te.index(i), img_idx_te.index(i), :] += hist
        histogram_te = histogram_te / np.sum(histogram_te, axis=1)[:, np.newaxis]
        
        ### label
        label_tr = np.zeros(len(class_list)*img_sel[0])
        label_te = np.zeros(len(class_list)*img_sel[1])        
        for i in range(1,11):
            label_tr[(i-1) * img_sel[0]: i * img_sel[0]] = i
            label_te[(i-1) * img_sel[1]: i * img_sel[1]] = i
            
        return histogram_tr, label_tr, histogram_te, label_te