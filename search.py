from scipy.cluster.vq import *
from sklearn import preprocessing
from sklearn.externals import joblib
from matplotlib import pylab
from pylab import *
from PIL import Image
from calsift import RootSIFT
import argparse as ap
import cv2
import imutils
import numpy as np
import os
import numpy as np
import pickle
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--imgpath",help="ImagePath for searching",type=str,required=True)
args=parser.parse_args()
class BowSearch(object):
    def __init__(self):
        self.image_path=args.imgpath
        self.pare_paths=[]
        self.des_list=[]
        self.fea_det = cv2.xfeatures2d.SIFT_create()
        self.des_ext = cv2.xfeatures2d.SIFT_create()

    def return_images(self):
        # Load the classifier, class names, scaler, number of clusters and vocabulary
        with open("bof5000.pkl","rb") as bof:
            im_features, image_paths, idf, numWords,voc = pickle.load(bof)
        im = cv2.imread(self.image_path)
        (h, w)= im.shape[:2]
        rw = 500
        dim = (rw, int(h*(rw/w)))
        im = cv2.resize(im, dim, cv2.INTER_NEAREST)
        kpts =self.fea_det.detect(im)
        kpts, des = self.des_ext.compute(im, kpts)
        self.des_list.append((self.image_path, des))

        descriptors = self.des_list[0][1]

        test_features = np.zeros((1, numWords), "float32")
        words, distance = vq(descriptors,voc)
        for w in words:
            test_features[0][w] += 1

        # Perform Tf-Idf vectorization and L2 normalization
        test_features = test_features*idf
        test_features = preprocessing.normalize(test_features, norm='l2')

        score = np.dot(test_features, im_features.T)
        rank_ID = np.argsort(-score)
        # Visualize the results
        figure()
        gray()
        subplot(5,4,1)
        imshow(im[:,:,::-1])
        axis('off')
        for i, ID in enumerate(rank_ID[0][0:8]):
            img = Image.open(image_paths[ID])
            self.pare_paths.append(image_paths[ID])
            gray()
            subplot(5,4,i+5)
            imshow(img)
            axis('off')
            print(image_paths[ID])
        with open("info.txt","w") as f:
            for item in self.pare_paths:
                f.write(item)
                f.write("\n")
        show()
if __name__ == "__main__":
    bow=BowSearch()
    bow.return_images()
