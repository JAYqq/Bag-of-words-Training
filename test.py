import cv2
import numpy as np
import os
import math
import pickle
import argparse
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.cluster import MiniBatchKMeans
from sklearn import datasets
from sklearn import preprocessing
from calsift import RootSIFT
from sklearn.externals import joblib

#     #coding:utf-8
# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from sklearn import datasets
# np.random.seed(5)
# iris = datasets.load_iris()
# X = iris.data
# clf = MiniBatchKMeans(n_clusters = 3)
# clf.fit(X)
# pre_clu = clf.labels_
# #将原始数据集X和聚类结果标签组合成新数据集并输出前10条
# new_X = np.column_stack((X, pre_clu))
# print new_X[:10]
#nvidia-smi
parser=argparse.ArgumentParser()
parser.add_argument("--imgpath",help="ImagePath for training",type=str,required=True)
args=parser.parse_args()
class trainBow(object):
    def __init__(self):
        self.imageLst = []
        self.des_list = []
        self.numWords = 8000
        self.count=0 
        self.dataset_path = args.imgpath
        self.fea_det = cv2.xfeatures2d.SIFT_create()
        self.des_ext = cv2.xfeatures2d.SIFT_create()
        self.pca=PCA(n_components=100)
        np.random.seed(5)
        self.clf =  MiniBatchKMeans(init='k-means++', n_clusters=5000,max_iter=20, batch_size=self.numWords,
                    n_init=10, max_no_improvement=10, verbose=0)
        self.iris=datasets.load_iris()


    def dfsImage(self):
        for root, _, files in os.walk(self.dataset_path):
            for f in files:
                if f.endswith('.jpg'):
                    self.imageLst.append(os.path.join(root, f))

    def test(self):
        # List where all the descriptors are stored
        for i, image_path in enumerate(self.imageLst):
            if i < len(self.des_list):
                f_p, _ = self.des_list[i]
                if f_p == image_path:
                    print("skip:" + image_path)
                    continue
            im = cv2.imread(image_path)
            print(image_path)
            (h, w)= im.shape[:2]
            rw = 500
            # if w > rw:
            dim = (rw, int(h*(rw/w)))
            im = cv2.resize(im, dim, cv2.INTER_NEAREST)
            print(im.shape[:2])
            try:
                print("Extract SIFT of %s image, %d of %d images" %(image_path, i+1, len(self.imageLst)))
                kpts=self.fea_det.detect(im)
                #_, des = self.fea_det.detectAndCompute(im, None)
                print(len(kpts))
                kpts,des=self.fea_det.compute(im,kpts)
                self.des_list.append((image_path, des))#(image_path,图片sift特征)
            except Exception as e:
                 print("%d image error"%i)
                 print(e)
                 return
        self.des_list
        self.descriptors = self.des_list[0][1]
        i=1
        for image_path, descriptor in self.des_list[1:]:
            try:
                print(image_path,"stack",i)
                i+=1
                self.descriptors = np.vstack((self.descriptors, descriptor))
            except Exception as e:
                print("Stack Error!")
                print(e)
                continue
        self.key_means()

    def key_means(self):
        print("Start k-means: %d words, %d key points" %(self.numWords, self.descriptors.shape[0]))
        #如果机器内存充足可以直接使用kmeans
        # self.voc, self.variance = kmeans(self.descriptors, self.numWords,iter=1)
        self.clf.fit(self.descriptors)
        self.voc=self.clf.cluster_centers_
        print(self.voc,"%%%%%%")
        self.im_features =np.zeros((len(self.imageLst), self.numWords), "float32")
        for i in range(len(self.imageLst)):
            print(i," calculate the histogram of features ")
            words, distance = vq(self.des_list[i][1], self.voc)
            for w in words:
                self.im_features[i][w] += 1

        # Perform Tf-Idf vectorization
        self.nbr_occurences = np.sum((self.im_features > 0) * 1, axis=0)
        self.idf = np.array(np.log(
            (1.0*len(self.imageLst)+1) / (1.0*self.nbr_occurences + 1)), 'float32')

        # Perform L2 normalization
        self.im_features = self.im_features*self.idf
        self.im_features = preprocessing.normalize(self.im_features, norm='l2')
        print(self.im_features)
        print(self.im_features.shape,"imgshape")
        print("begin dump")
        with open("bof5000.pkl", "wb") as bf:
            pickle.dump((self.im_features, self.imageLst, self.idf, self.numWords,self.voc), bf, -1)


if __name__ == "__main__":
    train1 = trainBow()
    train1.dfsImage()
    train1.test()
