#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/2 by jade
# 邮箱：jadehh@live.com
# 描述：RKNN人脸特征提取
# 最近修改：2019/9/2  下午5:09 modify by jade
from rknn.api import RKNN
import cv2
from sklearn import preprocessing
import numpy as np
import cv2
import argparse

paraser = argparse.ArgumentParser("RK3399Pro")
paraser.add_argument("--model_path",
                     default="model/rknn_rec_model_128/mobilefacenet_face_2019-08-27_onnx.rknn",
                     help="path to load model")
paraser.add_argument('--image_size', default='112,112', help='')
paraser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = paraser.parse_args()

class RecModel():
    def __init__(self,args):
        self.model_path = args.model_path
        self.input_size = args.image_size
        self.threshold = args.threshold
        self.rknn = RKNN()
        self.load_model()

    def load_model(self):
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print('load rknn model failed')
            exit(ret)
        print('load model success')
        ret = self.rknn.init_runtime(target="rk3399pro", device_id="TD033101190400338")
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('init runtime success')
        version = self.rknn.get_sdk_version()
        print(version)
        # Inference
        print('--> Running model')

    def extract_features(self,img):
        if img.shape[0] > 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        if img.shape[0] < 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[img])[0]
        embedding = preprocessing.normalize(outputs).flatten()
        return embedding

    def load_facebank(self):
        self.features = np.load('npy/facebank_mtcnn_rknn_128.npy')
        self.names = np.load('npy/names_mtcnn_rknn_128.npy')



    def compare_feature(self,img):
        feature = self.extract_features(img)
        diff = np.expand_dims(feature,2) - np.expand_dims(np.transpose(self.features,[1,0]), 0)
        dist = np.sum(np.power(diff, 2),axis=1)
        minimum = np.min(dist, axis=1)
        min_idx = np.argmin(dist,axis=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        if min_idx == -1:
            return (np.array([['None']]),np.array([0]))
        else:
            return self.names[min_idx], minimum

recModel = RecModel(args)

if __name__ == '__main__':

    img = cv2.imread('examples/test.jpg')
    feature = recModel.extract_features(img)
    print(feature)