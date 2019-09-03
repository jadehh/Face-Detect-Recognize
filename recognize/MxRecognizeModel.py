#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/2 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/9/2  下午5:09 modify by jade

import numpy as np
import cv2
from recognize.face_model import FaceModel

import argparse

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='model/mxnet_rec_model_128/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

class RecModel():
    def __init__(self,args):
        self.model = FaceModel(args)
        self.threshold = 1.24


    def extract_features(self,img):
        if img.shape[0] > 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        if img.shape[0] < 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img,(2,0,1))
        emb = self.model.get_feature(img)
        return emb

    def load_facebank(self):
        self.features = np.load('npy/facebank_mtcnn_mxnet_128.npy')
        self.names = np.load('npy/names_mtcnn_mxnet_128.npy')



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