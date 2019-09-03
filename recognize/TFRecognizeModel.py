#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：2019/8/26 by jade
# 邮箱：jadehh@live.com
# 描述：tensorflow 人脸特征提取
# 最近修改：2019/8/26  下午5:31 modify by jade
import cv2
import numpy as np
from sklearn import preprocessing
import tensorflow as tf


class RecModel():
    def __init__(self):
        export_dir = "model/recognition"
        self.features, self.names = self.load_facebank()
        self.threshold = 1.24
        self.sess_recognition = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess_recognition, [tf.saved_model.tag_constants.SERVING], export_dir)
        self.x = self.sess_recognition.graph.get_tensor_by_name('data:0')
        self.y = self.sess_recognition.graph.get_tensor_by_name('fc1/add_1:0')

    def load_facebank(self):
        embeddings = np.load('facebank_512.npy')
        names = np.load('names_512.npy')
        return embeddings, names

    def extract_features(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(112,112),interpolation=cv2.INTER_AREA)
        _y = self.sess_recognition.run(self.y, feed_dict={self.x: np.array([img])})
        embedding = preprocessing.normalize(_y)
        return embedding

    def compare_feature(self,img):
        feature = self.extract_features(img)
        diff = np.expand_dims(feature,2) - np.expand_dims(np.transpose(self.features,[1,0]), 0)
        dist = np.sum(np.power(diff, 2),axis=1)
        minimum = np.min(dist, axis=1)
        min_idx = np.argmin(dist,axis=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return self.names[min_idx], minimum


recModel = RecModel()