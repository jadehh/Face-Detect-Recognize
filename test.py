#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/26 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/26  下午5:31 modify by jade
from jade import *
from face_detection import detect
if __name__ == '__main__':
    img = cv2.imread("examples/face.jpg")
    bboxes, label_tests, labels, scores = detect(img)
    CVShowBoxes(img,bboxes,label_tests,labels,scores,waitkey=0)