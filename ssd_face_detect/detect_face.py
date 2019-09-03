#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/7/18 16:58 by jade
# 邮箱：jadehh@live.com
# 描述：图片目标检测预测
# 最近修改：2019/7/18 16:58 modify by jade
import argparse
from jade import *
from ssd_face_detect.objects_model import ObjectModel
import cv2
import time


class DetectFace():
    def __init__(self,args):
        self.args = args
        self.load_model()
    def load_model(self):
        self.detectModel = ObjectModel(self.args)
    def predict(self,img):
        detectionResult = self.detectModel.predict(img, 0.9)
        image = CVShowBoxes(img,detectionResult,waitkey=-1)
        cv2.imshow("image",image)
        face_boxes = []
        face_labels = []
        face_labelIDs = []
        face_scores = []
        max_area = 0.0
        faceImg = np.array([])
        for i in range(len(detectionResult.label_texts)):
            if detectionResult.label_texts[i] == "face":
                area = (float(detectionResult.boxes[i][2]) - float(detectionResult.boxes[i][0])) * (float(detectionResult.boxes[i][3] - float(detectionResult.boxes[i][1])))
                if max_area < area:
                    max_area = area
                    face_boxes.append(detectionResult.boxes[i])
                    face_labels.append("face")
                    face_labelIDs.append(1)
                    face_scores.append(detectionResult.scores[i])
                    faceImg = CutImageWithBox(img, detectionResult.boxes[i])

        faceDetectResults = DetectResultModel(face_boxes,face_labels,face_labelIDs,face_boxes)
        return faceDetectResults,faceImg


paraser = argparse.ArgumentParser(description="Detect car")
    # genearl
paraser.add_argument("--model_path",
                         default="/home/jade/Models/GestureFaceModels/ssd_mobilenet_v1_gesture_face_2019-08-21",
                         help="path to load model")
paraser.add_argument("--label_path", default="/home/jade/Data/GestureFace/gesture_face.prototxt",
                         help="path to labels")
paraser.add_argument("--num_classes", default=3, help="the number of classes")
paraser.add_argument("--gpu_memory_fraction", default=0.8, help="the memory of gpu")
args = paraser.parse_args()
detectFace = DetectFace(args)

def detect(img):
    return detectFace.predict(img)



if __name__ == '__main__':
    img = cv2.imread("/home/jade/Data/FaceImg/users/168/52463a9f-f33d-414c-a199-b5195f000eee.jpg")
    faceDetectResults,faceImg = detect(img)
    cv2.imshow("result",faceImg)
    cv2.waitKey(0)

