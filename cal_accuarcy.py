#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/2 by jade
# 邮箱：jadehh@live.com
# 描述：计算人脸检测+人脸识别的准确率
# 最近修改：2019/9/2  下午6:04 modify by jade

from recognize.MxRecognizeModel import recModel
from mtcnn.face_detection import detect
from jade import *
def predict_names(img,x_name):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    bboxes, label_tests, labels, scores,face_img = detect(img)
    if face_img.shape[0] > 0:
        y_name = recModel.compare_feature(face_img)
    else:
        y_name = (np.array([['None']]),np.array([0]))
    if x_name == y_name[0][0]:
        return True,x_name,y_name[0][0],y_name[1][0]
    elif y_name[0][0] == "None":
        return False,x_name,"没有检测到人脸",0
    else:
        return False,x_name,y_name[0][0],y_name[1][0]
if __name__ == '__main__':
    root_path = "/home/jade/Data/FaceImg/images"
    names = os.listdir(root_path)
    features = []
    processBar = ProcessBar()
    processBar.count = len(names)
    truth = 0
    num = 0

    for name in names:
        if name != "0":
            processBar.start_time = time.time()
            for path in os.listdir(os.path.join(root_path,name)):

                img = cv2.imread(os.path.join(root_path,name,path))
                issuccess,x_name,y_name,score =  predict_names(img,name)
                if issuccess:
                    truth = truth + 1
                # else:
                #     text = "truth label = {}, predict label = {} , score = {}".format(x_name,y_name,score)
                #     index = 0
                #     for text_split in text.split(","):
                #         cv2.putText(img,text_split,(10,10+20*index),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
                #         index = index + 1
                #     cv2.imshow("wrong",img)
                #     cv2.waitKey(0)
                #     print(os.path.join(root_path, name, path))
                #     print("预测错误")
                num = num + 1
            NoLinePrint("预测人脸名称",processBar)
    print("acc = {}".format(truth/float(num)))



