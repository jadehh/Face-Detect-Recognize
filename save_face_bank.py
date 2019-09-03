#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/26 by jade
# 邮箱：jadehh@live.com
# 描述：提取特征后，保存人脸特征数据
# 最近修改：2019/8/26  下午5:31 modify by jade
from mtcnn.face_detection import detect
from recognize.MxRecognizeModel import recModel
from jade import *
def extract_feature(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    bboxes, label_tests, labels, scores,face_img = detect(img)
    #CVShowBoxes(img,bboxes,label_tests,labels,scores,waitkey=-1)
    embeddings = recModel.extract_features(face_img)
    return embeddings

if __name__ == '__main__':
    root_path = "/home/jade/Data/FaceImg/users"
    names = os.listdir(root_path)
    features = []
    processBar = ProcessBar()
    processBar.count = len(names)
    for name in names:
        processBar.start_time = time.time()
        embs = []
        for path in os.listdir(os.path.join(root_path,name)):
            img = cv2.imread(os.path.join(root_path,name,path))
            emb = extract_feature(img)
            embs.append(emb)
        if len(embs) > 1:
            features.append(np.mean(np.reshape(np.array(embs),[embs[0].shape[0]]),axis=0))
        else:
            features.append(np.reshape(np.array(embs),[embs[0].shape[0]]))
        NoLinePrint("extarct features",processBar)
    np.save("names_mxnet_512.npy",np.array(names))
    np.save("facebank_mxnet_512.npy",np.array(features))



