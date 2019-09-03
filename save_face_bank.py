#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/26 by jade
# 邮箱：jadehh@live.com
# 描述：提取特征后，保存人脸特征数据
# 最近修改：2019/8/26  下午5:31 modify by jade
from mtcnn.face_detection import detect
from recognize.RKNNRecognizeModel import recModel
from jade import *
def extract_feature(img):
    detect_results,face_img = detect(img)
    #CVShowBoxes(img,detect_results,waitkey=0)
    if face_img.shape[0] > 0:
        embeddings = recModel.extract_features(face_img)
        return embeddings
    else:
        return np.array([])

if __name__ == '__main__':
    root_path = "/home/jade/Data/FaceImg/users"
    names = os.listdir(root_path)
    features = []
    processBar = ProcessBar()
    save_names = []
    processBar.count = len(names)
    for name in names:
        processBar.start_time = time.time()
        embs = []
        for path in os.listdir(os.path.join(root_path,name)):
            img = cv2.imread(os.path.join(root_path,name,path))
            emb = extract_feature(img)
            if emb.shape[0] > 0:
                embs.append(emb)
        if len(embs) > 1:
            features.append(np.mean(np.reshape(np.array(embs),[embs[0].shape[0]]),axis=0))
            save_names.append(name)
        elif len(embs) == 1:
            features.append(np.reshape(np.array(embs),[embs[0].shape[0]]))
            save_names.append(name)

        NoLinePrint("extarct features",processBar)
    np.save("npy/names_mtcnn_rknn_do_quantization_128.npy",np.array(save_names))
    np.save("npy/facebank_mtcnn_rknn_do_quantization_128.npy",np.array(features))



