#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/5 by jade
# 邮箱：jadehh@live.com
# 描述：npy文件转txt，可以给Android端读取
# 最近修改：2019/9/5  下午3:35 modify by jade
from jade import *
image_paths = np.load("npy/image_path_mtcnn_rknn_128.npy")
names = np.load("npy/names_mtcnn_rknn_128.npy")
features = np.load("npy/facebank_mtcnn_rknn_128.npy")
def replace_xiahuaxin(name:str):

    return  name.replace("-","_")

with open("txt/names.txt",'a') as f:
    for i in range(names.shape[0]):
        f.write(names[i]+" "+replace_xiahuaxin(image_paths[i])+"\n")

with open("txt/facebank.txt",'a') as f:
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            f.write(str(features[i,j])+" ")
        f.write("\n")

if __name__ == '__main__':
    root_path = "/home/jade/Data/FaceImg/users"
    names = os.listdir(root_path)
    features = []
    save_names = []
    processBar = ProcessBar()
    processBar.count = len(names)
    image_paths = []
    new_root_path = CreateSavePath(root_path+"_1")
    for name in names:
        CreateSavePath(os.path.join(new_root_path,name))
        processBar.start_time = time.time()
        embs = []
        for path in os.listdir(os.path.join(root_path,name)):
            shutil.copy(os.path.join(root_path,name,path),os.path.join(new_root_path,name,replace_xiahuaxin(path)))
