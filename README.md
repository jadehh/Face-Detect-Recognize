# Face-Detect-Recognize
人脸检测, 人脸特征提取，之后做人脸比对

### 人脸检测 
```
  python face_detection.py
```
---
### 人脸对齐
关键点对齐
```
   img = alignment(image, bounding_boxes[i][0:4], landmark, (112, 112))
```
---
### 人脸特征向量提取
在人脸图上提取特征 
```
  python test.py
```
得到的是一个人脸特征向量,shape = [None,128]

---
model文件夹下面存放的是人脸检测模型+人脸识别模型

[人脸检测模型(mtcnn model)](https://pan.baidu.com/s/1OKDLM_y0afz3h2KUnihhVQ)          提取码:vnhq

[人脸识别模型 (mxnet_rec_model_128)](https://pan.baidu.com/s/1WYwyDYjtap6zdUnKI8lMqQ) 提取码: n759

[人脸识别模型 (mxnet_rec_model_512)](https://pan.baidu.com/s/1FCmvoSeXLlxcRmm7oNCMvA) 提取码: h4gc

[人脸识别模型 (tf_rec_model_512)](https://pan.baidu.com/s/1qjghULTE4-QVqgKe2fp6Ig) 提取码: vij8

---

### 制作人脸特征数据库
``` 
   python save_face_bank.py
```
npy/目录下存储的是不同人脸识别模型的人脸数据库

---

### 计算人脸检测+人脸识别模型的准确率
```
   python cal_accuarcy.py
```

---
>这里只有预测代码，训练代码见mtcnn和insightface
