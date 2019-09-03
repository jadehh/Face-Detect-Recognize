# Face-Detect-Recognize
人脸检测, 人脸特征提取，之后做人脸比对

### 人脸检测 
```
  python face_detection.py
```
### 人脸特征向量提取
 
```
  在人脸图上提取特征
  python test.py
  得到的是一个人脸特征向量,shape = [None,512]
```

### 制作人脸特征数据库
``` 
   python save_face_bank.py
```
npy/目录下存储的是不同人脸识别模型的人脸数据库


### 计算人脸检测+人脸识别模型的准确率
```
   python cal_accuarcy.py
```

>这里只有预测代码，训练代码见mtcnn和insightface
