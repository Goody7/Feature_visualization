# Feature_visualization

## Requirements

- Python 2.7
- TensorFlow 1.4.1
- Numpy
- keras 

## Introduction

该代码主要利用Keras框架实现特征可视化。
该代码利用SVHN数据库训练了一个含3个卷积层、2个池化层、1个Flatten层、2个全连接层的神经网络。并利用Adam函数进行模型优化。
该神经网络的整体架构为：输入图片→卷积层→卷积层→池化层→卷积层→池化层→Flatten层→全连接层→输出层。
然后利用SVHN数据库的图片对搭建好的神经网络进行训练。SVHN数据库包含train文件夹，test文件夹以及extra文件夹，分别包含33402、13068、202353个标记图片。用SVHN库中的训练集和测试集训练和评估模型（模型识别准确率达0.8763），并保存训练好的模型。

## Usage

result文件夹:用来存储特征可视化结果
test文件夹: 从SVHN数据库extra集任选的几个测试图像（png格式）
build_model.py: 训练模型并保存
svhn.py: 加载训练模型的输入数据并进行预处理
get_feature_map.py: 获得三个卷积层输出端的特征图谱
model.h5: 已经训练好的模型

只需运行 get_feature_map.py 文件就可以在result文件夹中看到保存的结果图，运行之前记得将每个文件的路径修改为自己的路径

## Result

利用SVHN数据库extra文件夹中的任一张图片进行特征可视化。
输入模型的测试图片：
![image](https://github.com/Goody7/Feature_visualization/raw/master/result/test_image.jpg)
