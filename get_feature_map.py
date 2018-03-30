#coding=utf-8

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

path='/home/...' # 修改路径

# 输入图像的维度，SVHN库的图片大小均为32*32
img_rows, img_cols = 32, 32
# 卷积层中使用的卷积核的个数
nb_filters = 32
# 池化层操作的范围
pool_size = (2,2)
# 卷积核的大小
kernel_size = (3,3)
input_shape = (img_rows, img_cols, 3)

# 加载已保存的模型 
model = load_model(path+'/conv_visualization/model.h5') 
# 清除result文件夹中上一次运行保存的结果图
for i in os.listdir(path+'/conv_visualization/result'): 
	path_file = os.path.join(path+'/conv_visualization/result',i) 
	os.remove(path_file)

#在测试集中任选一张图片进行测试
test_x = []
img = Image.open(path+'/conv_visualization/test/19.png')
#img=img.convert('L') #转换为灰度图像
img=img.resize((32,32))
img.save(path+'/conv_visualization/result/test_image.jpg') #存储选中的测试图像

img_array = np.asarray(img,dtype="float32")
img_array = (255 - img_array) / 255
img_array = np.reshape(img_array, (32, 32, 3))
test_x.append(img_array)
###################################### 第一层卷积层可视化 ####################################
layer = model.layers[0]
weight = layer.get_weights()
# 输出权重值
print(np.asarray(weight).shape)
model_v1 = Sequential()
# 第一层卷积层包含32个3x3大小的卷积核
model_v1.add(Convolution2D( 32, kernel_size[0] ,kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model_v1.add(Activation('relu'))
model_v1.layers[0].set_weights(weight)

re = model_v1.predict(np.array(test_x))
print(np.shape(re))
re = np.transpose(re, (0,3,1,2))
# 32个卷积核所提取的特征以4x8方阵显示出来
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.imshow(re[0][i]) 
plt.savefig(path+'/conv_visualization/result/Conv2D_1 Feature Map.jpg') #将输出特征图谱以JPG格式存储至result文件夹
####################################### 第二层卷积层可视化 #######################################
model_v2 = Sequential()
# 第一层卷积层包含32个3x3大小的卷积核
model_v2.add(Convolution2D(32, kernel_size[0] ,kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model_v2.add(Activation('relu'))
# 第二层卷积层包含64个3x3大小的卷积核
model_v2.add(Convolution2D( 64,kernel_size[0] ,kernel_size[1]))
model_v2.add(Activation('relu'))

print(len(model_v2.layers))
layer0 = model.layers[0]
weight0 = layer0.get_weights()
model_v2.layers[0].set_weights(weight0)
layer2 = model.layers[2]
weight2 = layer2.get_weights()
model_v2.layers[2].set_weights(weight2)
re2 = model_v2.predict(np.array(test_x))
re2 = np.transpose(re2, (0,3,1,2))
# 64个卷积核所提取的特征以8x8方阵显示出来
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(re2[0][i]) 
plt.savefig(path+'/conv_visualization/result/Conv2D_2 Feature Map.jpg') #将输出特征图谱以JPG格式存储至result文件夹
#####################################  第三层卷积层可视化  #######################################
model_v3 = Sequential()
# 第一层卷积层包含32个3x3大小的卷积核
model_v3.add(Convolution2D(32, kernel_size[0] ,kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model_v3.add(Activation('relu'))
# 第二层卷积层包含64个3x3大小的卷积核
model_v3.add(Convolution2D( 64,kernel_size[0] ,kernel_size[1]))
model_v3.add(Activation('relu'))

model_v3.add(MaxPooling2D(pool_size=pool_size))
model_v3.add(Dropout(0.25))

# 第三层卷积层包含128个3x3大小的卷积核
model_v3.add(Convolution2D( 128,kernel_size[0] ,kernel_size[1]))
model_v3.add(Activation('relu'))

print(len(model_v3.layers))
layer0 = model.layers[0]
weight0 = layer0.get_weights()
model_v3.layers[0].set_weights(weight0)
layer2 = model.layers[2]
weight2 = layer2.get_weights()
model_v3.layers[2].set_weights(weight2)
layer6 = model.layers[6]
weight6 = layer6.get_weights()
model_v3.layers[6].set_weights(weight6)
re3 = model_v3.predict(np.array(test_x))
re3 = np.transpose(re3, (0,3,1,2))
# 121个卷积核所提取的特征以11x11方阵显示出来（为便于观察只显示121个卷积核结果）
for i in range(121):
    plt.subplot(11,11,i+1)
    plt.imshow(re3[0][i]) 
plt.savefig(path+'/conv_visualization/result/Conv2D_3 Feature Map.jpg') #将输出特征图谱以JPG格式存储至result文件夹
