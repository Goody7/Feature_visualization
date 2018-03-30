# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
from svhn import SVHN

batch_size = 128
nb_classes = 10
nb_epoch = 20

path='/home/...' # 修改路径

# 输入图像的维度，SVHN库的图片大小均为32*32
img_rows, img_cols = 32, 32
# 卷积层中使用的卷积核的个数
nb_filters = 32
# 池化层操作的范围
pool_size = (2,2)
# 卷积核的大小
kernel_size = (3,3)
# 加载SVHN数据库中的图片，并设置训练集和测试集
svhn = SVHN('/home/cv503/Desktop/conv_visualization/svhn_dataset', nb_classes, gray=False)
X_train = svhn.train_data
Y_train = svhn.train_labels
X_test = svhn.test_data
Y_test = svhn.test_labels

# 后端使用tensorflow时，即tf模式下，
# 第一个维度是样本维，表示样本的数目，
# 第二和第三个维度是高和宽，
# 最后一个维度是通道维，表示颜色通道数
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

# 将X_train, X_test的数据格式转为float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255
# 打印出相关信息
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 建立序贯模型
model = Sequential()

# 卷积层，对二维输入进行滑动窗卷积
# 当使用该层为第一层时，应提供input_shape参数，在tf模式中，通道维位于第三个位置
# border_mode：边界模式，为"valid","same"或"full"，即图像外的边缘点是补0
# 还是补成相同像素，或者是补1
model.add(Convolution2D( 32, kernel_size[0] ,kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape)) 
model.add(Activation('relu'))

# 卷积层，激活函数是ReLu
model.add(Convolution2D( 64,kernel_size[0] ,kernel_size[1],border_mode='valid'))
model.add(Activation('relu'))
# 池化层，选用Maxpooling，给定pool_size，dropout比例为0.25
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# 卷积层，激活函数是ReLu
model.add(Convolution2D( 128,kernel_size[0] ,kernel_size[1],border_mode='valid')) 
model.add(Activation('relu'))
# 池化层，选用Maxpooling，给定pool_size，dropout比例为0.25
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())

# 包含128个神经元的全连接层，激活函数为ReLu，dropout比例为0.5
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 包含10个神经元的输出层，激活函数为Sigmoid
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# 输出模型的参数信息
model.summary()
# 配置模型的学习过程
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-4),
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

# 按batch计算在某些输入数据上模型的误差
score = model.evaluate(X_test, Y_test, verbose=0)

# 存储模型   
model.save(path+'/conv_visualization/model.h5')   # HDF5 文件  
del model  # 删除已有模型  
