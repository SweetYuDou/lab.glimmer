from pickletools import optimize
from typing_extensions import runtime
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os
from IPython import get_ipython
from livelossplot.inputs.tf_keras import PlotLossesCallback
import tensorflow as tf
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from IPython.display import SVG, Image
import cv2
import torch.nn as nn
import torchvision
import torch.utils.data as Data
#测试电脑识别表情的效果
'''
i=1
plt.figure(figsize=(8,8))
for expression in os.listdir('E:/typing/pytorch/archive/test/'):
   img =image_utils.img_to_array(image_utils.load_img(('E:/typing/pytorch/archive/test/'+ expression +'/'+ os.listdir('E:/typing/pytorch/archive/test/' + expression)[1])))
   img_gray = np.zeros([48,48],img.dtype)
   
   for k in range(48):
      for l in range(48):
         m = img[k,l]
         img_gray[k,l] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)    
   img_binary = np.where(img_gray/255>=0.5,1,0)  
   plt.subplot(1,7,i)
   plt.imshow(img_gray,cmap='gray')
   plt.title(expression)
   plt.axis('off')
   i=i+1
plt.show()
'''

datagen_train = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)#imagedatagenerator对图像加强处理，强化训练
#输入文件途径时windows系统要把'\'改成'/'！！！！！！！！
train_generator = datagen_train.flow_from_directory('E:/typing/pytorch/archive/test/',#导入数据，设置数据批数
                                                batch_size = 64,
                                                target_size=(48, 48),#图像大小
                                                shuffle=True, #每次训练随机打乱
                                                color_mode='grayscale',#图像灰度化
                                                class_mode='categorical')

datagen_test = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

test_generator = datagen_test.flow_from_directory('E:/typing/pytorch/archive/test/',
                                                batch_size = 64,
                                                target_size=(48, 48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')

model = tf.keras.models.Sequential()#搭建卷积神经网络   

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape =(48,48,1)))
model.add(BatchNormalization())#将batch_size初始为64，因为batch_size会下降（为什么）
model.add(MaxPooling2D(2, 2))#池化加强数据
model.add(Dropout(0.25))#防止过拟合

model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01) ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(8, activation='softmax'))#取概率中最大的，机器最有可能识别该图片的表情种类

model.compile(optimizer=Adam(learning_rate=0.0005, decay=1e-6),#评价函数用adam优化器和交叉熵损失函数对训练结果分析
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()#输出该卷积神经网络各层的参数
epochs = 50 #神经网络训练次数
steps_per_epoch = train_generator.n/train_generator.batch_size #图像数/单次训练批数
testing_steps = test_generator.n/test_generator.batch_size
#保持每次训练的数据
checkpoint = ModelCheckpoint("model_weights.h5", monitor="val_accuracy", save_weights_only=True, mode='max', verbose=1)
#逐渐减小学习率提升机器学习效果（正确率）
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, patience = 2, min_lr=0.00001, model='auto')
#登记回调函数，用于下面的model.fit调用，监视训练过程中的变量，PlotLossesCallback绘图
callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]

history = model.fit( #训练，记录loss和accuracy变化
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=testing_steps,
    callbacks=callbacks
)

