# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 20:29:02 2018

@author: servo97
"""
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        #initialize model with "channels_last" and channel dimension itself
        inputShape = (height,width,depth)
        chanDim = -1
        #if we are using channels_first", update input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        #1st CONV=>RELU=>POOL block
        model.add(Conv2D(64,(3,3),padding = "same",
                         input_shape= inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size = (3,3)))
        model.add(Dropout(0.15))
        #2nd (CONV=>RELU)*2=>POOL block
        model.add(Conv2D(128,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128,(3,3),padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.15))
        #3rd (CONV=>RELU)*2=>POOL block
        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.15))
        #4th (CONV=>RELU)*2=>POOL block
        model.add(Conv2D(512,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.15))
        # #5th (CONV=>RELU)*2=>POOL block
        # model.add(Conv2D(1024,(3,3),padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(1024,(3,3),padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.15))
        #1st fully connected layer =>RELU
        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return architecture
        return model
