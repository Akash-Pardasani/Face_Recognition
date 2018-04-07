#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 00:25:30 2018

@author: architaggarwal
"""

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.models import load_model
from matplotlib import pyplot as plt
import sklearn
import itertools
import matplotlib.pyplot as plt
#  %matplotlib inline

class FaceID(object):
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.history = None
        self.epochs = 10
        self.split = 0.33
        self.batch_size = 50
        self.test_model = None
        self.answers = None
    
    def load_train_test(self):
        test1 = np.load("drive/app/Data/Test_Data_2_wo_final.npy", encoding = 'latin1')
        np.random.shuffle(test1)
        train1 = np.load("drive/app/Data/train1.npy", encoding = 'latin1')
        np.random.shuffle(train1)
        X_train = []
        Y_train = []
        i = 0
        for image in train1:
          X_train.append(image[0])
          buckets = [0]*7
          buckets[image[1]-1] = 1
          Y_train.append(buckets[:])
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        X_test = []
        Y_test = []
        w = 0
        q = 0
        for image in test1:
          q = q + 1
          if (image[1] - 1) == 6:
            w = w + 1
          X_test.append(image[0])
          buckets = [0]*7
          buckets[image[1]-1] = 1
          Y_test.append(buckets[:])
        print(w)
        print(q)
        self.X_test = np.array(X_test)
        self.Y_test = np.array(Y_test)
        w = 0
        for label in range(len(Y_test)):
          print(self.Y_test[label])
          if self.Y_test[label][6] == 1:
            w = w + 1
        print(w)
    
    def store_pretrained_vgg16(self):
        vgg16_model = keras.applications.vgg16.VGG16()
        print(vgg16_model.summary())
        print(type(vgg16_model))
        vgg16_model.save("drive/app/model.h5")
        
    def load_vgg(self):
        new_model = load_model("drive/app/model.h5")
        model = Sequential()
        for layer in new_model.layers:
            model.add(layer)
        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(7, activation='softmax'))
        self.model = model
                       
    def train_model(self):
        self.model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.Y_train, validation_split=self.split, epochs=self.epochs, batch_size=self.batch_size, verbose=2, shuffle=True)
        self.history = history
        self.model.save("drive/app/trained_model.h5")
    
    def plot_loss(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    def make_predictions(self):
        predictions = self.model.predict(self.X_test, verbose=2)
        answers = []
        c = 0
        d = 0
        e = 0
        for j in range(len(predictions)):
            max = -1
            index = -1
            for i in range(len(predictions[j])):
                if(predictions[j][i] > max):
                    max = predictions[j][i]
                    index = i
            if self.Y_test[j][index] == 1:
              d = d + 1
              c = c + 1
        print("Test Accuracy = {}".format(c/len(predictions)))
if __name__ == '__main__':
    obj = FaceID()
    obj.load_train_test()
    print("Download Model?")
    first_time = input()
    first_time = int(first_time)
    if first_time == 1:
        obj.store_pretrained_vgg16()
    print("Fit model?")
    first_time = input()
    first_time = int(first_time)
    if first_time == 1:
        obj.load_vgg()
        obj.train_model()
        obj.plot_loss()
    print("Test model?")
    first_time = input()
    first_time = int(first_time)
    if first_time == 1:
        obj.make_predictions()
