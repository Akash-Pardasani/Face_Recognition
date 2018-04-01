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
#from sklearn.metrics import confusion_metrics
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
        self.batch_size = 100
        self.test_model = None
        self.answers = None
    
    def load_train_test(self):
        test1 = np.load("drive/app/test1.npy")
        train1 = np.load("drive/app/train1.npy")
        X_train = []
        Y_train = []
        for image in train1:
            X_train.append(image[0])
            Y_train.append(image[1])
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        X_test = []
        Y_test = []
        for image in test1:
            X_test.append(image[0])
            Y_test.append(image[1])
        self.X_test = np.array(X_test)
        self.Y_test = np.array(Y_test)
    
    def store_pretrained_vgg16(self):
        vgg16_model = keras.applications.vgg16.VGG16()
        print(vgg16_model.summary())
        print(type(vgg16_model))
        model = Sequential()
        for layer in vgg16_model.layers:
            model.add(layer)
        print(model.summary())
        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(7, activation='softmax'))
        print(model.summary())
        model.save("drive/app/model.h5")
        
    def load_vgg(self):
        new_model = load_model("drive/app/model.h5")
        print(new_model.summary())
        print(new_model.get_weights())
        self.model = new_model
                       
    def train_model(self):
        self.model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.Y_train, validation_split=self.split, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        self.history = history
        self.model.save("drive/app/trained_model.h5")
    
    def plot_loss(self):
        print(self.history.history.keys())
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'])
        plt.show()
    
    def test_model(self):
        self.test_model = load_model("drive/app/trained_model.h5")
        predictions = self.test_model.predict(self.X_test, verbose=2)
        answers = []
        for prediction in predictions:
            max = -1
            index = -1
            for i in range(len(prediction)):
                if(prediction[i] > max):
                    max = prediction[i]
                    index = i
            answers.append(index)
        print answers
        self.answers = np.array(answers)
    """
    def plot_train_test(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.answers, self.Y_test)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'])
        plt.show()
    """
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
        obj.test_model()
        #obj.plot_train_test()
