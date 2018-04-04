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
        self.batch_size = 50
        self.test_model = None
        self.answers = None
    
    def load_train_test(self):
        test1 = np.load("drive/app/Data/test1.npy", encoding = 'latin1')
        print('3')
        train1 = np.load("drive/app/Data/train1.npy", encoding = 'latin1')
        print('4')
        X_train = []
        Y_train = []
        i = 0
        for image in train1:
            if i % 2 == 0:
              X_train.append(image[0])
              buckets = [0]*7
              buckets[image[1]-1] = 1
              Y_train.append(buckets)
            i = i + 1
        print('5')
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        print('6')
        X_test = []
        Y_test = []
        i = 0
        for image in test1:
            if i % 2 == 0:
              X_test.append(image[0])
              buckets = [0]*7
              buckets[image[1]-1] = 1
              Y_test.append(buckets)
            i = i + 1
        print('7')
        self.X_test = np.array(X_test)
        self.Y_test = np.array(Y_test)
        print('8')
    
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
        print(model.summary())
        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(7, activation='softmax'))
        print(model.summary())
        print(model.get_weights())
        self.model = model
                       
    def train_model(self):
        self.model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.Y_train, validation_split=self.split, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        self.history = history
        self.model.save("drive/app/trained_model.h5")
    
    def plot_loss(self):
        """
        print(self.history.history.keys())
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'])
        plt.show()
        """
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    def make_predictions(self):
        print("1")
        #test_model = load_model("drive/app/trained_model.h5")
        print("2")
        #self.test_model = test_model
        predictions = self.model.predict(self.X_test, verbose=2)
        print("3")
        answers = []
        """
        c = 0
        for i in range(len(predictions)):
          for j in range(len(predictions))
          if self.Y_test[i] == predictions[i]:
            c = c + 1
        accuracy = c/len(predictions)
        print(accuracy)
        print(c)
        print(len(predictions))
        """
        c = 0
        for j in range(len(predictions)):
            max = -1
            index = -1
            for i in range(len(predictions[j])):
                if(predictions[j][i] > max):
                    max = predictions[j][i]
                    index = i
            if self.Y_test[j][index] == 1:
              c = c + 1
            #answers.append(index)
        print(c/len(predictions))
        print(c)
        print(len(predictions))
        #print(answers)
        #self.answers = np.array(answers)
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
    print("1")
    obj.load_train_test()
    print('2')
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
        #obj.plot_train_test()
