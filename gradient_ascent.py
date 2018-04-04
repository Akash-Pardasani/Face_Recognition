#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 00:36:38 2018

@author: architaggarwal
"""

from keras.preprocessing import image as image_utils
import numpy as np
import scipy
import argparse
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16
from keras import backend as K
imageName = str("drive/app/atul.jpg")
image = image_utils.load_img(imageName, target_size=(224, 224))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
input_img = image
image_preprocessed=np.rollaxis(image,1,4)
model = keras.applications.vgg16.VGG16()

def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input],[model.layers[layer_idx].output])
	activations = get_activations([X_batch,0])
	return(activations)

activations = get_featuremaps(model, len(model.layers)-18, image)
#plt.imshow(self.activations[0][0][63],cmap = 'gray')
print(type(activations))
print(len(activations))
print(activations)
output_shape = np.shape(activations[0])
featuremap_size = np.shape(activations[0][0][0])
num_of_featuremaps = (np.shape(activations[0][0]))[2]
print(num_of_featuremaps)
layer_info=model.layers[len(model.layers)-18].get_config()
layer_name=layer_info['name']
input_shape=model.layers[len(model.layers)-18].input_shape
layer_param=model.layers[len(model.layers)-18].count_params()
if len(output_shape)==2:
	fig=plt.figure(figsize=(100,100))
	plt.imshow(activations[0].T,cmap='gray')
	plt.savefig("drive/app/featuremaps-layer-{}".format(len(model.layers)-18) + '.png')

else:
	fig=plt.figure(figsize=(100,100))
	subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
	print(subplot_num)
	for i in range(int(num_of_featuremaps)):
		print(i)
		ax = fig.add_subplot(subplot_num, subplot_num, i+1)
		#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
		ax.imshow(activations[0][0][:,:,i],cmap='gray')
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		#plt.tight_layout()
	plt
	plt.savefig("drive/app/featuremaps-layer-{}".format(len(model.layers)-18) + '.png')
