import numpy as np
import os
import cv2 as cv

test_data = []
train_data = []
crowd_data = []
face_data = []


for i in range(1,7):
	f_path = ("/home/akash/Desktop/{0}/Faces/".format(i))
	c_path = ("/home/akash/Desktop/{0}/Crowd/".format(i))	
	while images in f_path:
		im = cv.imread(images)
		face_data.append((im, i))
		print i
	while images1 in c_path:
		im1 = cv.imread(images1)	
		crowd_data.append((im1, 7))

crowd_data = crowd_data[:3500]
for tup in crowd_data:
	train_data.append(tup)
for tup in face_data:
	train_data.append(tup)

np.random.shuffle(train_data)
np.random.shuffle(train_data)
length = train_data.shape[1]
test_length = 0.1*length
test_data = train_data[:test_length]
train_data = [test_length:]
np.save("test.npy", test_data)
np.save("train.npy", train_data)	
