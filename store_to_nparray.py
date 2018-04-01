import numpy as np
import os
from scipy.misc import imread, imsave, imresize

test_data = []
train_data = []
crowd_data = []
face_data = []


for i in range(1,7):
	f_path = ("/home/akash/Desktop/{0}/Faces".format(i))
	c_path = ("/home/akash/Desktop/{0}/Crowd".format(i))
		
	for images in os.listdir(f_path):
		image = os.path.join(f_path, images)
		im = imread(image)
		face_data.append((im, i))
		print "face", i
	for images1 in os.listdir(c_path):
		image1 = os.path.join(c_path, images1)
		im1 = imread(image1)	
		crowd_data.append((im1, 7))
		print "crowd", i

np.random.shuffle(crowd_data)
crowd_data = crowd_data[:4000]
for tup in crowd_data:
	train_data.append(tup)
for tup in face_data:
	train_data.append(tup)

np.random.shuffle(train_data)
np.random.shuffle(train_data)
length = len(train_data)
test_length = 0.1*length
test_data = train_data[:int(test_length)]
train_data = train_data[int(test_length):]
np.save("test.npy", test_data)
np.save("train.npy", train_data)
test1 = np.load("train.npy")
for a in test1:
	print a[0].shape	
