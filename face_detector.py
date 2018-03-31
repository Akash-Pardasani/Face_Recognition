# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os.path

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
for i in range(60000):
	imagePath = ("/home/akash/Desktop/Shailendra/Frames/frame%d" %i)
	#print(i)
	#imagePath = 'frame321.jpg'
	if os.path.exists(imagePath):
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
		rects = detector(gray, 1)
		print(len(rects))

# loop over the face detections
		for (j, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			if (len(rects) == 1):
				im1 = image[y:y+h, x:x+w]
				im1 = cv2.resize(im1, (240,240))	
				cv2.imwrite("/home/akash/Desktop/Shailendra/Faces/face%d" %i, im1)
	# loop over the face parts individually
	
		
