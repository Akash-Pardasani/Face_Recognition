import numpy as np
import cv2
import os.path

cascFacePath = "haarcascade_frontalface_default.xml"
cascEyePath = "haarcascade_eye.xml"
data = []
num_speaker = 0
num_bckg = 0
label = []
face_cascade = cv2.CascadeClassifier(cascFacePath)
eye_cascade = cv2.CascadeClassifier(cascEyePath)
for k in range(7):   # 1- Atul, 2 - Pant, 3 - Sadhguru, 4 - Raman, 5 - Sandeep, 6 - Shailendra
	for i in range(70000):
		if k==1:
			i = i+17002
		imagePath = ("/home/akash/Desktop/{0}/Frames/frame{1}.jpg".format(k,i))
		print(i)
		#imagePath = 'frame321.jpg'
		if os.path.exists(imagePath):
			image = cv2.imread(imagePath)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.2, 5)
			eyes = eye_cascade.detectMultiScale(gray, 1.2, 5)
			
			if (len(faces) == 1):
				(x, y, w, h) = faces[0]
				if (image.shape[1]!=0):
					im1 = image[y:y+h, x:x+w]
				if (im1.shape[1]!=0):
					im1 = cv2.resize(im1, (224,224))
					data.append((im1,k))
					num_speaker = num_speaker + 1
					cv2.imwrite("/home/akash/Desktop/{0}/Faces/face{1}.jpg".format(k,i), im1)				
					print("face detected")

			elif (len(faces) == 0):	
				if(len(eyes) == 2):
					(x1,y1,w1,h1) = eyes[0]
					(x2,y2,w2,h2) = eyes[1]
					e_span = (x2-x1-w1)
					y_mid = (y1+y1+h1+y2+y2+h2)/4
					height = 1.6*4*e_span
					if (image.shape[1]!=0):
						im1 = image[y_mid-int(height):y_mid+int(2*height), x1-int(6*e_span/5):x2+w2+int(6*e_span/5)]
					if(im1.shape[1]!=0):					
						im1 = cv2.resize(im1, (224,224))
						data.append((im1,k))
						num_speaker = num_speaker + 1
						cv2.imwrite("/home/akash/Desktop/{0}/Faces/face{1}.jpg".format(k,i), im1)
						print("eyes detected")
				else:
					if(image.shape[1]!=0):					
						im1 = cv2.resize(image, (224,224))
						data.append((im1,7))
						num_bckg = num_bckg + 1
						cv2.imwrite("/home/akash/Desktop/{0}/Crowd/crowd{1}.jpg".format(k,i), im1)
						print("bckg detected")
		
			else:
				if(image.shape[1]!=0):
					im1 = cv2.resize(image, (224,224))
					data.append((im1,7))
					num_bckg = num_bckg + 1
					cv2.imwrite("/home/akash/Desktop/{0}/Crowd/crowd{1}.jpg".format(k,i), im1)
					print("bckg detected")

data = np.array(data)
np.save('./Data_Face_Reg.npy', data)
print num_speaker, num_bckg
