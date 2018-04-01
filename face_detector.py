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
for k in range(2,3):   # 1- Atul(18278), 2 - Pant, 3 - Sadhguru (video -2 is bad), 4 - Raman, 5 - Sandeep, 6 - Shailendra
	i = 0
	imagePath = ("/home/akash/Desktop/{0}/Frames/frame{1}.jpg".format(k,i))
	while (os.path.exists(imagePath)):
		imagePath = ("/home/akash/Desktop/{0}/Frames/frame{1}.jpg".format(k,i))
		if (k==5):
			i = i + 26
		else:
			i = i + 13
		print k, i
		if os.path.exists(imagePath):
			image = cv2.imread(imagePath)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			if (len(faces) == 1):
				(x, y, w, h) = faces[0]
				try:
					im1 = image[y:y+h, x:x+w]
					im1 = cv2.resize(im1, (224,224))
					data.append((im1,k))
					num_speaker = num_speaker + 1
					cv2.imwrite("/home/akash/Desktop/{0}/Faces/face{1}.jpg".format(k,i), im1)				
					print("face detected")
				except:
					continue

			else:
				eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)	
				if(len(eyes) == 2):
					(x1,y1,w1,h1) = eyes[0]
					(x2,y2,w2,h2) = eyes[1]
					e_span = (x2-x1-w1)
					y_mid = (y1+y1+h1+y2+y2+h2)/4
					height = 1.6*4*e_span
					try:
						im1 = image[max(y1,y2)-2*max(h1,h2):max(y1,y2)+5*max(h1,h2), max(x1,x2)-3*max(w1,w2):max(x1,x2)+2*max(w1,w2)]	
						im1 = cv2.resize(im1, (224,224))
						data.append((im1,k))
						num_speaker = num_speaker + 1
						cv2.imwrite("/home/akash/Desktop/{0}/Faces/face{1}.jpg".format(k,i), im1)
						print("eyes detected")
					except:
						continue
				else:
					try:					
						im1 = cv2.resize(image, (224,224))
						data.append((im1,7))
						num_bckg = num_bckg + 1
						cv2.imwrite("/home/akash/Desktop/{0}/Crowd/crowd{1}.jpg".format(k,i), im1)
						print("bckg detected")
		
					except:		
						continue
		

data = np.array(data)
np.save('./Data_Face_Reg.npy', data)
print num_speaker, num_bckg
