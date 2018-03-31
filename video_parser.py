import cv2
import os.path
print(cv2.__version__)
count = 0
for i in range(1,10):
	loc = ('/home/akash/Desktop/Sandeep/Videos/vid%d.mp4' %i)
	if os.path.exists(loc):
		vidcap = cv2.VideoCapture(loc)
		success,image = vidcap.read()
		success = True
		while success:
			cv2.imwrite("/home/akash/Desktop/Sandeep/Frames/frame%d.jpg" % count, image)     # save frame as JPEG file
			success,image = vidcap.read()
			#print 'Read a new frame: ', success
			count += 1
print "Done"
