import cv2
from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface import VGGFace



classes = 6
hidden_dim = 512





class Face_Reg:
	def __init__(self):
		self.im_path = ""





	def detect_and_resize(self, imagePath):
		faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
		    gray,
		    scaleFactor=1.1,
		    minNeighbors=5,
		    minSize=(30, 30)
		)
		faces = faceCascade.detectMultiScale(gray, 1.2, 5)
		if(len(faces)>1):
			print("Audience")
		else:
			im = Image.open(imagePath)
			(x, y, w, h) = faces[0]
			center_x = x+w/2
			center_y = y+h/2
			b_dim = min(max(w,h)*1.2,im.width, im.height) # WARNING : this formula in incorrect
			box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)
			crpim = im.crop(box).resize((224,224))

	def init_model(self):
		vgg = VGGFace(model='vgg16', include_top=False, input_shape=(224,224,3), pooling ='max')
		last_layer = vgg_model
