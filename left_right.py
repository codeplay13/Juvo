# USAGE
# python left_right.py --image images/acute.jpg --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
from scipy.spatial import distance as dist
import pyttsx3
import argparse
import cv2
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "table",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(args["image"])
image = cv2.resize(image, (0,0), fx=0.1, fy=0.1)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

centX = [0]*5
centY = [0]*5
k = 0

speak = pyttsx3.init()

def find_scale(aX, aY, bX, bY):
	if(bX-aX != 0):
		m = abs((bY-aY)/(bX-aX))
		z = math.degrees(math.atan(m))+15
		print(z, "Degrees")
		if(z<38):
			return (math.cos(math.radians(z)))*4.5
		z = 90-z
		return (math.cos(math.radians(z)))*15
	elif (bY-aY == 0):
		return 4.5
	else:
		return 15.6

def left_or_right(k, centX):
	for j in range(k):
		if (centX[j] < 187):
			speak.say("Bottle" + str(j+1) + "is at the left")
		elif (centX[j] < 222):
			speak.say("Bottle" + str(j+1) + "is at the center")
		else:
			speak.say("Bottle" + str(j+1) + "is at the right")
	
# loop over the detections
for i in np.arange(0, detections.shape[2]):
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object	
	if (5 == int(detections[0, 0, i, 1])):
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# display the prediction
		label = "Bottle"
		cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[5], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[5], 2)
		cv2.rectangle(image, (414, 310), (416, 312), COLORS[5], 2)
		#cv2.line(image, (208, 1), (208, 310), (0,0,255), 2)
		centX[k] = int((startX+endX)/2)
		centY[k] = int((startY+endY)/2)
		k+=1
	if(i>0):
		cv2.line(image, (centX[0], centY[0]), (centX[1], centY[1]), (0,0,255), 2)
		D = dist.euclidean((centX[0], centY[0]), (centX[1], centY[1]))
		scale = find_scale(centX[0], centY[0], centX[1], centY[1])
		D = int(D*0.0265*scale)
		print(D, "Centimeters")
		D = str(D)
		speak.say("Bottles are " + D + " centimeters apart!")
left_or_right(k, centX)

# show the output image
speak.runAndWait()
cv2.imshow("Output", image)
cv2.waitKey(0)