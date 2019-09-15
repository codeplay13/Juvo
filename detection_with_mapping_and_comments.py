# USAGE
# python detection_with_mapping_and_comments.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import FPS
import speech_recognition as sr
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
# trained using Pascal VOC2012 dataset which consists 20 classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()
speak = pyttsx3.init()
r = sr.Recognizer()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	
	ch = cv2.waitKey(1)

	# loop from 0 to all detections i.e. output layer
	for i in np.arange(0, detections.shape[2]):      # shape gives the size e.g. Y = (n, m) then Y.shape[0] = n;
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			label2 = "{}".format(CLASSES[idx])
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			if ch == ord("y"):
				speak.say("There is " + label2)
			
			if ch == ord("s"):
				with sr.Microphone() as source:
					audio = r.listen(source)
				try:
					input = r.recognize_google(audio)
					if(input == label2):
						speak.say("Yes! There is" + label2)
						for i in range(len(CLASSES)):
							if CLASSES[i]==input:
								index = i
								break
						while True:	
							ret, frame = cap.read()
							frame = imutils.resize(frame, width=400)
							(h, w) = frame.shape[:2]
							blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
							net.setInput(blob)
							detections = net.forward()
							ch2 = cv2.waitKey(1)
							for i in np.arange(0, detections.shape[2]):
								if index == int(detections[0, 0, i, 1]):
									box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
									(startX, startY, endX, endY) = box.astype("int")
									cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
							cv2.imshow("Mapped", frame)
							if ch2 == ord("m"):
								cv2.destroyWindow("Mapped")
								break
					else:
						speak.say("There is no " + input)
				except:
					speak.say("Couldn't hear ya!")

	# show the output frame
	cv2.imshow("Input", frame)
	
	if ch == ord("q"):
		break

	# update the FPS counter
	fps.update()
	speak.runAndWait()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()