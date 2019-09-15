# python detection_with_mapping.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import FPS
import speech_recognition as sr
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
	"dog", "horse", "motorbike", "person", "pottedplant", 
	"sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()
engine = pyttsx3.init()
r = sr.Recognizer()
while True:
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()	
	ch = cv2.waitKey(1)
	for i in np.arange(0, detections.shape[2]):  
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			label2 = "{}".format(CLASSES[idx])
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			if ch == ord("y"):
				engine.say("There is " + label2)
				engine.runAndWait()
			if ch == ord("s"):
				with sr.Microphone() as source:
					audio = r.listen(source, timeout = None)
				try:
					input = r.recognize_google(audio)
					if(input == label2):
						engine.say("Yes! There is" + label2)
						for i in range(len(CLASSES)):
							if CLASSES[i]==input:
								index = i
								break
						engine.runAndWait()
						while True:	
							ret, single_frame = cap.read()
							single_frame = imutils.resize(single_frame, width=400)
							(h, w) = single_frame.shape[:2]
							blob_to_read = cv2.dnn.blobFromImage(single_frame, 0.007843, (300, 300), 127.5)
							net.setInput(blob_to_read)
							all_detections = net.forward()
							ch2 = cv2.waitKey(1)
							for i in np.arange(0, all_detections.shape[2]):
								if index == int(all_detections[0, 0, i, 1]):
									box_coordinates = all_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
									(sX, sY, eX, eY) = box_coordinates.astype("int")
									cv2.rectangle(single_frame, (sX, sY), (eX, eY), (255,0,0), 2)
							cv2.imshow("Mapped", single_frame)
							if ch2 == ord("m"):
								cv2.destroyWindow("Mapped")
								break
					else:
						engine.say("There is no " + input)
						engine.runAndWait()
				except:
					engine.say("Couldn't hear you!")
					engine.runAndWait()
	cv2.imshow("Input", frame)
	if ch == ord("q"):
		break
	fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
cap.release()