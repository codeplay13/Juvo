# python mapping_with_IP.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

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

def left_or_right(k, centX, input):
	for j in range(k):
		if (centX[j] < 187):
			engine.say(input + str(j+1) + "is at the left")
		elif (centX[j] < 222):
			engine.say(input + str(j+1) + "is at the center")
		else:
			engine.say(input + str(j+1) + "is at the right")
		engine.runAndWait()

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
					audio = r.listen(source)
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
							centX = [0]*5
							centY = [0]*5
							k = 0
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
									centX[k] = int((startX+endX)/2)
									centY[k] = int((startY+endY)/2)
									k+=1
							frame_to_thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
							thresh = cv2.inRange(frame_to_thresh, (100, 145, 80), (255, 255, 255)) # set min & max HSV values
							kernel = np.ones((5,5),np.uint8)
							mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
							mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
							cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
							center = None
							if len(cnts) > 0:
								c = max(cnts, key=cv2.contourArea)
								((x, y), radius) = cv2.minEnclosingCircle(c)
								M = cv2.moments(c)
								center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
								cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
								cv2.circle(frame, center, 3, (0, 0, 255), -1)
								cv2.putText(frame,"centroid", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
								cv2.putText(frame,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
							if ch2 == ord("x"):
								left_or_right(k, centX, input)
							cv2.imshow("Mapped", frame)
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