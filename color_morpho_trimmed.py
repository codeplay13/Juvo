# python color_morpho_trimmed.py 

from imutils.video import VideoStream
import cv2
import numpy as np
 
camera = VideoStream(src=0).start()
 
while True:
	image = camera.read()
	frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	thresh = cv2.inRange(frame_to_thresh, (100, 145, 80), (255, 255, 255)) # set min & max HSV values
	
	kernel = np.ones((5,5),np.uint8)
	mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
 
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
 
	if len(cnts) > 0:

		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 10:
			cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(image, center, 3, (0, 0, 255), -1)
			cv2.putText(image,"centroid", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
			cv2.putText(image,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)

		cv2.imshow("Original", image)
		cv2.imshow("Thresh", thresh)
		
		if cv2.waitKey(1) & 0xFF is ord('q'):
			break

cv2.destroyAllWindows()
camera.stop()