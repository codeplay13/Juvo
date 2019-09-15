from imutils.video import VideoStream
import cv2
import numpy as np

vs = VideoStream(src=0).start()
while True:
	frame = vs.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_range = np.array([100, 145, 80])
	upper_range = np.array([255, 255, 255])

	mask = cv2.inRange(hsv, lower_range, upper_range)
	cv2.imshow("Output", frame)

	result = cv2.bitwise_and(frame, frame, mask=mask)
	cv2.imshow("Result", result)

	if cv2.waitKey(1) == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()