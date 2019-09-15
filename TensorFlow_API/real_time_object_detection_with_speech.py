import numpy as np
import cv2
import tensorflow as tf
import imutils
import pyttsx3

from imutils.video import FPS
from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_CKPT = 'ssd_mobilenet.pb'

PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

engine = pyttsx3.init()
cap = cv2.VideoCapture(0)
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		fps = FPS().start()
		while True:
			ret, frame = cap.read()
			frame = imutils.resize(frame, width=400)
			image_np_expanded = np.expand_dims(frame, axis=0)
			ch = cv2.waitKey(1)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
			vis_util.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
			if ch == ord('y'):
				objects = vis_util.objects_in_frame(np.squeeze(scores), np.squeeze(classes))
				for i in objects:
					engine.say("There is" + category_index[int(objects[i])]['name'])
					engine.runAndWait()
			cv2.imshow('image',frame)
			if ch == ord('q'):
				break
			fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()				
cap.release()