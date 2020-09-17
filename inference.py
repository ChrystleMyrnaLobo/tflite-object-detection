"""
Inference on tflite SSD Mobilenet v2 for single image - prepare input, post process output, evalute mAP, draw BB

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
"""

import argparse
import time
import os

import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
import xml.etree.ElementTree as Et

# Append to $PYTHONPATH path to models/research and cocoapi/PythonAPI
from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields
from object_detection.utils.label_map_util import create_categories_from_labelmap, get_label_map_dict

def prepare_input(image_path):
	""" Input image preprocessing for SSD MobileNet format
	args:
		image_path: path to image
	returns:
		input_data: numpy array of shape (1, width, height, channel) after preprocessing
	"""
	# NxHxWxC, H:1, W:2
	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]
	img = Image.open(image_path).convert('RGB').resize((width, height))
	# Using OpenCV
	# img = cv2.resize(cv2.imread(image_path), (width,height))
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# add N dim
	input_data = np.expand_dims(img, axis=0)

	# check the type of the input tensor
	if input_details[0]['dtype'] == np.float32:
		input_data = (np.float32(input_data) - args.input_mean) / args.input_std
	elif input_details[0]['dtype'] == np.uint8:
		input_scale, input_zero_point = input_details[0]["quantization"]
		input_data = input_data / input_scale + input_zero_point
		input_data = input_data.astype(np.uint8)
	return input_data

def postprocess_output(image_path):
	""" Output post processing
	args:
		image_path: path to image
	returns:
		boxes: numpy array (num_det, 4) of boundary boxes at image scale
		classes: numpy array (num_det) of class index
		scores: numpy array (num_det) of scores
		num_det: (int) the number of detections
	"""
	# SSD Mobilenet tflite model returns 10 boxes by default.
	# Use the output tensor at 4th index to get the number of valid boxes
	num_det = int(interpreter.get_tensor(output_details[3]['index']))
	boxes = interpreter.get_tensor(output_details[0]['index'])[0][:num_det]
	classes = interpreter.get_tensor(output_details[1]['index'])[0][:num_det]
	scores = interpreter.get_tensor(output_details[2]['index'])[0][:num_det]

	# Scale the output to the input image size
	img_width, img_height = Image.open(image_path).size # PIL
	# img_height, img_width, _ = cv2.imread(image_path).shape # OpenCV

	df = pd.DataFrame(boxes)
	df['ymin'] = df[0].apply(lambda y: max(1,(y*img_height)))
	df['xmin'] = df[1].apply(lambda x: max(1,(x*img_width)))
	df['ymax'] = df[2].apply(lambda y: min(img_height,(y*img_height)))
	df['xmax'] = df[3].apply(lambda x: min(img_width,(x * img_width)))
	boxes_scaled = df[['ymin', 'xmin', 'ymax', 'xmax']].to_numpy()

	return boxes_scaled, classes, scores, num_det

def draw_boundaryboxes(image_path, annotation_path):
	""" Draw the detection boundary boxes
	args:
		image_path: path to image
		annotation_path: path to groundtruth in Pascal VOC format .xml
	"""
	# Draw detection boundary boxes
	dt_boxes, dt_classes, dt_scores, num_det = postprocess_output(image_path)
	image = cv2.imread(image_path)
	for i in range(num_det):
		[ymin, xmin, ymax, xmax] = list(map(int, dt_boxes[i]))
		cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
		cv2.putText(image, '{}% score'.format(int(dt_scores[i]*100)), (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,255,0), 1)

	# Draw groundtruth boundary boxes
	label_map_dict = get_label_map_dict(args.label_file)
	# Read groundtruth from XML file in Pascal VOC format
	gt_boxes, gt_classes = voc_parser(annotation_path, label_map_dict)
	for i in range(len(gt_boxes)):
		[ymin, xmin, ymax, xmax] = list(map(int, gt_boxes[i]))
		cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0, 255, 255), 2)

	saved_path = "out_" + os.path.basename(image_path)
	cv2.imwrite(os.path.join(saved_path), image)
	print("Saved at", saved_path)

def voc_parser(path_to_xml_file, label_map_dict):
	"""Parser for Pascal VOC format annotation to TF OD API format
	args:
		path_to_xml_file : path to annotation in Pascal VOC format
		label_map_dict : dictionary of class name to index
	returns
		boxes: array of boundary boxes (m, 4) where each row is [ymin, xmin, ymax, xmax]
		classes: list of class index (m, 1)
		where m is the number of objects
	"""
	boxes = []
	classes = []

	xml = open(path_to_xml_file, "r")
	tree = Et.parse(xml)
	root = tree.getroot()
	xml_size = root.find("size")

	objects = root.findall("object")
	if len(objects) == 0:
		print("No objects for {}")
		return boxes, classes

	obj_index = 0
	for obj in objects:
		class_id = label_map_dict[obj.find("name").text]
		xml_bndbox = obj.find("bndbox")
		xmin = float(xml_bndbox.find("xmin").text)
		ymin = float(xml_bndbox.find("ymin").text)
		xmax = float(xml_bndbox.find("xmax").text)
		ymax = float(xml_bndbox.find("ymax").text)
		boxes.append([ymin, xmin, ymax, xmax])
		classes.append(class_id)
	return boxes, classes

def evaluate_single_image(image_path, annotation_path):
	""" Evaluate mAP on image
	args:
		image_path: path to image
		annotation_path: path to groundtruth in Pascal VOC format .xml
	"""
	categories = create_categories_from_labelmap(args.label_file)
	label_map_dict = get_label_map_dict(args.label_file)
	coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories)
	image_name = os.path.basename(image_path).split('.')[0]

	# Read groundtruth from XML file in Pascal VOC format
	gt_boxes, gt_classes = voc_parser(annotation_path, label_map_dict)
	dt_boxes, dt_classes, dt_scores, num_det = postprocess_output(image_path)

	coco_evaluator.add_single_ground_truth_image_info(
		image_id=image_name,
		groundtruth_dict={
			standard_fields.InputDataFields.groundtruth_boxes:
			np.array(gt_boxes),
			standard_fields.InputDataFields.groundtruth_classes:
			np.array(gt_classes)
	})
	coco_evaluator.add_single_detected_image_info(
		image_id=image_name,
		detections_dict={
			standard_fields.DetectionResultFields.detection_boxes:
			dt_boxes,
			standard_fields.DetectionResultFields.detection_scores:
			dt_scores,
			standard_fields.DetectionResultFields.detection_classes:
			dt_classes
		})

	coco_evaluator.evaluate()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-i',
		'--image',
		default='images/1.jpg',
		help='image for object detection')
	parser.add_argument(
		'-a',
		'--annotation',
		default='annotations/1.xml',
		help='ground truth for object detection in Pascal VOC format')
	parser.add_argument(
		'-m',
		'--model_file',
		default='ssd_mobilenet_oid_v1_float.tflite',
		help='.tflite model to be executed')
	parser.add_argument(
		'-l',
		'--label_file',
		default='label_map.pbtxt',
		help='name of file containing labels pascal voc format')
	parser.add_argument(
		'--input_mean',
		default=127.5, type=float,
		help='input_mean')
	parser.add_argument(
		'--input_std',
		default=127.5, type=float,
		help='input standard deviation')
	args = parser.parse_args()

	interpreter = tf.lite.Interpreter(model_path=args.model_file)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	input_data = prepare_input(args.image)
	interpreter.set_tensor(input_details[0]['index'], input_data)

	start_time = time.time()
	interpreter.invoke()
	stop_time = time.time()
	print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

	boxes, classes, scores, num_det = postprocess_output(args.image)
	evaluate_single_image(args.image, args.annotation)

	draw_boundaryboxes(args.image, args.annotation)
