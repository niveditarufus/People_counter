from resources.centroidtracker import CentroidTracker
from resources.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
help="# of skip frames between detections")
args = vars(ap.parse_args())

#initialize the list of labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

def preprocess(frame, H, W, writer):
	# resize the frame to have a maximum width of 500 pixels 
	# conversion of the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width = 500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# set the frame dimensions 
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# initilize writer if you want to save the video to disk
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

	return frame, rgb, (H, W), writer


def getFromModel(model):
	# convert the frame to a blob and pass the blob through the
	# network and obtain the detections
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
	model.setInput(blob)
	detections = model.forward()
	return detections


def getTracker(startX, startY, endX, endY):
	# construct a dlib rectangle from the box coordinates 
	# start the dlib correlation tracker
	tracker = dlib.correlation_tracker()
	rect = dlib.rectangle(startX, startY, endX, endY)
	tracker.start_track(rgb, rect)
	return tracker


def updated_position(pos):
	startX = int(pos.left())
	startY = int(pos.top())
	endX = int(pos.right())
	endY = int(pos.bottom())
	return startX, startY, endX, endY


def boxesFromDetection(detections, trackers):
	# loop over detections
	for i in range(detections.shape[2]):
		# filter outr weak detections
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:

			# get the class label
			index = int(detections[0, 0, i, 1])

			# ignore if the label is not a person
			if CLASSES[index] != "person":
				continue

			#get coordinates of box
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")

			# obtain the dlib tracker and add it to our list of trackers
			tracker = getTracker(startX, startY, endX, endY)
			trackers.append(tracker)


def counter(objects, trackableObjects, Up, Down, totalCount):
	for (objectID, centroid) in objects.items():
		# check if a trackable object exists with objectID
		to = trackableObjects.get(objectID, None)

		# if it doesn't exist, create one 
		if to is None:
			to = TrackableObject(objectID, centroid)

		# there is a trackable object and we can find the direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids give the
			# in which direction the object is moving
			y = [c[1] for c in to.centroids]
			direction = centroid[1]-np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has already been counted
			if not to.counted:

				# direction is negative(moving up)
				# AND centroid is above centre, count the obect
				if direction < 0 and centroid[1] < H // 2:
					Up += 1
					to.counted = True

				# direction is positive(moving down)
				# AND centroid is below centre, count the obect
				elif direction > 0 and centroid[1] > H // 2:
					Down += 1
					to.counted = True
		# store the trackable object
		trackableObjects[objectID] = to

		# get count of people inside
		totalCount = Down - Up

		# draw ID and centroid of object
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	return Up, Down, totalCount


def display(info, frame, writer):
	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	# write to disk if required
	if writer is not None:
		writer.write(frame)
	# display the frame
	cv2.imshow("Frame", frame)

# load the given Model
print("[INFO] Loading the given model...")
model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if there is no video path supplied, exit
if not args.get("input", False):
	print("[INFO] Did not receive any video file, exiting...")
	sys.exit(1)

print("[INFO] Opening video feed...")
vs = cv2.VideoCapture(args["input"])

# initilization of the video writer
writer = None

# intialize the dimensions of the frame,
# it will be initialized when the first frame is received
W = None
H = None

# instansiate the centroid Tracker
ct = CentroidTracker(maxDisappeared = 40, maxDistance = 50)

# instantiate a list to store each dlib tracker, and a dictionary 
# for mapping each unique objectID to a TrackableObject
trackers = []
trackableObjects = {}

# inititialize total frme processed thus far
# with the total number of objects that have moved either up or down
# and count of people inside
totalFrames = 0
Down = 0
Up = 0
totalCount = 0

# start the fps throughput estimator
fps = FPS().start()

# loop through all frames
while True:
	frame = vs.read()[1]

	# the end of video feed is reached
	if args["input"] is not None and frame is None:
		print("[INFO] End of video stream reached, exiting...")
		break

	# some preprocess operations
	frame, rgb, (H, W), writer = preprocess(frame, H, W, writer)

	# intialise the current status and list of bounding box per frame
	status = "Waiting"
	boundingBoxes = []

	if(totalFrames % args["skip_frames"] == 0):
		# detection phase
		# set status as detecting and
		# initialize new set of object trackers
		status = "Detecting"
		trackers = []

		# obtain detections through the given model
		detections = getFromModel(model)

		# obtain the list of dlib trackers per frame
		boxesFromDetection(detections, trackers)

	else:
		# tracking phase
		status = "Tracking"

		# loop over the trackers
		for tracker in trackers:
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpanpack updated positions
			startX, startY, endX, endY = updated_position(pos)

			# add the new coordinates to the list
			boundingBoxes.append((startX, startY, endX, endY))

	# draw a horizontal line in the center
	#once an object crosses this line we will 
	# determine whether they were moving 'up' or 'down'
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	#update object list with newly computed centroids
	objects = ct.updateObjects(boundingBoxes)

	# get the count of people inside
	Up, Down, totalCount = counter(objects, trackableObjects, Up, Down, totalCount)

	# a tuple of information to be displayed
	info = [("Inside", totalCount),("Exit", Up), ("Entry", Down), ("status", status)]
	display(info, frame, writer)

	# exit if 'q' key was pressed
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	# increment frames
	totalFrames += 1
	fps.update()

# stop the timer and relay the fps info
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# relase writer
if writer is not None:
	writer.release()

# release the video file pointer
vs.release()

cv2.destroyAllWindows()

