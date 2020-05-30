from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared = 50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.nextObjectID = 0
		self.maxDistance = maxDistance
		self.maxDisappeared = maxDisappeared

	def registerObjects(self, centroid):
		# when registering an object, use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregisterObjects(self, objectID):
		# delete the object ID from
		# both dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def updateObjects(self, boundingBoxes):
		# if list of bounding boxes is empty
		if len(boundingBoxes) == 0:
			# mark all existing objects as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] +=1;

				# if the maximum limit is reached, deregister the object
				if(self.disappeared[objectID] > self.maxDisappeared):
					self.deregisterObjects(objectID)
			# return early as there are no tracking info
			return self.objects

		# initialise array to store the centroids in the frame
		inputCentroids = np.zeros((len(boundingBoxes), 2), dtype = "int")


		for (i, (startX, startY, endX, endY)) in enumerate(boundingBoxes):
			# derive the centroid from coordinates of the bounding boxes
			centroidX = int((startX + endX) / 2.0)
			centroidY = int((startY + endY) / 2.0)
			inputCentroids[i] = (centroidX, centroidY)

		# if there no objects being tracked currently
		if(len(self.objects) == 0):
			for centroid in inputCentroids:
				self.registerObjects(centroid)

		# try to match the input centroids to existing objects
		# and register them
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis = 1).argsort()
			cols = D.argmin(axis = 1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue;

				# if distance is greater than threshold don't 
				# associate the centroids to same object
				if D[row, col ] > self.maxDistance:
					continue;

				# set its new centroid, and reset the disappeared counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0;

				# indicate that we have examined each of the 
				# row and column indexes
				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if(D.shape[0] >= D.shape[1]):
				for row in unusedRows:
					objectID = objectIDs[row];
					self.disappeared[objectID] += 1

					if(self.disappeared[objectID] > self.maxDisappeared):
						self.deregisterObjects(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.registerObjects(inputCentroids[col])
					
		# return the trackable objects
		return self.objects