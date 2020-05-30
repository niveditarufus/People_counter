class TrackableObject(object):
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		self.objectID = objectID
		self.centroids = [centroid]
		# a boolean to check if the object is counted or not
		self.counted = False
		