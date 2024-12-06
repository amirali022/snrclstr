import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def clustering( image_path: str,
			    threshold: float,
				eps: float,
				min_samples: int):
	"""Finds and Returns Cluster Centers in image

	Args:
		image_path: path to input image.
		threshold: a float scalar ranges from 0.0 to 1.0. determines brightness threshold
		eps: parameter of DBSCAN algorithm. The maximum distance between two samples for one to be considered as in the neighborhood of the other
		min_samples: parameter of DBSCAN algorithm. The number of samples in a neighborhood for a point to be considered as a core point
	
	Returns:
		cluster_centers: NDArray
	"""

	image = cv2.imread( image_path)
	gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)

	height, width = gray_image.shape

	mask = np.zeros_like( gray_image)

	polygon = np.array( [
		[ width * 0.33, height],	# Bottom-left
		[ width * 0.66, height],	# Botton-Right
		[ width, 0],				# Top-Right
		[ 0, 0],					# Top-Left
	], np.int32)

	mask = cv2.fillPoly( mask, [ polygon], 255)

	masked_image = cv2.bitwise_and( gray_image, mask)

	blurred_image = cv2.GaussianBlur( masked_image, ( 5, 5), sigmaX=2, sigmaY=2)

	_, binary_image = cv2.threshold( blurred_image, threshold * 255, 255, cv2.THRESH_BINARY)

	bright_pixels = np.column_stack( np.where( binary_image > 0))

	# There is no bright pixel
	if len( bright_pixels) < 1:
		return None

	dbscan = DBSCAN( eps=eps, min_samples=min_samples).fit( bright_pixels)
	labels = dbscan.labels_

	unique_labels = set( labels)

	cluster_centers = []

	for label in unique_labels:

		if label == -1:
			continue

		cluster_points = bright_pixels[ labels == label]

		center = cluster_points.mean( axis=0).astype( int)

		cluster_centers.append( center)

	return np.array( cluster_centers)

if __name__ == "__main__":
	clusters = clustering( image_path="sample.png",
					       threshold=0.5,
						   eps=20,
						   min_samples=10)
	
	for c in clusters:
		print( f"[x:{ c[ 1]}\ty:{ c[ 0]}]")