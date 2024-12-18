import argparse
import cv2
import numpy as np
from scipy.io import savemat
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

def clustering( image: NDArray,
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

	parser = argparse.ArgumentParser( description="script for clustering frame of video")

	parser.add_argument( "-i", "--input", type=str, required=True, help="Path to the Video")
	parser.add_argument( "-t", "--threshold", type=float, default=0.5, required=False, help="Binary Threshold")
	parser.add_argument( "-e", "--eps", type=int, default=20, required=False, help="Minimum Distance (in pixels)")
	parser.add_argument( "-s", "--samples", type=int, default=10, required=False, help="Minimum Number of Samples in each Cluster")

	args = vars( parser.parse_args())

	input_video = args[ "input"]
	threshold = args[ "threshold"]
	eps = args[ "eps"]
	samples = args[ "samples"]

	capture = cv2.VideoCapture( input_video)

	frames = []

	while capture.isOpened():
		ret, frame = capture.read()

		if not ret:
			break

		frames.append( frame)

	frames = np.array( frames)

	all_clusters = []

	for i in range( len( frames)):
		print( f"Frame { i + 1}/{ len( frames)}")

		image = frames[ i]

		clusters = clustering( image=image,
							   threshold=threshold,
							   eps=eps,
							   min_samples=samples)

		if clusters is not None:
			all_clusters.append( np.c_[ clusters, ( i + 1) * np.ones_like( clusters[ :, 0])])

	savemat( "clusters.mat", mdict={ "clusters": np.concatenate( all_clusters)})