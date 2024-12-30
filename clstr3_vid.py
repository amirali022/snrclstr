from typing import Tuple, Optional
import warnings
import argparse
import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat
from skimage.morphology import disk, erosion
from sklearn.cluster import KMeans

def co_dist( points: NDArray):
	"""Calculates pairwise distance between every points in points

	Args:
		points: list or array of points

	Returns:
		Pairwise distance matrix
	"""
	with warnings.catch_warnings():
		warnings.filterwarnings( "ignore")
		c = np.array( [ [ np.linalg.norm( points[ i] - points[ j]) for j in range( len( points))] for i in range( len( points))])

		# Set self distance to infinity
		for i in range( len( c)):
			c[ i][ i] = float( "inf")

		c[ np.isnan( c)] = float( "inf")

		return c

def clustering( image: NDArray,
			    mask_polygon: NDArray,
				blur_kernel_size: Tuple[ int, int] = ( 5, 5),
				blur_sigmaX: float = 2.0,
				blur_sigmaY: float = 2.0,
				threshold: float = 0.5,
				erosion_disk_radius: int = 4,
				max_clusters: int = 20,
				min_dist: float = 25) -> Optional[ NDArray]:
	"""Performs image processing and clustering on a given image

	Args:
		image: input image.
		mask_polygon: Polygon defining the region of interest.
		blur_kernel_size: Kernel size for Gaussian blurring.
		blur_sigmaX: Standard deviation in x direction for Gaussian blurring.
		blur_sigmaY: Standard deviation in y direction for Gaussian blurring.
		threshold: Threshold for binary image conversion.
		erosion_disk_radius: Radius of the disk structuring element for erosion.
		max_clusters: Maximum number of clusters for K-Means.
		min_dist: Minimum distance between centroids.

	Returns:
        Cluster centers as a NumPy array, or None if no bright pixels are found.
	"""

	# Convert to grayscale
	gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)

	# Create a mask using the polygon
	mask = cv2.fillPoly( img=np.zeros_like( gray_image),
					     pts=[ mask_polygon],
						 color=255)
	
	# Apply the mask to the grayscale image
	masked_image = cv2.bitwise_and( gray_image, mask)

	# Blur the masked image
	blurred_image = cv2.GaussianBlur( src=masked_image,
								  	  ksize=blur_kernel_size,
									  sigmaX=blur_sigmaX,
									  sigmaY=blur_sigmaY)
	
	# Threshold the blurred image to obtain a binary image
	_, binary_image = cv2.threshold( src=blurred_image,
								 	 thresh=threshold * 255,
									 maxval=255,
									 type=cv2.THRESH_BINARY)
	
	# Erode the binary image to remove small noise
	footprint = disk( radius=erosion_disk_radius)
	eroded_frame = erosion( image=binary_image,
							footprint=footprint)
	
	# Extract bright pixel coordinates
	bright_pixels = np.column_stack( np.where( eroded_frame > 0))

	# If no bright pixels are found, return None
	if len( bright_pixels) < 1:
		return None
	
	# Perform K-Means clustering on the bright pixels
	kmeans = KMeans( n_clusters=max_clusters, n_init="auto").fit( bright_pixels)

	labels = kmeans.labels_
	kmeans_cluster_centers = kmeans.cluster_centers_

	# Calculate pairwise cluster centeroids distance
	c = co_dist( kmeans_cluster_centers)

	# Loop until there exists a pair that violating minimum distance
	while ( c < min_dist).any():
		# Finding closest pair
		a, b = np.unravel_index( c.argmin(), c.shape)

		# Counting number of points in each centeroid
		a_n = ( labels == a).sum()
		b_n = ( labels == b).sum()

		# Merging two centeroids by weighted mean of the two clusters
		kmeans_cluster_centers[ a] = ( ( a_n * kmeans_cluster_centers[ a]) + ( b_n * kmeans_cluster_centers[ b])) / ( a_n + b_n)

		# Purging the other centeroid
		kmeans_cluster_centers[ b] = float( "inf")
		
		# Assigning samples of the purged centeroid to the remaining one
		labels[ labels == b] = a

		# Calculate pairwise distance again
		c = co_dist( kmeans_cluster_centers)

	# Selecting valid and remaining centeroids
	cluster_centers = kmeans_cluster_centers[ np.where( np.isfinite( kmeans_cluster_centers).all( axis=1))[ 0]].astype( int)

	return np.array( cluster_centers)

if __name__ == "__main__":

	parser = argparse.ArgumentParser( description="script for clustering frame of video")

	parser.add_argument( "-i", "--input", type=str, required=True, help="Path to the Video")
	parser.add_argument( "-t", "--threshold", type=float, default=0.5, required=False, help="Binary Threshold")
	parser.add_argument( "-c", "--clusters", type=int, required=True, help="Maximum Number of Clusters")
	parser.add_argument( "-m", "--min_dist", type=float, required=True, help="Minimum Distance Between Clusters")

	args = vars( parser.parse_args())

	input_video = args[ "input"]
	threshold = args[ "threshold"]
	max_clusters = args[ "clusters"]
	min_dist = args[ "min_dist"]

	capture = cv2.VideoCapture( input_video)

	frames = []

	while capture.isOpened():
		ret, frame = capture.read()

		if not ret:
			break

		frames.append( frame)

	frames = np.array( frames)

	height, width, _ = frames[ 0].shape

	polygon = np.array( [
		[ width * 0.33, height],	# Bottom-left
		[ width * 0.66, height],	# Botton-Right
		[ width, 0],				# Top-Right
		[ 0, 0],					# Top-Left
	], np.int32)

	all_clusters = []

	for i in range( len( frames)):
		print( f"Frame { i + 1}/{ len( frames)}", end=" ")

		image = frames[ i]

		clusters = clustering( image=image,
							   mask_polygon=polygon,
							   blur_kernel_size=( 5, 5),
							   blur_sigmaX=2.0,
							   blur_sigmaY=2.0,
							   threshold=threshold,
							   erosion_disk_radius=4,
							   max_clusters=max_clusters,
							   min_dist=min_dist)
		if clusters is not None:
			print( f"Found { len( clusters)} clusters.")
			all_clusters.append( np.c_[ clusters, ( i + 1) * np.ones_like( clusters[ :, 0])])
		else:
			print( "Found No Clusters")
			
	savemat( "clusters.mat", mdict={ "clusters": np.concatenate( all_clusters)})