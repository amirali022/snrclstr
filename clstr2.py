from typing import Tuple, Optional
import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.morphology import disk, erosion

def KMeans( X: NDArray,
		    n_clusters: int,
			centeroid_distance_lambda: float,
			min_dist: float,
			tol: float = 1e-4,
			max_iter: int = 100,
			n_iter: int = 10) -> Tuple[ NDArray, NDArray]:
	"""KMeans Clustering Algorithm with Minimum Centeroid Distance Constraint
	
	Args:
		X: Training instances to cluster of shape (n_samples, n_features).
		n_clusters: The number of clusters to form as well as the number of centroids to generate.
		centeroid_distance_lambda: The regularization parameter for the minimum distance constraint.
		min_dist: The minimum distance between centroids.
		tol: Tolerance for early stopping.
		max_iter: Maximum number of iterations for a single run.
		n_iter: Number of times the k-means algorithm will be run with different centroid seeds.

	Returns:
		A tuple containing:
			- The cluster centeroids.
			- The labels of each data point.
	"""

	n, _ = X.shape

	# Initialize best inertia and centroids
	best_inertia = float( "inf")
	best_centeroids = None
	best_labels = None

	# Run K-Means multiple times with different initializations
	for _ in range( n_iter):
		
		# Initialize centroids randomly
		np.random.seed()

		centeroids = X[ np.random.choice( n, n_clusters, replace=False)].astype( float)

		for _ in range( max_iter):
			new_centeroids = centeroids.copy()

			# Assign data points to the nearest centroid
			distances = np.linalg.norm( X[ :, np.newaxis] - centeroids, axis=2)
			labels = np.argmin( distances, axis=1)

			# Update centroids with regularization
			for i in range( n_clusters):

				# Calculate the mean of points assigned to this cluster
				if len( X[ labels == i]) > 0:
					mean_point = np.mean( X[ labels == i], axis=0) - centeroids[ i]
				else:
					mean_point = np.zeros_like( centeroids[ i])
				
				# Calculate the regularization term
				co_dist = np.array( [ np.linalg.norm( centeroids[ i] - centeroids[ j]) for j in range( n_clusters)])
				regularization_term = centeroid_distance_lambda * ( ( centeroids[ i] - centeroids[ co_dist < min_dist]) * 0.5).sum( axis=0)
				
				new_centeroids[ i] += mean_point + regularization_term

			# Check for convergence
			if np.linalg.norm( new_centeroids - centeroids) < tol:
				break

			centeroids = new_centeroids

		# Calculate inertia
		distances = np.linalg.norm( X[ :, np.newaxis] - centeroids, axis=2)
		inertia = np.sum( np.min( distances, axis=1) ** 2)

		# Update best solution if current inertia is lower
		if inertia < best_inertia:
			best_inertia = inertia
			best_centeroids = centeroids
			best_labels = labels

	return best_centeroids, best_labels

def clustering( image_path: str,
			    mask_polygon: NDArray,
				blur_kernel_size: Tuple[ int, int] = ( 5, 5),
				blur_sigmaX: float = 2.0,
				blur_sigmaY: float = 2.0,
				threshold: float = 0.5,
				erosion_disk_radius: int = 4,
				kmeans_n_clusters: int = 10,
				kmeans_centeroid_distance_lambda: float = 0.5,
				kmeans_min_dist: float = 10,
				kmeans_tol: float = 1e-4,
				kmeans_max_iter: int = 100,
				kmeans_n_iter: int = 10) -> Optional[ NDArray]:
	"""Performs image processing and clustering on a given image

	Args:
		image_path: Path to the input image.
		mask_polygon: Polygon defining the region of interest.
		blur_kernel_size: Kernel size for Gaussian blurring.
		blur_sigmaX: Standard deviation in x direction for Gaussian blurring.
		blur_sigmaY: Standard deviation in y direction for Gaussian blurring.
		threshold: Threshold for binary image conversion.
		erosion_disk_radius: Radius of the disk structuring element for erosion.
		kmeans_n_clusters: Number of clusters for K-Means.
		kmeans_centeroid_distance_lambda: Regularization parameter for minimum distance constraint.
		kmeans_min_dist: Minimum distance between centroids.
		kmeans_tol: Tolerance for early stopping in K-Means.
		kmeans_max_iter: Maximum iterations for K-Means.
		kmeans_n_iter: Number of times to run K-Means with different initializations.

	Returns:
        Cluster centers as a NumPy array, or None if no bright pixels are found.
	"""
	
	# Read the image
	image = cv2.imread( image_path)

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
	clusters, labels = KMeans( X=bright_pixels,
						   	   n_clusters=kmeans_n_clusters,
						   	   centeroid_distance_lambda=kmeans_centeroid_distance_lambda,
							   min_dist=kmeans_min_dist,
							   tol=kmeans_tol,
							   max_iter=kmeans_max_iter,
							   n_iter=kmeans_n_iter)
	
	# Convert cluster centers to integer coordinates
	cluster_centers = clusters.astype( int)

	return np.array( cluster_centers)

if __name__ == "__main__":

	image = cv2.imread( "sample.png")

	height, width, _ = image.shape

	polygon = np.array( [
		[ width * 0.33, height],	# Bottom-left
		[ width * 0.66, height],	# Botton-Right
		[ width, 0],				# Top-Right
		[ 0, 0],					# Top-Left
	], np.int32)

	clusters = clustering( image_path="sample.png",
					       mask_polygon=polygon,
						   blur_kernel_size=( 5, 5),
						   blur_sigmaX=2.0,
						   blur_sigmaY=2.0,
						   threshold=0.5,
						   erosion_disk_radius=4,
						   kmeans_n_clusters=10,
						   kmeans_centeroid_distance_lambda=0.5,
						   kmeans_min_dist=10,
						   kmeans_tol=1e-4,
						   kmeans_max_iter=100,
						   kmeans_n_iter=10)

	for c in clusters:
		print( f"[x:{ c[ 1]}\ty:{ c[ 0]}]")