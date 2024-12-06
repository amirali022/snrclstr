# Clustering Region/Area of Interest of Sonar Images using Different Methods

## Overview
This project implements clustering algorithm to find the centers of bright regions in a sonar image. It includes image processing techniques like masking, blurring, and thresholding to prepare the image data for clustering.

## Methods

- DBScan
- KMeans
- KMeans with a regularization term to enforce minimum distance between cluster centroids

## Preprocessing Steps

- convertimg image to grayscale
- Applying a mask to isolate the region of interest
- Bluring the masked image to reduce noise
- Thresholding the blurred image to obtain a binary image
- Erosion of binary image to remove small noise
- Extracting bright pixel coordinates from the eroded image

### Requirements

- Python 3.11
- Jupyter Notebook

### Libraries

- numpy
- matplotlib
- cv2
- scikit-learn
- scikit-image