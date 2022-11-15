import cv2
import numpy as np

KMEANS_ITERATIONS = 100
KMEANS_EPSILON = 0.0
KMEANS_RUNS = 10
KMEANS_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    KMEANS_ITERATIONS,
    KMEANS_EPSILON
)


def kmeans(points, clusters, image_shape):
    _, labels, centers = cv2.kmeans(points, clusters, None, KMEANS_CRITERIA, KMEANS_RUNS, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    if image_shape is not None:
        labels = labels.reshape((image_shape[0], image_shape[1]))

    return centers, labels
