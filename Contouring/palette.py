import cv2
import numpy as np

from sklearn.cluster import KMeans

KMEANS_ITERATIONS = 10
KMEANS_RUNS = 3


def kmeans(clusters, points, weights):
    process = KMeans(n_clusters=clusters, init='random', n_init=KMEANS_RUNS, max_iter=KMEANS_ITERATIONS, tol=0.5, algorithm='lloyd')
    process.fit(points, sample_weight=weights)
    result = process.predict(points, sample_weight=weights)
    return result


def expand_labels(colors, labels, image):
    # Each unique pixel color gets assigned
    # the label from the clustering process
    lookup = dict(zip(map(tuple, colors), labels))

    # Each pixel in the original image is once
    # resolved to its label using the lookup table.
    # This way, each pixel in the image gets its
    # cluster label assigned, so that contouring
    # can directly use pixel locations to find the
    # label instead of using a dict/hashmap.
    height, width, _ = image.shape
    expanded = np.empty((height, width))

    for y in range(height):
        for x in range(width):
            color = image[y][x]
            label = lookup[tuple(color)]
            expanded[y][x] = label

    return expanded
