import cv2
import numpy as np

from sklearn.cluster import KMeans

KMEANS_ITERATIONS = 100
KMEANS_RUNS = 10


def kmeans(clusters, points, weights):
    process = KMeans(n_clusters=clusters, init='random', n_init=KMEANS_RUNS, max_iter=KMEANS_ITERATIONS)
    process.fit(points, sample_weight=weights)
    result = process.predict(points, sample_weight=weights)
    return result


def expand_labels(image_as_ints, unique_ints, labels, shape):
    # Each unique color is represented as an int
    # and is used as an index to its own label.
    lookup = np.empty((256 ** 3), dtype=np.uint8)
    for i in range(len(unique_ints)):
        index = unique_ints[i]
        label = labels[i]
        lookup[index] = label

    # Each image pixel is also represented as an int.
    # Its value is used to index the lookup table,
    # where each unique color stores its label.
    height, width, _ = shape
    expanded = np.array([lookup[p] for p in image_as_ints])
    expanded = expanded.reshape((height, width))
    return expanded
