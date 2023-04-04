import faiss
import numpy as np

KMEANS_ITERATIONS = 100
KMEANS_RUNS = 10


def kmeans(clusters, points, weights):
    process = faiss.Kmeans(d=points.shape[1], k=clusters, niter=KMEANS_ITERATIONS, nredo=KMEANS_RUNS)
    process.train(points, weights)
    result = process.index.search(points, 1)[1]
    result = result.ravel()
    return result


def expand_labels(image_as_ints, unique_ints, labels, shape):
    # Each unique color is represented as an int
    # and is used as an index to its own label.
    lookup = np.empty((256 ** 3), dtype=np.int32)
    np.put(lookup, unique_ints, labels)

    # Each image pixel is also represented as an int.
    # Its value is used to index the lookup table,
    # where each unique color stores its label.
    height, width, _ = shape
    expanded = np.take(lookup, image_as_ints)
    expanded = expanded.reshape((height, width))
    return expanded
