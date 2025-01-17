from sklearn.cluster import KMeans
import numpy as np
from time import time
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin

def kmeans_colors(img, k, max_iter=100):
    """
    Performs k-means clusering on the pixel values of an image.
    Used for color-quantization/compression.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of color clusters to be computed

    Returns:
        img_cl:  The color quantized image of shape [h, w, 3]

    """

    img_cl = None

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the pixel values of the image img.     #
    #######################################################################
    
    kmeans = KMeans(n_clusters = k, random_state = 0)
    kmeans = kmeans.fit(img.reshape((-1, 3)))
    cluster_centers = kmeans.cluster_centers_[kmeans.labels_, :]
    img_cl = np.reshape(cluster_centers, img.shape)
    img_cl = img_cl.astype(int) #for imshow

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return img_cl
