import numpy as np
from sklearn.mixture import GaussianMixture

import time


def em_segmentation(img, k, max_iter=20):
    """
    Learns a MoG model using the EM-algorithm for image-segmentation.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of gaussians to be used

    Returns:
        label_img: A matrix of labels indicating the gaussian of size [h, w]

    """

    label_img = None

    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #
    #      generate the label-image                                       #
    #######################################################################
    
    h, w, c = img.shape
    xv, yv = np.meshgrid(np.arange(0, w), np.arange(0, h))

    coordinates = np.stack([yv, xv], axis=2)
    img = np.concatenate([img, coordinates], axis=2)
    img = np.reshape(img, (h * w, c + 2)) # +2 to match shape

    mog = GaussianMixture(n_components=k, max_iter=max_iter)
    mog = mog.fit(img)
    label_img = mog.predict(img)

    means_ = np.delete(mog.means_, [3, 4], axis=1).astype(int)
    _t = np.take(means_, label_img, axis=0)
    label_img = np.reshape(_t, (h, w, c))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return label_img
