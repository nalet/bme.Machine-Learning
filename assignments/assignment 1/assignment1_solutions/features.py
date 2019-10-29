from skimage.feature import hog
import numpy as np

def hog_features(X):
    """
    Extract HOG features from input images

    Args:
        X: Data matrix of shape [num_train, 577]

    Returns:
        hogs: Extracted hog features

    """
    
    hog_list = []
    
    for i in range(X.shape[0]):
        #######################################################################
        # TODO:                                                               #
        # Extract HOG features from each image and append them to the         #
        # hog_list                                                            #
        #                                                                     #
        # Hint: Make sure that you reshape the imput features to size (24,24) #
        #                                                                     #
        #######################################################################

        img = np.reshape(X[i,1:X.shape[1]],[24,24], order='F')
        feature_vector, hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                    block_norm = 'L2-Hys', visualise=True, transform_sqrt=False,
                    feature_vector=True)
        hog_list.append(np.concatenate((np.ones((1)), feature_vector), axis=0))

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        
    hogs = np.stack(hog_list,axis=0)
    hogs = np.concatenate((np.ones((X.shape[0], 1)), np.reshape(hogs,(X.shape[0],-1))), axis=1)

    return hogs