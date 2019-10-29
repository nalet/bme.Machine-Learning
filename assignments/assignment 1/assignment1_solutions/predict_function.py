from tanh import tanh
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.

    """

    preds = None
    accuracy = None
    #######################################################################
    # TODO:                                                               #
    # Compute predictions on X using the parameters theta.                #
    # If y is provided compute the accuracy of the classifier as well.    #
    #                                                                     #
    #######################################################################
    
    preds = np.where(tanh(np.dot(theta, X.T)) > 0, 1, -1)
    
    if (y.any != None):
        accuracy = 1.0/y.shape[0] * np.sum(np.where(y - preds == 0, 1, 0)) # to check if similar
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return preds, accuracy