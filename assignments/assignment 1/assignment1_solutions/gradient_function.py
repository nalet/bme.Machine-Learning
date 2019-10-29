from tanh import tanh
import numpy as np


def gradient_function(theta, X, y):
    """
    Compute gradient for regression w.r.t. to the parameters theta.

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        grad: The gradient of the cost w.r.t. theta

    """

    grad = None
    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of theta.              #
    # Compute the partial derivatives and set grad to the partial         #
    # derivatives of the cost w.r.t. each parameter in theta              #
    #                                                                     #
    #######################################################################
    
    if(len(X.shape) != 2):
        h_theta = tanh(np.dot(X,theta))
        grad = np.sum( (1/X.shape[0]) * (h_theta - y) * (1 - h_theta**2) * X )
    else:
        h_theta = tanh(np.matmul(X,theta))
        grad = (1/X.shape[0]) * 2 * np.dot((h_theta - y) * (1 - h_theta**2),X )

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad