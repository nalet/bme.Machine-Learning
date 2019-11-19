import numpy as np


def svm_gradient(w, b, x, y, C):
    """
    Compute gradient for SVM w.r.t. to the parameters w and b on a mini-batch (x, y)

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        x: A mini-batch of training example [k, num_features]
        y: Labels corresponding to x of size [k]

    Returns:
        grad_w: The gradient of the SVM objective w.r.t. w of shape [num_features]
        grad_v: The gradient of the SVM objective w.r.t. b of shape [1]

    """

    grad_w = 0
    grad_b = 0


    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of w and b.            #
    # Compute the partial derivatives and set grad_w and grad_b to the    #
    # partial derivatives of the cost w.r.t. both parameters              #
    #                                                                     #
    #######################################################################

    grad = np.where(y * (x @ w + b) > 1, np.zeros(y.shape[0]), -y)

    grad_w = ((x.T * grad).T + w / C).sum(axis=0) / x.shape[0]
    grad_b = grad.sum() /x.shape[0]

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad_w, grad_b
