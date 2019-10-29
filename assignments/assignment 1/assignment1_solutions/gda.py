from cost_function import cost_function
import numpy as np
import time


def gda(X, y):
    """
    Perform Gaussian Discriminant Analysis.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]

    Returns:
        theta: The value of the parameters after logistic regression

    """

    theta = None
    phi = None
    mu_0 = None
    mu_1 = None
    sigma = None

    X = X[:, 1:]    # Note: We remove the bias term!
    start = time.time()

    #######################################################################
    # TODO:                                                               #
    # Perform GDA:                                                        #
    #   - Compute the values for phi, mu_0, mu_1 and sigma                #
    #                                                                     #
    #######################################################################

    yC = np.where(y == 0, 1, 0)
    mu_1 = 1.0 / np.sum(y) * X.T.dot(y)
    mu_0 = 1.0 / np.sum(yC) * X.T.dot(yC)
    sigma = np.zeros((X.shape[1], X.shape[1]))
    
    for i in range(y.shape[0]):
        if(y[i] == 0):
            sigma += (X[i, :] - mu_0)[:, None].dot((X[i, :] - mu_0)[:, None].T)
        else:
            sigma += (X[i, :] - mu_1)[:, None].dot((X[i, :] - mu_1)[:, None].T)
    sigma = 1.0 / X.shape[0] * sigma

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    # Compute theta from the results of GDA
    sigma_inv = np.linalg.inv(sigma)
    quad_form = lambda A, x: np.dot(x.T, np.dot(A, x))
    b = 0.5*quad_form(sigma_inv, mu_0) - 0.5*quad_form(sigma_inv, mu_1) + np.log(phi/(1-phi))
    w = np.dot((mu_1-mu_0), sigma_inv)
    theta = np.concatenate([[b], w])
    exec_time = time.time() - start

    # Add the bias to X and compute the cost
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    loss = cost_function(theta, X, y)

    print('Iter 1/1: cost = {}  ({}s)'.format(loss, exec_time))

    return theta, None
