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

    # y = y[:, np.newaxis]

    # yp1 = np.where(y == 1,1,0)
    # ym1 = np.where(y == -1,1,0)

    
    # # phi = (1/X.shape[0]) * np.sum(yp1)
    
    # # mu_0 = np.sum(ym1 * X) / np.sum(ym1)
    # # mu_1 = np.sum(yp1 * X) / np.sum(yp1)
    
    # # matrix = (X - (ym1 * mu_0 + yp1 * mu_1))
    
    # # sigma = np.matmul(matrix.T, matrix) / y.shape[0]

    yp1 = np.where(y == 1,1,0)
    ym1 = np.where(y == -1,1,0)

    phi = (1/X.shape[0]) * np.sum(yp1)
    mu_0 = np.dot(ym1,X) / np.sum(ym1)
    mu_1 = np.dot(yp1,X) / np.sum(yp1)
    
    mu_y = np.outer(ym1,mu_0) + np.outer(yp1,mu_1)
    sigma = 1.0/X.shape[0]*np.dot((X-mu_y).T,X-mu_y)

    # # print(mu_0)

    # phi = (1/X.shape[0]) * np.sum(yp1)
    # mu_0 = np.dot(ym1.T,X) / np.sum(ym1.T)
    # mu_1 = np.dot(yp1.T,X) / np.sum(yp1.T)
    
    # mu_y = np.outer(ym1.T,mu_0) + np.outer(yp1.T,mu_1)
    # sigma = (1.0/X.shape[0])*np.dot((X-mu_y).T,X-mu_y)
    # print(sigma.shape)

    # indicator_1 = [[1 if y[i]==1 else 0] for i in range(X.shape[0])]
    # indicator_0 = [[1 if y[i]==-1 else 0] for i in range(X.shape[0])]

    # # Scalar
    # phi = 1.0/X.shape[0] * np.sum(indicator_1)
    
    # # Creates 1xn array
    # mu_0 = np.dot(np.transpose(indicator_0),X)/np.sum(indicator_0)
    
    # # Creates 1xn array
    # mu_1 = np.dot(np.transpose(indicator_1), X)/np.sum(indicator_1)
    
    # mu = X - (np.outer(y,mu_1) + np.outer((1-y), mu_0))
    # mu_transposed = np.transpose(mu)
    
    # dot_product = np.dot(mu, mu_transposed)
    
    # sigma = dot_product.T

    # print(X.shape)
    # print(np.linalg.det(sigma))


    # y[y==-1]=0
    # print(X.shape)
    # phi = np.mean(y)
    # mu_0 = np.dot(1-y,X) / np.sum(1-y)
    # mu_1 = np.dot(y,X) / np.sum(y)
    
    # mu_y = np.outer(1-y,mu_0) + np.outer(y,mu_1)
    # sigma = 1.0/X.shape[0]*np.dot((X-mu_y).T,X-mu_y)

    # print(sigma)

    # # Computation of phi (the mean of the bernoulli distribution)
    # y[y==-1]=0
    # phi = np.mean(y)

    # # Computation of the mean vector mu_0 and mu_1
    # #np.sum(..., axis=0) is the collumn-sum...so the means for every feature over the whole batch is calculated
    # mu_0 = np.divide(np.sum(X[y==0],axis=0),np.sum((y==0)*1.0))
    # mu_1 = np.divide(np.sum(X[y==1],axis=0),np.sum((y==1)*1.0))

    # # Compute the mean vector mu_y depending on the vectors mu_y_0asMAt and mu_y_1asMAt 
    # # by computing a matrix of size "X.shape[0] x X.shape[1]"
    # # some notes to np.array: -np.array([1,2,3]) creates a collumnvector [1,2,3]^T.
    # #                         -mu_0 and mu_1 are collumnvectors as [1,2,3] in the above argument in np.array([1,2,3])
    # #                         -np.array(mu_0) creates a collumn-vector mu_0^T
    # #                         -np.array([[1,2,3]]) creates a rowvoctor [1,2,3].
    # #                         -mu_0 and mu_1 are collumnvectors as [1,2,3] in the above argument in np.array([[1,2,3]])
    # #                         -np.array([mu_0]) creates a collumn-vector mu_0
    # mu_y_0asMAt = np.multiply(np.array([mu_0]*X.shape[0]),np.transpose(1-np.array([y]*X.shape[1])))
    # mu_y_1asMAt = np.multiply(np.array([mu_1]*X.shape[0]),np.transpose(np.array([y]*X.shape[1])))
    # mu_y = mu_y_0asMAt+mu_y_1asMAt

    # # Computation of the covariance matrix sigma
    # sigma = 1.0/X.shape[0]*np.matmul(np.transpose(np.subtract(X,mu_y)),np.subtract(X,mu_y))


    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    # Compute theta from the results of GDA
    sigma_inv = np.linalg.inv(sigma)
    quad_form = lambda A, x: np.dot(x.T, np.dot(A, x))
    b = 0.5*quad_form(sigma_inv, mu_0) - 0.5*quad_form(sigma_inv, mu_1) + np.log(phi/(1-phi))
    w = np.dot((mu_1-mu_0), sigma_inv)
    print(np.array([b]).shape,w.shape)
    theta = np.concatenate([[b], w])
    exec_time = time.time() - start

    # Add the bias to X and compute the cost
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    loss = cost_function(theta, X, y)

    print('Iter 1/1: cost = {}  ({}s)'.format(loss, exec_time))

    return theta, None
