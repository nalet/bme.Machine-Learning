from numpy import exp
from numpy import tanh as np_tanh


def tanh(z):
    """
    Computes the tanh of z element-wise.

    Args:
        z: An np.array of arbitrary shape

    Returns:
        g: An np.array of the same shape as z

    """

    g = None
    #######################################################################
    # TODO:                                                               #
    # Compute and return the tanh of z in g                            #
    #######################################################################

    #g = (exp(2*z)-1) / (exp(2*z)+1)
    g = np_tanh(z)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return g
