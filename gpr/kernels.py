import jax.numpy as np


def RBFKernel(X, Z, params: dict):
    """
    RBF Kernel - K(X,Z)
    :param X: Array of data points (x_i).
    :param Z: Array of data points (z_i.
    :param params: Parameters of the RBF Kernel.
    :return: RBFKernel of X and Z.
    """

    assert "length" in params
    assert "var" in params

    lengthscale = params["length"]
    var = params["var"]

    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = np.sum(
        np.multiply(scaled_X, scaled_X), 1, keepdims=True
    )  # sum col of the matrix
    Z2 = np.sum(np.multiply(scaled_Z, scaled_Z), 1, keepdims=True)
    XZ = np.matmul(scaled_X, scaled_Z.T)

    K0 = X2 + Z2.T - 2 * XZ

    K = var * np.exp(-0.5 * K0)
    return K
