from numpy.linalg import norm
import numpy as np


def power_method(A, max_iter=1000, tol=1e-6):
    """Compute spectral norm of A using the power method.

    Parameters
    ----------
    A : ndarray, shape (n, d)
        Matrix whose spectral norm is computed.
    max_iter : int, default=1000.
        Maximum number of power iterations.
    tol : float, default=1e-6
        Tolerance : if the estimated norm changes less than tol, the algorithm
        stops.

    Returns
    -------
    spec_norm : float
        The estimated spectral norm.
    """
    n, d = A.shape
    np.random.seed(1)
    u = np.random.randn(n)
    v = np.random.randn(d)
    spec_norm = 0
    for _ in range(max_iter):
        spec_norm_old = spec_norm
        u = A @ v
        u /= norm(u)
        v = A.T @ u
        spec_norm = norm(v)
        v /= spec_norm
        if np.abs(spec_norm - spec_norm_old) < tol:
            break
    return spec_norm
