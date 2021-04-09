from numba import njit
import numpy as np


@njit
def lm_l1_sum1(x, C):
    """ Computes
        Argmin_{1.T y = 1, ||y||_1 <= C} < x ; y >
    """
    arg_min = -1
    arg_max = -1
    min_value = np.inf
    max_value = -np.inf
    for (i, v) in enumerate(x):
        if v > max_value:
            max_value = v
            arg_max = i
        if v < min_value:
            min_value = v
            arg_min = i

    if arg_max != arg_min:
        return (arg_min, arg_max)
    else:
        return (arg_min, arg_min+1)


@njit
def away_l1_sum1(x, A, C):
    m = -np.inf
    n = A.shape[0]
    idx = (-1, -1)
    for i in range(n):
        for j in range(n):
            if A[i, j] > 1e-12:
                dot_prod = (C+1)/2*x[i] + (1-C)/2*x[j]
                if dot_prod > m:
                    idx = (i, j)
                    m = dot_prod
    return idx


@njit
def FW(RTR, c, tol, C, max_iter=1000, verbose=False):
    for it in range(max_iter):
        grad_c = RTR @ c
        # d = lm_l1_sum1(grad_c, C)-c
        d = - c.copy()
        idx = lm_l1_sum1(grad_c, C)
        d[idx[0]] += (C+1)/2
        d[idx[1]] += (1-C)/2
        xRTRd = ((C+1)**2/4*RTR[idx[0], idx[0]] +
                 (C-1)**2/4*RTR[idx[1], idx[1]] +
                 (1 - C**2)/2*RTR[idx[0], idx[1]])
        xRTRd -= (C+1)/2*grad_c[idx[0]] + (1-C)/2*grad_c[idx[1]]
        fw_gap = - grad_c.dot(d)
        if fw_gap <= tol:
            return (c, fw_gap, np.sum(np.abs(c)) >= 0.1*C)
        # t = min(max(np.exp(np.log(fw_gap)-np.log(d.dot(RTR @ d))), 0), 1)
        t = min(max(np.exp(np.log(fw_gap) -
                           np.log(xRTRd + fw_gap)), 0), 1)
        # t = 1/(2+it)
        c += t * d
    if verbose:
        print("max iterations number reached")
    return (c, fw_gap, np.sum(np.abs(c)) >= 0.1*C)


def AFW(RTR, tol, C, max_iter=1000, verbose=False):
    n = RTR.shape[0]
    vertex_set = np.zeros((n, n))
    c = np.zeros(n)
    c[0] = (C+1)/2
    c[1] = (1-C)/2
    vertex_set[0, 1] = 1
    for it in range(max_iter):
        grad_c = RTR @ c
        d_fw = - c.copy()
        idx_fw = lm_l1_sum1(grad_c, C)
        d_fw[idx_fw[0]] += (C+1)/2
        d_fw[idx_fw[1]] += (1-C)/2
        fw_gap = - grad_c.dot(d_fw)
        if fw_gap <= tol:
            if verbose:
                print("success")
            return (c, fw_gap)
        idx_away = away_l1_sum1(grad_c, vertex_set, C)
        d_a = c.copy()
        d_a[idx_away[0]] -= (C+1)/2
        d_a[idx_away[1]] -= (1-C)/2
        if - grad_c.dot(d_a) <= fw_gap:
            d = d_fw
            gamma = 1
            idx = idx_fw
            away = -1
        else:
            d = d_a
            gamma = (vertex_set[idx_away[0], idx_away[1]] /
                     (1-vertex_set[idx_away[0], idx_away[1]]))
            idx = idx_away
            away = 1
        t = min(max(np.exp(np.log(-grad_c.dot(d))-np.log(d.dot(RTR @ d))), 0),
                gamma)
        # t = 1/(2+it)
        c += t * d
        vertex_set *= (1 + away*t)
        vertex_set[idx[0], idx[1]] -= away*t
    return (c, fw_gap)
