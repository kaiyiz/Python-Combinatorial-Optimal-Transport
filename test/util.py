import numpy as np
from scipy.spatial.distance import cdist

def get_synthetic_input_transport(n = 1000, norm_cost=1, metric = 'sqeuclidean', seed = 0):
    """
    This function creates synthetic experiment data by randomly generating points in a 2d unit square.
    """
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, metric)
    C = cost.max()
    if norm_cost:
        cost = cost/C
    DA = np.random.rand(n)
    SB = np.random.rand(n)
    DA[DA==0] = 1e-6
    SB[SB==0] = 1e-6
    DA = DA/np.sum(DA)
    SB = SB/np.sum(SB)
    return DA, SB, cost

def get_synthetic_input_assignment(n = 1000, norm_cost=1, metric = 'sqeuclidean', seed = 0):
    """
    This function creates synthetic experiment data by randomly generating points in a 2d unit square.
    """
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, metric)
    C = cost.max()
    if norm_cost:
        cost = cost/C
    DA = np.ones(n)/n
    SB = np.ones(n)/n
    return DA, SB, cost