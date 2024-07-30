import numpy as np
import jpype
import jpype.imports
from jpype.types import *
import pkg_resources

jarLocation = pkg_resources.resource_filename('cot', 'optimaltransport.jar')
jpype.startJVM("-Xmx128g", classpath=[jarLocation])
# jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

def transport_lmr(DA, SB, C, delta):
    """
    This function computes an additive approximation of 1-wasserstein distance between two discrete distributions [1]

    Parameters
    ----------
    DA : ndarray
        A n by 1 array, each DA(i) represent the mass of demand on ith type a vertex. The sum of DA should equal to 1.
    SB : ndarray
        A n by 1 array, each SB(i) represent the mass of supply on ith type b vertex. The sum of SB should equal to 1.
    C : tensor
        A n by n cost matrix, each i and j represent the cost between ith type b and jth type a vertex.
    delta : tensor
        The scaling factor (scalar) of cost metric. The value of epsilon in paper. 
    
    Returns
    -------

    References
    ----------
    .. [1] Lahn, Nathaniel, Deepika Mulchandani, and Sharath Raghvendra. A graph theoretic additive approximation of optimal transport. 
        Advances in Neural Information Processing Systems (NeurIPS) 32, 2019
    """
    nz = len(DA)
    gtSolver = Mapping(nz, list(DA), list(SB), C, delta)
    ot_cost = gtSolver.getTotalCost()
    return ot_cost

def OT_Profile(DA, SB, C, delta, p=1):
    """
    This function computes the OT-profile between two discrete distributions [3]

    Parameters
    ----------
    DA : ndarray
        A n by 1 array, each DA(i) represent the mass of demand on ith type a vertex. The sum of DA should equal to 1.
    SB : ndarray
        A n by 1 array, each SB(i) represent the mass of supply on ith type b vertex. The sum of SB should equal to 1.
    C : tensor
        A n by n cost matrix, each i and j represent the cost between ith type b and jth type a vertex.
    delta : tensor
        The scaling factor (scalar) of cost metric. The value of epsilon in paper. 
    
    Returns
    -------
    OT_profile : ndarray
        A 2 by k array, first row represent the amount of transported flow, second row represent the corresponding optimal partial transport.

    References
    ----------
    .. [3] Phatak, Abhijeet, et al. Computing all optimal partial transports. International Conference on Learning Representations (ICLR). 2023.
    """
    # delta : acceptable additive error
    # q_idx : index to get returned values
    nz = len(DA)
    C = C**p
    alphaa = 4.0*np.max(C)/delta
    gtSolver = Mapping(nz, list(DA), list(SB), C, delta)
    APinfo = np.array(gtSolver.getAPinfo()) # augmenting path information
    # 0->Number of iterations(phase id)
    # 1->Length of augmenting path(AP)
    # 2->Flow of AP (transported mass)
    # 3->AP transportation cost
    # 4->Dual weight of the AP beginning vertex (AP net cost we actually use)(matching cost is the cumulative sum)(matching cost 1st derivative)
    # 5->Vertex index at the beginning of AP
    # 6->lt value of current phase((matching cost 2nd derivative = lt/number of pathes in phase)

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]

    cost_AP = APinfo_cleaned[:,4] * APinfo_cleaned[:,2]
    cumCost = (np.cumsum(cost_AP)/(alphaa*alphaa*nz))**(1/p)

    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))
    totalFlow = cumFlow[-1]
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    OT_profile = np.vstack((flowProgress, cumCost))
    return OT_profile

def RPW(X=None, Y=None, dist=None, delta=0.1, k=1, p=1):
    """

    Args:
        X ([type], optional): Defaults to None.
        Y ([type], optional): Defaults to None.
        dist ([type], optional): Defaults to None.
        delta (float, optional): Defaults to 0.1.
        k (int, optional): Defaults to 1.
        p (int, optional): [Defaults to 1.

    Returns:
        RPW distance (float)
        RPW distance between two discrete distributions
    """
    # delta : acceptable additive error
    # q_idx : index to get returned values
    nz = len(X)
    dist = dist**p
    alphaa = 4.0*np.max(dist)/delta
    gtSolver = Mapping(nz, list(X), list(Y), dist, delta)
    APinfo = np.array(gtSolver.getAPinfo()) # augmenting path information
    # 0->Number of iterations(phase id)
    # 1->Length of augmenting path(AP)
    # 2->Flow of AP (transported mass)
    # 3->AP transportation cost
    # 4->Dual weight of the AP beginning vertex (AP net cost we actually use)(matching cost is the cumulative sum)(matching cost 1st derivative)
    # 5->Vertex index at the beginning of AP
    # 6->lt value of current phase((matching cost 2nd derivative = lt/number of pathes in phase)

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]

    cost_AP = APinfo_cleaned[:,4] * APinfo_cleaned[:,2]
    cumCost = (np.cumsum(cost_AP)/(alphaa*alphaa*nz))**(1/p)
    # cumCost = np.cumsum(cost_AP)/(alphaa*alphaa*nz)

    cumCost *= 1/k
    totalCost = cumCost[-1]
    if totalCost == 0:
        normalized_cumcost = (cumCost) * 0.0
    else:
        normalized_cumcost = (cumCost)/(1.0 * totalCost)

    maxdual = APinfo_cleaned[:,4]/alphaa*1/k
    final_dual = maxdual[-1]
    if final_dual == 0:
        normalized_maxdual = maxdual * 0.0
    else:
        normalized_maxdual = maxdual/final_dual

    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))
    totalFlow = cumFlow[-1]
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    d_cost = (1 - flowProgress) - cumCost
    d_ind_a = np.nonzero(d_cost<=0)[0][0]-1
    d_ind_b = d_ind_a + 1
    alpha = find_intersection_point(flowProgress[d_ind_a], d_cost[d_ind_a], flowProgress[d_ind_b], d_cost[d_ind_b])
    res = 1 - alpha
    return res

def find_intersection_point(x1, y1, x2, y2):
    # x1 < x2
    # y1 > 0
    # y2 < 0
    # y = ax + b
    # find x when y = 0
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    x = -b/a
    return x