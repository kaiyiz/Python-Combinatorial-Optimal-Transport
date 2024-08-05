import numpy as np
import jpype
import jpype.imports
from jpype.types import *
import pkg_resources

jarLocation = pkg_resources.resource_filename('cot', 'optimaltransport.jar')
try:
    jpype.startJVM("-Xmx128g", classpath=[jarLocation])
except OSError as e:
    if 'JVM is already started' in str(e):
        pass
    else:
        raise e
# jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping


def transport_lmr(DA, SB, C, eps):
    """
    This function sloves the additive approximation of optimal transport problem between two discrete distributions 
    and returns the approximated cost based on the graph-based additive approximation algorithm [1]_ for OT.

    Parameters
    ----------
    DA : numpy array, shape (n,)
        A n by 1 tensor, the weight of samples from the demand distribution (type a), each DA[i] represents the mass of demand on i-th type a vertex. The sum of DA should equal to 1.
    SB : numpy array, shape (n,)
        A n by 1 tensor, the weight of samples from the source distribution (type b), each SB[i] represents the mass of supply on i-th type b vertex. The sum of SB should equal to 1.
    C : numpy array, shape (n, n)
        A n by n cost matrix, each C(i,j) represents the cost between i-th type b and j-th type a vertex.
    eps : float
        The additive error of optimal transport distance, the value of :math:`\epsilon` in the LMR paper [1]_ .
    
    Returns
    -------
    ot_cost : float

    References
    ----------
    .. [1] Lahn, Nathaniel, Deepika Mulchandani, and Sharath Raghvendra. A graph theoretic additive approximation of optimal transport. 
        Advances in Neural Information Processing Systems (NeurIPS) 32, 2019
    """
    nz = len(DA)
    gtSolver = Mapping(nz, list(DA), list(SB), C, eps)
    ot_cost = gtSolver.getTotalCost()
    return ot_cost


def ot_profile(DA, SB, C, eps, p=1):
    """
    This function computes the approximated Optimal Transport profile (OT-Profile) [3]_ between two discrete distributions by leveraging the LMR algorithm [1]_.
    The OT-profile is a function of the cost of α-optimal paritial transport cost as the transported mass α the variables. 
    This returns the OT-profile as a 2 by k array, where the first row represents the amount of transported mass and the second row represents the corresponding cost of optimal partial transport.

    Parameters
    ----------
    DA : numpy array
        A n by 1 array, each DA(i) represent the mass of demand on ith type a vertex. The sum of DA should equal to 1.
    SB : numpy array
        A n by 1 array, each SB(i) represent the mass of supply on ith type b vertex. The sum of SB should equal to 1.
    C : numpy array
        A n by n cost matrix, each C(i,j) represents the cost between ith type b and jth type a vertex.
    eps : float
        The additive error of OT-Profile, the value of :math:`\epsilon` in paper [3]_.
    
    Returns
    -------
    ot_profile : 2 by k numpy array
        A 2 by k array, first row represent the amount of transported mass, second row represent the corresponding cost of optimal partial transport.

    References
    ----------
    .. [1] Lahn, Nathaniel, Deepika Mulchandani, and Sharath Raghvendra. A graph theoretic additive approximation of optimal transport. 
        Advances in Neural Information Processing Systems (NeurIPS) 32, 2019
    .. [3] Phatak, Abhijeet, et al. Computing all optimal partial transports. International Conference on Learning Representations (ICLR). 2023.
    """
    # eps : acceptable additive error
    # q_idx : index to get returned values
    nz = len(DA)
    C = C**p
    alphaa = 4.0*np.max(C)/eps
    gtSolver = Mapping(nz, list(DA), list(SB), C, eps)
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

def rpw(DA=None, SB=None, dist=None, eps=0.1, k=1, p=1):
    """
    Computes the approximated Robust Partial p-Wasserstein (RPW) distance [4]_ between two discrete distributions.
    The RPW metric provides a robust distance between distributions by considering partial optimal transport plan.

    Parameters
    ----------
    DA : numpy array, shape (n,)
        A n by 1 array, each DA(i) represent the mass of demand on ith type a vertex. The sum of DA should equal to 1.
    SB : numpy array, shape (n,)
        A n by 1 array, each SB(i) represent the mass of supply on ith type b vertex. The sum of SB should equal to 1.
    dist : numpy array, shape (n, n)
        A n by n cost matrix, each C(i,j) represents the cost between ith type b and jth type a vertex.
    eps : float, default=0.1
        The additive error of OT-Profile, the value of :math:`\epsilon` in paper [4]_.
    k : int, default=1
        Scaling factor in the RPW distance.
    p : int, default=1
        The order of the Wasserstein distance.

    Returns
    -------
    pk_rpw : float
        The computed approximated RPW distance between the two distributions.

    References
    ----------
    .. [4] Raghvendra, Sharath, Pouyan Shirzadian, and Kaiyi Zhang. "A New Robust Partial p-Wasserstein-Based Metric for Comparing Distributions." Forty-first International Conference on Machine Learning.
    """

    nz = len(DA)
    dist = dist**p
    alphaa = 4.0*np.max(dist)/eps
    gtSolver = Mapping(nz, list(DA), list(SB), dist, eps)
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
    pk_rpw = 1 - alpha
    return pk_rpw

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