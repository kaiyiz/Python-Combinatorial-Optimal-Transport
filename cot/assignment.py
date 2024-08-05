import numpy as np
import torch

def feasibility_check(ind_b, ind_a, yB, yA, Ma, cost):
    # check feasibility
    if yA[ind_a] + yB[ind_b] > cost[ind_b][ind_a] + 1:
                print("assertion failed")
    if Ma[ind_a] == ind_b and yA[ind_a] + yB[ind_b] != cost[ind_b][ind_a]:
                print("assertion failed in assignment")
                print(yA[ind_a] + yB[ind_b])
                print(cost[ind_b][ind_a])

def assignment_check(Ma, Mb):
    # check assignment
    if np.size(np.where(Mb == -1)) != 0:
        print("exist empty assignment")
    if np.size(np.where(Ma == -1)) != 0:
        print("exist empty assignment")
    for ind_b in range(Mb.shape[0]):
        ind_a = Mb[ind_b]
        if ind_a != -1 and ind_b != Ma[ind_a]:
            print("misassignment")
    for ind_a in range(Ma.shape[0]):
        ind_b = Ma[ind_a]
        if ind_b != -1 and ind_a != Mb[ind_b]:
            print("misassignment")

def assignment_check_torch(Ma, Mb):
    # check assignment
    if torch.size(torch.where(Mb == -1)) != 0:
        print("exist empty assignment")
    if torch.size(torch.where(Ma == -1)) != 0:
        print("exist empty assignment")
    for ind_b in range(Mb.shape[0]):
        ind_a = Mb[ind_b]
        if ind_a != -1 and ind_b != Ma[ind_a]:
            print("misassignment")
    for ind_a in range(Ma.shape[0]):
        ind_b = Ma[ind_a]
        if ind_b != -1 and ind_a != Mb[ind_b]:
            print("misassignment")

def assignment(W, C, eps, seed=0):
    """
    This function sloves an additive approximation of the assignment problem between two discrete distributions and returns assignment plan and cost.
    This function is a numpy implementation version of the parallelizable combinatorial algorithm [2]_ for assignment problem.

    Parameters
    ----------------
    W : numpy array, shape (n, n)
        A n by n cost matrix, each W(i,j) represents the cost between i-th type b and j-th type vertex from the samples of supply and demand distribution.
    C : float
        The scale of cost metric.
    eps : float
        The additive error of optimal transport distance, the value of :math:`\epsilon` in paper [2]_.
    
    Returns
    ----------------
    Mb : numpy array, shape (n,)
        A 1 by n array, each i represents the index of type a vertex (j-th column in cost matrix, i.e. W[:,j]) that is assigned with ith type b vertex.
    yA : numpy array, shape (n,)
        A 1 by n array, each i represents the final dual value of ith type a vertex.
    yB : numpy array, shape (n,)
        A 1 by n array, each i represents the final dual value of ith type b vertex.
    total_cost : float
        Total cost of the final assignment.

    References
    ----------------

    .. [2] Lahn, Nathaniel, Sharath Raghvendra, and Kaiyi Zhang. A combinatorial algori-thm for approximating the optimal transport in the parallel and mpc settings. Advances in Neural Information Processing Systems (NeurIPS) 36, 2023

    """
    n = W.shape[0]
    S = (3*W//(eps)).astype(int) 
    # cost = (3*W//(eps)).astype(int)
    yB = np.ones(n, dtype=int)
    yA = np.zeros(n, dtype=int)
    Mb = np.ones(n, dtype=int) * -1
    Ma = np.ones(n, dtype=int) * -1
    f = n
    iteration = 0

    np.random.seed(seed)

    while f > n*eps/C:
        ind_b_free = np.where(Mb==-1)
        ind_S_zero = np.where(S[ind_b_free]==0)

        # find push edges
        ind_b_tent_ind, free_S_edge_B_ind_range_lt_inclusive = np.unique(ind_S_zero[0],return_index=True) #get tentative B to push and left index of B range
        ind_b_tent = ind_b_free[0][ind_b_tent_ind]
        free_S_edge_B_ind_range_rt_exclusive = np.append(free_S_edge_B_ind_range_lt_inclusive[1:], len(ind_S_zero[0])) #right index of B range
        free_S_edge_B_ind_rand = np.random.randint(free_S_edge_B_ind_range_lt_inclusive, free_S_edge_B_ind_range_rt_exclusive) #random pick an index from each unique B range
        ind_a_tent = ind_S_zero[1][free_S_edge_B_ind_rand] #get tentative A to push
        ind_a_push, tent_ind = np.unique(ind_a_tent, return_index=True) #find exact a to push, and corresponding index 
        ind_b_push = ind_b_tent[tent_ind] #find exact b to push
        # find release edges
        ind_release = np.nonzero(Ma[ind_a_push] != -1)[0]
        edges_released = (Ma[ind_a_push][ind_release], ind_a_push[ind_release])
        # update flow
        f -= len(ind_a_push)-len(ind_release) 
        # release edges
        Mb[Ma[edges_released[1]]] = -1
        # push edges
        edges_pushed = (ind_b_push, ind_a_push)
        Ma[ind_a_push] = ind_b_push
        Mb[ind_b_push] = ind_a_push
        yA[ind_a_push] -= 1
        # find b that not able to be pushed
        ind_b_not_pushed = np.setdiff1d(ind_b_free[0], ind_b_tent)
        yB[ind_b_not_pushed] += 1
        #update slack
        S[edges_released] += 1
        S[edges_pushed] -= 1
        S[:,edges_pushed[1]] += 1
        S[ind_b_not_pushed, :] -= 1
        iteration += 1

    # for ind_b in range(n):
    #     for ind_a in range(n):
    #         feasibility_check(ind_b, ind_a, yB, yA, Ma, cost)

    ind_a = 0
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b

    # assignment_check(Ma, Mb)
    # print("assignment check passed")

    assignment_cost = 0
    for ind_b in range(n):
        assignment_cost += W[ind_b, Mb[ind_b]]
    assignment_cost = assignment_cost/n
    return Mb, yA, yB, assignment_cost

def assignment_torch(W, C, eps, device, seed=1):
    """
    This function sloves an additive approximation of the assignment problem between two discrete distributions and returns assignment plan and cost.
    This function is a torch implementation version of the parallelizable combinatorial algorithm [2]_ for assignment problem.

    Parameters
    ----------------
    W : tensor, shape (n, n)
        A n by n cost matrix, each W(i,j) represents the cost between i-th type b and j-th type vertex from the samples of supply and demand distribution.
    C : tensor
        The scale of cost metric.
    eps : tensor
        The additive error of optimal transport distance, the value of :math:`\epsilon` in paper [2]_.
    device : torch.device
        The device where the computation will be executed. (e.g. torch.device('cuda:0') for GPU)
    
    Returns
    ----------------
    Mb : tensor, shape (n,)
        A 1 by n array, each i represents the index of type a vertex (j-th column in cost matrix, i.e. W[:,j]) that is assigned with ith type b vertex.
    yA : tensor, shape (n,)
        A 1 by n array, each i represents the final dual value of ith type a vertex.
    yB : tensor, shape (n,)
        A 1 by n array, each i represents the final dual value of ith type b vertex.
    total_cost : float
        Total cost of the final assignment.

    References
    ----------------

    .. [2] Lahn, Nathaniel, Sharath Raghvendra, and Kaiyi Zhang. A combinatorial algori-thm for approximating the optimal transport in the parallel and mpc settings. Advances in Neural Information Processing Systems (NeurIPS) 36, 2023

    """
    dtyp = torch.int64
    n = W.shape[1]
    m = W.shape[0]

    S = (3*W/(eps)).type(dtyp).to(device)
    # cost = (3*W/(eps)).type(dtyp).to(device) # scaled cost for feasibility validation
    yB = torch.ones(m, device=device, dtype=dtyp, requires_grad=False)
    yA = torch.zeros(n, device=device, dtype=dtyp, requires_grad=False)
    Mb = torch.ones(m, device=device, dtype=dtyp, requires_grad=False) * -1
    Ma = torch.ones(n, device=device, dtype=dtyp, requires_grad=False) * -1

    f = n
    iteration = 0

    n = W.shape[0]
    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    m_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]

    f_threshold = n*eps/C

    torch.manual_seed(1)
    while f > f_threshold:
        ind_b_free = torch.where(Mb == m_one)
        ind_S_zero_ind = torch.where(S[ind_b_free] == zero)

        # find push edges
        ind_b_tent_ind, free_S_edge_B_ind_range_lt_inclusive = unique(ind_S_zero_ind[0], input_sorted=True)
        ind_b_tent = ind_b_free[0][ind_b_tent_ind]
        free_S_edge_B_ind_range_rt_exclusive = torch.cat((free_S_edge_B_ind_range_lt_inclusive[1:], torch.tensor(ind_S_zero_ind[0].shape, device=device, dtype=dtyp, requires_grad=False))) #right index of B range
        rand_n = torch.rand(ind_b_tent.shape[0], device=device)
        free_S_edge_B_ind_rand = free_S_edge_B_ind_range_lt_inclusive + ((free_S_edge_B_ind_range_rt_exclusive - free_S_edge_B_ind_range_lt_inclusive)*rand_n).to(dtyp)
        ind_a_tent = ind_S_zero_ind[1][free_S_edge_B_ind_rand] #get tentative A to push
        ind_a_push, tent_ind = unique(ind_a_tent, input_sorted=False) #find exact a to push, and corresponding index
        ind_b_push = ind_b_tent[tent_ind] #find exact b to push
        # find release edges
        ind_release = torch.nonzero(Ma[ind_a_push] != -1, as_tuple=True)[0]
        edges_released = (Ma[ind_a_push][ind_release], ind_a_push[ind_release])
        # update flow
        f -= len(ind_a_push)-len(ind_release) 
        # release edges
        Mb[Ma[edges_released[1]]] = m_one
        # push edges
        edges_pushed = (ind_b_push, ind_a_push)
        Ma[ind_a_push] = ind_b_push
        Mb[ind_b_push] = ind_a_push
        yA[ind_a_push] -= one
        # find b that not able to be pushed
        min_slack, _ = torch.min(S[ind_b_free[0],:], dim=1)
        min_slack_ind = torch.where(min_slack!=0)[0]
        ind_b_not_pushed = ind_b_free[0][min_slack_ind]
        yB[ind_b_not_pushed] += min_slack[min_slack_ind]
        #update slack
        S[edges_released] += one
        S[edges_pushed] -= one
        S[:,edges_pushed[1]] += one
        S[ind_b_not_pushed, :] -= min_slack[min_slack_ind][:,None]
        iteration += 1
    
    yA = yA.cpu().detach()   
    yB = yB.cpu().detach()
    Ma = Ma.cpu().detach()
    Mb = Mb.cpu().detach()
    
    ind_a = 0
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b
    
    # assignment_check(Ma, Mb) # check the validity of the assignment
    assignment_cost = torch.sum(W[torch.arange(0,n,dtype=torch.int64),Mb])
    assignment_cost = assignment_cost/n
    return Mb, yA, yB, assignment_cost

def unique(x, input_sorted = False):
    """""
    Returns the unique elements of array x, and the indices of the first occurrences of the unique values in the original array
    """""
    unique, inverse_ind, unique_count = torch.unique(x, return_inverse=True, return_counts=True)
    unique_ind = unique_count.cumsum(0)
    if not unique_ind.size()[0] == 0:
        unique_ind = torch.cat((torch.tensor([0], dtype=x.dtype, device=x.device), unique_ind[:-1]))
    if not input_sorted:
        _, sort2ori_ind = torch.sort(inverse_ind, stable=True)
        unique_ind = sort2ori_ind[unique_ind]
    return unique, unique_ind