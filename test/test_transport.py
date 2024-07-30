import numpy as np
import torch
from cot.transport import transport_torch
from cot.LMR import transport_lmr, OT_Profile, RPW

def test_transport_torch():
    device = 'cpu'
    DA = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[0]
    SB = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[1]
    cost = np.loadtxt('./test/test_data/transport_cost_test.csv', delimiter=',')
    T_res = np.loadtxt('./test/test_data/transport_T_res_test.csv', delimiter=',')
    total_cost_res = np.loadtxt('./test/test_data/transport_total_cost_test.csv', delimiter=',')
    delta = np.loadtxt('./test/test_data/transport_delta_test.csv', delimiter=',')
    C = cost.max()
    cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
    DA_tensor = torch.tensor(DA, device=device, requires_grad=False)
    SB_tensor = torch.tensor(SB, device=device, requires_grad=False)
    delta_tensor = torch.tensor(delta, device=device, requires_grad=False)
    T, yA, yB, total_cost = transport_torch(DA_tensor, SB_tensor, cost_tensor, delta_tensor, device=device)
    assert np.allclose(T_res, T.numpy(), atol=1e-5)
    assert np.allclose(total_cost_res, total_cost.numpy(), atol=1e-5)
    assert np.allclose(np.sum(T.numpy(), axis=0), DA, atol=1e-5)
    assert np.allclose(np.sum(T.numpy(), axis=1), SB, atol=1e-5)

def test_LMR_precomputed():
    DA = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[0]
    SB = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[1]
    cost = np.loadtxt('./test/test_data/transport_cost_test.csv', delimiter=',')
    ot_cost_lmr_res = np.loadtxt('./test/test_data/ot_cost_lmr.csv', delimiter=',')
    delta = np.loadtxt('./test/test_data/transport_delta_test.csv', delimiter=',')
    ot_cost_lmr = transport_lmr(DA, SB, cost, delta)
    assert np.allclose(ot_cost_lmr_res, ot_cost_lmr, atol=1e-5)

def test_OT_Profile_precomputed():
    DA = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[0]
    SB = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[1]
    cost = np.loadtxt('./test/test_data/transport_cost_test.csv', delimiter=',')
    ot_profile_res = np.loadtxt('./test/test_data/ot_profile.csv', delimiter=',')
    delta = np.loadtxt('./test/test_data/transport_delta_test.csv', delimiter=',')
    ot_profile = OT_Profile(DA, SB, cost, delta)
    assert np.allclose(ot_profile_res, ot_profile, atol=1e-5)

def test_RPW_precomputed():
    DA = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[0]
    SB = np.loadtxt('./test/test_data/transport_A_B_test.csv', delimiter=',')[1]
    cost = np.loadtxt('./test/test_data/transport_cost_test.csv', delimiter=',')
    rpw_dist_res = np.loadtxt('./test/test_data/rpw_dist.csv', delimiter=',')
    delta = np.loadtxt('./test/test_data/transport_delta_test.csv', delimiter=',')
    rpw_dist = RPW(DA, SB, cost, delta)
    assert np.allclose(rpw_dist_res, rpw_dist, atol=1e-5)