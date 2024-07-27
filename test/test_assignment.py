# test the assignment_torch function by comparing with computed results from test_data
import numpy as np
import torch
from cot.assignment import assignment_torch

def test_assignment_torch_precomputed():
# load the test data
    device = 'cpu'
    cost = np.loadtxt('./test/test_data/assignment_cost_test.csv', delimiter=',')
    M_res = np.loadtxt('./test/test_data/assignment_M_res_test.csv', delimiter=',')
    total_cost_res = np.loadtxt('./test/test_data/assignment_total_cost_test.csv', delimiter=',')
    delta = np.loadtxt('./test/test_data/assignment_delta_test.csv', delimiter=',')
    C = cost.max()
    cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
    C_tensor = torch.tensor([C], device=device, requires_grad=False)
    delta_tensor = torch.tensor(delta, device=device, requires_grad=False)
    M, yA, yB, total_cost, iteration = assignment_torch(cost_tensor, C_tensor, delta_tensor, device=device)
    assert np.allclose(M_res, M.numpy().astype(int), atol=1e-5)
    assert np.allclose(total_cost_res, total_cost.numpy(), atol=1e-5)