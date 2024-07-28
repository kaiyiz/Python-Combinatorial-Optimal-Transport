# generate synthetic results for testing the code. The initial code is correct from experimental implementation, reference https://github.com/kaiyiz/Combinatorial-Parallel-OT
import numpy as np
import torch
from util import get_synthetic_input_transport, get_synthetic_input_assignment
from cot.assignment import assignment_torch, assignment
from cot.transport import transport_torch

n = 100
norm_cost = 1
delta = 0.01
metric = 'sqeuclidean'
seed = 0
device = 'cpu'

DA, SB, cost = get_synthetic_input_transport(n, norm_cost, metric, seed)
cost_max = cost.max()
cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
DA_tensor = torch.tensor(DA, device=device, requires_grad=False)
SB_tensor = torch.tensor(SB, device=device, requires_grad=False)
delta_tensor = torch.tensor([delta], device=device, requires_grad=False)
T, yA, yB, total_cost = transport_torch(DA_tensor, SB_tensor, cost_tensor, delta_tensor, device=device)
transport_A_B = []
transport_A_B.append(DA)
transport_A_B.append(SB)
np.savetxt('./test_data/transport_A_B_test.csv', transport_A_B, delimiter=',')
np.savetxt('./test_data/transport_cost_test.csv', cost, delimiter=',')
np.savetxt('./test_data/transport_T_res_test.csv', T.numpy(), delimiter=',')
np.savetxt('./test_data/transport_total_cost_test.csv', np.array([total_cost.numpy()]), delimiter=',')
np.savetxt('./test_data/transport_delta_test.csv', np.array([delta]), delimiter=',')

DA, SB, cost = get_synthetic_input_assignment(n, norm_cost, metric, seed)
cost_max = cost.max()
cost_tensor = torch.tensor(cost, device=device, requires_grad=False)
cost_max_tensor = torch.tensor([cost_max], device=device, requires_grad=False)
M, yA, yB, total_cost_torch_assignment = assignment_torch(cost_tensor, cost_max_tensor, delta_tensor, device=device)
np.savetxt('./test_data/asignment_cost_test.csv', cost, delimiter=',')
np.savetxt('./test_data/assignment_M_res_test.csv', M.numpy().astype(int), delimiter=',', fmt='%d')
np.savetxt('./test_data/assignment_total_cost_test.csv', np.array([total_cost_torch_assignment.numpy()]), delimiter=',')
np.savetxt('./test_data/assignment_delta_test.csv', np.array([delta]), delimiter=',')

# generate synthetic data for assignment (numpy)
M, yA, yB, total_cost_assignment = assignment(cost, cost_max, delta)
np.savetxt('./test_data/assignment_M_res_test_numpy.csv', M, delimiter=',', fmt='%d')
np.savetxt('./test_data/assignment_total_cost_test_numpy.csv', np.array([total_cost_assignment]), delimiter=',')
np.savetxt('./test_data/assignment_delta_test_numpy.csv', np.array([delta]), delimiter=',')