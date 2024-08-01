# Python Combinatorial Optimal Transport (PyCoOT)

## Overview

**PyCoOT** is a Python library for optimal transport computation using combinatorial methods, offering state-of-the-art algorithms to compute additive approximations of optimal transport distances between discrete distributions. PyCoOT supports large-scale computations through GPU acceleration, making it a powerful tool for OT problems. Additionaly, our library leverages the properties of combinatorial algorithms to compute various OT related problems efficiently.

## Key Features

- **1-Wasserstein Distance Computation:** Efficient computation of additive approximations of 1-Wasserstein distances based on LMR algorithms.
- **Scalable Combinatorial Approach:** PyCoOT employs combinatorial algorithms to solve OT in its original approximate formulation without entropic regularization and provides parallel implementations including GPU acceleration.
- **Fast Assignment Problem Solution:** Faster impletation for approximate assignment problems, a special case of OT, and also supporting GPU acceleration.
- **Optimal Transport Profile:** Detailed calculation of optimal transport profiles between discrete distributions.
- **Robust Partial $p$-Wasserstein Distance:** An outlier robust metric for comparing distributions based on partial $p$-Wasserstein distance.

## Installation

```bash
pip install PyCoOT
```

## Quick Start

### 1. Computing 1-Wasserstein Distance (LMR Algorithm)

```python
import numpy as np
from cot import transport_lmr

DA = np.array([...])  # Demand array
SB = np.array([...])  # Supply array
C = np.array([...])   # Cost matrix
delta = 0.1           # Approximation error

ot_cost = transport_lmr(DA, SB, C, delta)
print(f"1-Wasserstein Distance (LMR Algorithm): {ot_cost}")
```

### 2. Computing 1-Wasserstein Distance (Push-Relabel Algorithm, Torch Implementation)

```python
import torch
from cot import transport_torch

DA = torch.tensor([...])  # Demand array
SB = torch.tensor([...])  # Supply array
C = torch.tensor([...])   # Cost matrix
delta = 0.1               # Approximation error

ot_cost = transport_torch(DA, SB, C, delta)
print(f"1-Wasserstein Distance (Push-Relabel): {ot_cost}")
```

### 3. Solving Bipartite Assignment Problem (Push-Relabel Algorithm)

```python
import numpy as np
import torch
from cot import assignment, assignment_torch

W = np.array([...])   # Cost matrix
C = np.max(W)         # Scale of cost metric
delta = 0.1           # Approximation error

# NumPy implementation
Mb, yA, yB, assignment_cost = assignment(W, C, delta)
print(f"Bipartite Assignment Cost (Push-Relabel, NumPy): {assignment_cost}")

# Torch implementation
W_torch = torch.tensor(W)
Mb, yA, yB, assignment_cost = assignment_torch(W_torch, C, delta)
print(f"Bipartite Assignment Cost (Push-Relabel, Torch): {assignment_cost}")
```

### 4. Computing Optimal Transport Profile (OT Profile)

```python
import numpy as np
from cot import OT_Profile
from matplotlib import pyplot as plt

DA = np.array([...])
SB = np.array([...])
C = np.array([...])
delta = 0.1

ot_profile = OT_Profile(DA, SB, C, delta)
print(f"Optimal Transport Profile: {ot_profile}")

plt.plot(ot_profile[0], ot_profile[1])
plt.xlabel("Transported Mass")
plt.ylabel("Optimal Partial Transport Cost")
plt.show()
```

### 5. Robust Partial p-Wasserstein Distance (RPW)

```python
import numpy as np
from cot import RPW

DA = np.array([...])
SB = np.array([...])
C = np.array([...])
p = 1
k = 1
delta = 0.1

rpw_cost = RPW(DA, SB, C, delta, k, p)
print(f"Robust Partial p-Wasserstein Distance: {rpw_cost}")
```

## Research Background

### LMR Algorithm

The `cot.transport_lmr` module is based on [LMR algorithm](https://github.com/nathaniellahn/CombinatorialOptimalTransport), which developed an efficient combinatorial algorithm for computing additive approximations of optimal transport distances between discrete distributions.

**Citation:**

```
@article{lahn2019graph,
  title={A graph theoretic additive approximation of optimal transport},
  author={Lahn, Nathaniel and Mulchandani, Deepika and Raghvendra, Sharath},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

### Combinatorial Parallel OT

The modules `cot.assignment`, `cot.assignment_torch`, and `cot.transport_torch` are based on [research](https://github.com/kaiyiz/Combinatorial-Parallel-OT) "A combinatorial algorithm for approximating the optimal transport in the parallel and mpc settings", which develops efficient parallel combinatorial algorithms for computing OT and assignment problems (a special case of OT) using push-relabel techniques.

**Citation:**

```
@article{lahn2023combinatorial,
  title={A combinatorial algorithm for approximating the optimal transport in the parallel and mpc settings},
  author={Lahn, Nathaniel and Raghvendra, Sharath and Zhang, Kaiyi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={21675--21686},
  year={2023}
}
```

### Computing-all-optimal-partial-transport

The `LMR.OT_Profile` implementation is based on [research](https://github.com/kaiyiz/Computing-all-optimal-partial-transport) exploring methods for computing all partial optimal transports between discrete distributions. 

**Citation:**

```
@inproceedings{phatak2023computing,
  title={Computing all optimal partial transports},
  author={Phatak, Abhijeet and Raghvendra, Sharath and Tripathy, Chittaranjan and Zhang, Kaiyi},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
}
```

### Robust-Partial-p-Wasserstein-Based-Metric

The `LMR.RPW` module is derived from [research](https://github.com/kaiyiz/Robust-Partial-p-Wasserstein-Based-Metric) on robust metrics for comparing distributions, focusing on partial Wasserstein distance and robustness to outliers.

**Citation:**

```
@article{raghvendra2024new,
  title={A New Robust Partial $ p $-Wasserstein-Based Metric for Comparing Distributions},
  author={Raghvendra, Sharath and Shirzadian, Pouyan and Zhang, Kaiyi},
  journal={arXiv preprint arXiv:2405.03664},
  year={2024}
}
```

## Contributing

We welcome contributions to PyCoOT! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.