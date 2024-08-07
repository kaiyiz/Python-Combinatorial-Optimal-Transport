# Python Combinatorial Optimal Transport (PyCoOT)

Website: [https://kaiyiz.github.io/Python-Combinatorial-Optimal-Transport/](https://kaiyiz.github.io/Python-Combinatorial-Optimal-Transport/)

## Installation

```bash
pip install PyCoOT
```

## Overview

**PyCoOT** is a Python library for optimal transport computation using combinatorial methods, offering state-of-the-art algorithms to compute additive approximations of optimal transport distances between discrete distributions. PyCoOT supports large-scale computations through GPU acceleration, making it a powerful tool for OT problems. Additionaly, our library leverages the properties of combinatorial algorithms to compute various OT related problems efficiently.

## Key Features

- **1-Wasserstein Distance Computation:** Efficient computation of additive approximations of 1-Wasserstein distances based on LMR algorithms.
- **Scalable Combinatorial Approach:** PyCoOT employs combinatorial algorithms to solve OT in its original approximate formulation without entropic regularization and provides parallel implementations including GPU acceleration.
- **Fast Assignment Problem Solution:** Faster impletation for approximate assignment problems, a special case of OT, and also supporting GPU acceleration.
- **Optimal Transport Profile:** Detailed calculation of optimal transport profiles between discrete distributions.
- **Robust Partial $p$-Wasserstein Distance:** An outlier robust metric for comparing distributions based on partial $p$-Wasserstein distance.

## Dependencies
This library requires JVM (Java 8 or higher) for for the Java-based combinatorial algorithm implementation. 
The library is tested with Python 3.9 and relies on the following Python modules.
```
NumPy (v1.24)
PyTorch (v1.13)
jpype1 (v1.5) 
```

## Examples

### 1. LMR Algorithm (1-Wasserstein Distance Computation)
The `cot.transport_lmr` module is based on [LMR algorithm](https://github.com/nathaniellahn/CombinatorialOptimalTransport), which developed an efficient combinatorial algorithm for computing additive approximations of optimal transport distances between discrete distributions.

```python
import numpy as np
from cot import transport_lmr

n = 100
DA = np.random.rand(n)  # Demand array
DA = DA/np.sum(DA)
SB = np.random.rand(n)  # Supply array
SB = SB/np.sum(SB)
C = np.random.rand(n,n)   # Cost matrix
C = C/C.max()
delta = 0.1           # Approximation error

ot_cost = transport_lmr(DA, SB, C, delta)
print(f"1-Wasserstein Distance (LMR Algorithm): {ot_cost}")
```

### 2. Combinatorial Parallel OT Algorithms (1-Wasserstein Distance Computation)
The modules `cot.assignment`, `cot.assignment_torch`, and `cot.transport_torch` are based on research: [A combinatorial algorithm for approximating the optimal transport in the parallel and mpc settings]((https://github.com/kaiyiz/Combinatorial-Parallel-OT))". This work proposed efficient parallel combinatorial algorithms for computing OT and assignment problems (a special case of OT) based on push-relabel method.
#### a. Transport Problem (Torch Implementation)

```python
import torch
from cot import transport_torch

n = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DA = np.random.rand(n)  # Demand array
DA = DA/np.sum(DA)
DA_tensor = torch.tensor(DA, device=device)
SB = np.random.rand(n)  # Supply array
SB = SB/np.sum(SB)
SB_tensor = torch.tensor(SB, device=device)
C = np.random.rand(n,n)   # Cost matrix
C = C/C.max()
C_tensor = torch.tensor(C, device=device)
delta = 0.1               # Approximation error

F, yA, yB, ot_cost = transport_torch(DA_tensor, SB_tensor, C_tensor, delta, device=device)
print(f"1-Wasserstein Distance (Push-Relabel): {ot_cost}")
```

#### b. Assignment Problem (NumPy and Torch Implementation)

```python
import numpy as np
import torch
from cot import assignment, assignment_torch

n = 100
W = np.random.rand(n,n)   # Cost matrix
W = W/W.max()
C = 1             # Scale of cost metric
delta = 0.1           # Approximation error

# NumPy implementation
Mb, yA, yB, assignment_cost = assignment(W, C, delta)
print(f"Bipartite Assignment Cost (Push-Relabel, NumPy): {assignment_cost}")

# Torch implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W_torch = torch.tensor(W, device=device)
Mb, yA, yB, assignment_cost = assignment_torch(W_torch, C, delta, device=device)
print(f"Bipartite Assignment Cost (Push-Relabel, Torch): {assignment_cost}")
```

### 3. Computing Optimal Transport Profile (OT Profile)
The `LMR.OT_Profile` implementation is based on the research work: [Computing all optimal partial transport](https://github.com/kaiyiz/Computing-all-optimal-partial-transport), which proposed a method to compute a map between the cost of partial OT and transported mass. 

```python
import numpy as np
from cot import ot_profile
from matplotlib import pyplot as plt

n = 100
DA = np.random.rand(n)  # Demand array
DA = DA/np.sum(DA)
SB = np.random.rand(n)  # Supply array
SB = SB/np.sum(SB)
C = np.random.rand(n,n)   # Cost matrix
C = C/C.max()

otp = ot_profile(DA, SB, C, delta)
print(f"Optimal Transport Profile: {otp}")

plt.plot(otp[0], otp[1])
plt.xlabel("Transported Mass")
plt.ylabel("Optimal Partial Transport Cost")
plt.show()
```

### 4. Robust Partial p-Wasserstein Distance (RPW)
The `LMR.RPW` module is derived from research: [Robust Partial $p$-Wasserstein Based Metric](https://github.com/kaiyiz/Robust-Partial-p-Wasserstein-Based-Metric) that proposed a robust metrics for comparing distributions, focusing on partial Wasserstein distance and robustness to outliers.
```python
import numpy as np
from cot import rpw

n = 100
DA = np.random.rand(n)  # Demand array
DA = DA/np.sum(DA)
SB = np.random.rand(n)  # Supply array
SB = SB/np.sum(SB)
C = np.random.rand(n,n)   # Cost matrix
C = C/C.max()
p = 1
k = 1
delta = 0.1

rpw_cost = rpw(DA, SB, C, delta, k, p)
print(f"Robust Partial p-Wasserstein Distance: {rpw_cost}")
```

## Citations
If you find PyCoOT useful, please consider citing the following papers:

### LMR Algorithm

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

### Computing All Optimal Partial Transport

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

```
@inproceedings{raghvendranew,
  title={A New Robust Partial p-Wasserstein-Based Metric for Comparing Distributions},
  author={Raghvendra, Sharath and Shirzadian, Pouyan and Zhang, Kaiyi},
  booktitle={Forty-first International Conference on Machine Learning}
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