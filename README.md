# Eigenvectors_from_Eigenvalues
Python implementation of Terence Tao's paper "Eigenvectors from eigenvalues". 

## Introduction
Detailed process is in Terence Tao's [website](https://terrytao.wordpress.com/2019/08/13/eigenvectors-from-eigenvalues/). 

## Import Packages
```python
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
```

## Eigenvector-eigenvalue Identity Theorem

### Eigenvector-eigenvalue Identity
![Eigenvector-eigenvalue identity](https://github.com/penguinwang96825/Eigenvectors_from_Eigenvalues/blob/master/Eigenvector-eigenvalue%20identity.png)

Reference from [fansong](https://dearxxj.github.io/post/7/).
```python
def eigenvectors_from_eigenvalues(A, eig_val_A=None):
    """
    Implementation of Eigenvector-eigenvalue Identity Theorem

    Parameters:
        A: (n, n) Hermitian matrix (array-like)
        eig_val_A: (n, ) vector (float ndarray)
    Return: 
        eig_vec_A: Eigenvectors of matrix A
    """
    n = A.shape[0]
    # Produce eig_val_A by scipy.linalg.eigh() function
    if eig_val_A is None:
        eig_val_A, _ = eigh(A)
    eig_vec_A = np.zeros((n, n))
    start = time.time()
    for k in range(n):
        # Remove the k-th row
        M = np.delete(A, k, axis=0)
        # Remove the k-th column
        M = np.delete(M, k, axis=1)
        # Produce eig_val_M by scipy.linalg.eigh() function
        eig_val_M, _ = eigh(M)

        nominator = [np.prod(eig_val_A[i] - eig_val_M) for i in range(n)]
        denominator = [np.prod(np.delete(eig_val_A[i] - eig_val_A, i)) for i in range(n)]

        eig_vec_A[k, :] = np.array(nominator) / np.array(denominator)
    elapse_time = time.time() - start
    print("It takes {:.8f}s to compute eigenvectors using Eigenvector-eigenvalue Identity.".format(elapse_time))
    return eig_vec_A
```
Test on matrix A.
```python
A = np.array([[1, 1, -1], [1, 3, 1], [-1, 1, 3]])
eig_vec_A = eigenvectors_from_eigenvalues(A)
print(eig_vec_A)

start = time.time()
eig_val_A, eig_vec_A = eigh(A); eig_vec_A
print("\nIt takes {:.8f}s to compute eigenvectors using scipy.linalg.eigh() function.".format(time.time()-start))
print(eig_vec_A)
```
```console
It takes 0.00070190s to compute eigenvectors using Eigenvector-eigenvalue Identity.
[[0.66666667 0.33333333 0.        ]
 [0.16666667 0.33333333 0.5       ]
 [0.16666667 0.33333333 0.5       ]]

It takes 0.00016832s to compute eigenvectors using scipy.linalg.eigh() function.
[[ 0.81649658 -0.57735027  0.        ]
 [-0.40824829 -0.57735027  0.70710678]
 [ 0.40824829  0.57735027  0.70710678]]
```

### Generalized eigenvector-eigenvalue Identity
![Generalized eigenvector-eigenvalue identity](https://github.com/penguinwang96825/Eigenvectors_from_Eigenvalues/blob/master/Generalized%20eigenvector-eigenvalue%20identity.png)

## Conclusion
1. The eigenvector-eigenvalue identity only yields information about the magnitude of the components of a given eigenvector, but does not directly reveal the phase of these components. Otherwise, the eigenvector-eigenvalue identity may be more computationally feasible only if one has an application that requires only the component magnitudes.
2. It would be a computationally intensive task in general to compute all *n-1* eigenvalues of each of the *n* minors matrices.
3. An additional method would then be needed to calculate the signs of these components of eigenvectors.
4. It has not been seen that the eigenvector-eigenvalue identity has better speed at computing eigenvectors compared to `scipy.linalg.eigh()` function.

## Paper
1. Terence Tao, Eigenvectors from eigenvalues: a survey of a basic identity in linear algebra. [[paper](https://arxiv.org/pdf/1908.03795.pdf)]
2. Asok K. Mukherjee and Kali Kinkar Datta. Two new graph-theoretical methods for generation of eigenvectors of chemical graphs. [[paper](https://www.ias.ac.in/article/fulltext/jcsc/101/06/0499-0517)]
3. Peter B Denton, Stephen J Parke, and Xining Zhang. Eigenvalues: the Rosetta Stone for Neutrino Oscillations in Matter. [[paper](https://arxiv.org/pdf/1907.02534.pdf)]
