# Eigenvectors_from_Eigenvalues
Python implementation of Terence Tao's paper "Eigenvectors from eigenvalues".

## Import Packages
```python
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
```

## Computation of Eigenvalues and Eigenvectors

### Numpy Built-In Function
```python
%%time
A = np.array([[1, 1, -1], [1, 3, 1], [-1, 1, 3]])
eigenvalues_A, eigenvectors_A = np.linalg.eig(A)
eigenvalues_A = [round(float(x), 4) for x in eigenvalues_A]
print("Eigenvalues of matrix A: \n{}".format(eigenvalues_A))
print("Eigenvalues of matrix A: \n{}".format(eigenvectors_A))
```
```console
Eigenvalues of matrix A: 
[-0.0, 3.0, 4.0]
Eigenvalues of matrix A: 
[[-8.16496581e-01  5.77350269e-01 -3.14018492e-16]
 [ 4.08248290e-01  5.77350269e-01  7.07106781e-01]
 [-4.08248290e-01 -5.77350269e-01  7.07106781e-01]]
CPU times: user 367 µs, sys: 886 µs, total: 1.25 ms
Wall time: 1.32 ms
```

### Scipy Built-In Function
```python
%%time
A = np.array([[1, 1, -1], [1, 3, 1], [-1, 1, 3]])
eig_val, eig_vec = eigh(A)
eig_val = eig_val.tolist()
eig_val = [round(x, 4) for x in eig_val]
print("Eigenvalues of matrix A: \n{}".format(eig_val))
print("Eigenvectors of matrix A: \n{}".format(eig_vec))
```
```console
Eigenvalues of matrix A: 
[0.0, 3.0, 4.0]
Eigenvectors of matrix A: 
[[ 0.81649658 -0.57735027  0.        ]
 [-0.40824829 -0.57735027  0.70710678]
 [ 0.40824829  0.57735027  0.70710678]]
CPU times: user 772 µs, sys: 0 ns, total: 772 µs
Wall time: 911 µs
```

### Lanczos Method
Reference from [Christian Clason](https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method).
```python
def lanczos_method(A, m=100):
    """
    Reference from https://en.wikipedia.org/wiki/Lanczos_algorithm

    Parameters:
      A: n*n Hermitian matrix (array-like)
      v: initial n*1 vector with Euclidean norm 1 (array-like)
      m: number of iterations (int)
    Return:
      T: m*m tridiagonal real symmetric matrix (array-like)
      V: n*m matrix with orthonormal columns (array-like)
    """
    # Initialize parameters
    v = np.random.rand(A.shape[1])
    n = len(v)
    if m >= n: 
      m = n
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    vo = np.zeros(n)
    beta = 0

    for j in range(m - 1):
        w = np.dot(A, v)
        alfa = np.dot(w, v)
        w = w - alfa*v - beta*vo
        beta = np.sqrt(np.dot(w, w)) 
        vo = v
        v = w / beta 
        T[j, j] = alfa 
        T[j, j + 1] = beta
        T[j + 1, j] = beta
        V[j, :] = v
    w = np.dot(A, v)
    alfa = np.dot(w, v)
    w = w - alfa*v - beta*vo
    T[m - 1, m - 1] = np.dot(w, v)
    V[m - 1] = w / np.sqrt(np.dot(w, w)) 

    eigenvalues_T, eigenvectors_T = np.linalg.eig(T)
    eig_vec = V @ A

    eig_val = []
    for i in range(n):
        val = (np.dot(eig_vec[:, i].conj().T, np.dot(A, eig_vec[:, i]))) / (np.dot(eig_vec[:, i].conj().T, eig_vec[:, i]))
        eig_val.append(val)

    return eig_val, eig_vec
```

Test on a matrix.
```python
A = np.array([[1, 1, -1], [1, 3, 1], [-1, 1, 3]])
eig_val, eig_vec = lanczos_method(A)
print("Eigenvalues of matrix A: \n{}".format(eig_val))
print("Eigenvectors of matrix A: \n{}".format(eig_vec))
```
```console
Eigenvalues of matrix A: 
[0.730941255645759, 2.56483114861561, 3.0094993260146365]
Eigenvectors of matrix A: 
[[ 1.1153669   2.42871868  0.19798489]
 [-0.8453261   1.61596474  3.30661693]
 [ 0.01815338 -0.14486253 -0.18116928]]
CPU times: user 1.85 ms, sys: 0 ns, total: 1.85 ms
Wall time: 1.44 ms
```

### Power Method
Given a diagonalizable matrix A, Power Method will produce a number, which is the greatest (in absolute value) eigenvalue of A.
```python
def power_method(A, m=100):
    """
    Prameters:
        A: n*n matrix (array-like)
        m: iterations (int)
    Return:
        v_old: Eigenvector (array-like)
    """
    v_old = np.random.rand(A.shape[1])

    for _ in range(m):
        # Calculate the matrix-by-vector product A*v_old.
        v_new = np.dot(A, v_old)

        # Calculate the norm.
        v_new_norm = np.linalg.norm(v_new)

        # Re-normalize the vector.
        v_old = v_new / v_new_norm

    # Rayleigh quotient
    eig_val = (np.dot(v_old.conj().T, np.dot(A, v_old))) / (np.dot(v_old.conj().T, v_old))

    return eig_val, v_old
```

Test on Power Method.
```python
%%time
eig_val, eig_vec = power_method(A)
eig_vec = [round(x, 4) for x in eig_vec.tolist()]
print("Eigenvalues of matrix A: \n{}".format(eig_val))
print("Eigenvectors of matrix A: \n{}".format(eig_vec))
```
```console
Eigenvalues of matrix A: 
3.999999999999999
Eigenvectors of matrix A: 
[0.0, 0.7071, 0.7071]
CPU times: user 712 µs, sys: 1.94 ms, total: 2.65 ms
Wall time: 2.83 ms
```

## Paper
[Eigenvectors from eigenvalues](https://arxiv.org/pdf/1908.03795.pdf)
