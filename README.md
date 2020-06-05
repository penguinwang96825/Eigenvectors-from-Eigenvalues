# Eigenvectors_from_Eigenvalues
Python implementation of the paper "Eigenvectors from eigenvalues".

## Import Packages
```python
import numpy as np
import pandas as pd
```

## Eigenvalue

### Lanczos Method
Reference from https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method
```python
def lanczos_method(A, v, m=100):
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
    return T, V
```

Test on a matrix.
```python
A = np.array([[1, 1, -1], [1, 3, 1], [-1, 1, 3]])
v = np.random.rand(3, )
T, V = lanczos_method(A, v)

print(T)
print(V)
```
```console
[[1.49296872 1.74725883 0.        ]
 [1.74725883 2.64273504 0.59326054]
 [0.         0.59326054 0.24264706]]
[[-0.29656446  0.94276987  0.15242863]
 [ 0.00183568 -0.56649299  0.82406452]
 [-0.97292295 -0.2120978   0.09184473]]
```
Compare to `np.linalg.eig()` numpy built-in function.
```python
eigenvalues_A, _ = np.linalg.eig(A)
eigenvalues_T, _ = np.linalg.eig(T)

eigenvalues_A = [round(float(x), 4) for x in eigenvalues_A]
eigenvalues_T = [round(float(x), 4) for x in eigenvalues_T]

print("Eigenvalues of matrix A: {}".format(eigenvalues_A))
print("Eigenvalues of matrix T: {}".format(eigenvalues_T))
```
```console
Eigenvalues of matrix A: [-0.0, 3.0, 4.0]
Eigenvalues of matrix T: [3.9698, 0.55, -0.1415]
```


### Power Method
```python
def power_method(A, start, max_iter=100):
    """
    Parameters:
      A: matrix (array-like)
      start: initial vector
      max_iter: int
    """
    result = start
    for i in range(max_iter):
        result = A*result
        result = result / np.linalg.norm(result)
    return result
    
result = power_method(A, start=np.random.rand(3, ), max_iter=100); result
```
```console
array([[5.56688351e-49, 1.18137569e-48, 1.53922494e-48],
       [5.56688351e-49, 6.08854474e-01, 1.53922494e-48],
       [5.56688351e-49, 1.18137569e-48, 7.93281935e-01]])
```


## Paper
[Eigenvectors from eigenvalues](https://arxiv.org/pdf/1908.03795.pdf)
