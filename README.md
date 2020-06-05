# Eigenvectors_from_Eigenvalues
Python implementation of the paper "Eigenvectors from eigenvalues".

## Eigenvalue

### Lanczos Method
Reference from https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method
```python
def Lanczos( A, v, m=100 ):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    vo   = np.zeros(n)
    beta = 0
    for j in range( m-1 ):
        w    = np.dot( A, v )
        alfa = np.dot( w, v )
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) ) 
        vo   = v
        v    = w / beta 
        T[j,j  ] = alfa 
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) ) 
    return T, V
```

### Power Method
```python
def power_method(mat, start, maxit):
    """
    Does maxit iterations of the power method
    on the matrix mat starting at start.
    Returns an approximation of the largest
    eigenvector of the matrix mat.
    """
    result = start
    for i in xrange(maxit):
        result = mat*result
        result = result/np.linalg.norm(result)
    return result
```

## Paper
[Eigenvectors from eigenvalues](https://arxiv.org/pdf/1908.03795.pdf)
