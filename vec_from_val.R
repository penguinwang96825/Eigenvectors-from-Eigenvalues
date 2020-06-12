A = matrix(c(1, 1, -1, 1, 3, 1, -1, 1, 3), 3, 3)

vec_from_val = function(A){
  # A should be Hermitian matrix
  n = sqrt(length(A))
  Aeig = eigen(A)$values
  # V is eigenvecters matrix
  V = matrix(ncol=n, nrow=n)
  
  for (i in 1:n){
    AM = Aeig[-i]
    for (j in 1:n){
      # Minors matrix B
      B = A[-j, ]
      B = B[, -j]
      Beig = eigen(B)$values

      down = 1; up = 1
      # n_0 is the dimension of B
      n_0 = n - 1
      
      for (k in 1:n_0){
        down = down * (Aeig[i] - AM[k])
        up = up * (Aeig[i] - Beig[k])
      }
      
      V[i, j] = up / down
    }
  }
  
  return(t(V))
}

vec_from_val(A)
