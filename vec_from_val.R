A = matrix(c(1, 1, -1, 1, 3, 1, -1, 1, 3), 3, 3)

vec_from_val = function(A){
  n = sqrt(length(A))
  Aeig = eigen(A)$values
  V = matrix(ncol=n, nrow=n)
  
  for (i in 1:n){
    AM = Aeig[-1]
    for (j in 1:n){
      B = A[-j, ]
      B = B[, -j]
      Beig = eigen(B)$values
      down = 1; up = 1
      n0 = n - 1
      
      for (k in 1:n0){
        down = down * (Aeig[i] - AM[k])
        up = up * (Aeig[i] - Beig[k])
      }
      
      V[i, j] = up / down
    }
  }
  
  return(t(V))
}

vec_from_val(A)
