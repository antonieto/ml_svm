# TODO: Turn this into a package, import to other scripts

print_clasificacion <- function (x, w, b) {
  if ((t (w) %*% x + b) >= 0)
  {
    print (x)
    print("Pertence a la clase: 1")
  }
  else
  {
    print (x)
    print("Pertence a la clase: -1")
  }
}

dot_product_kernel <- function(v1, v2) {
  return(t(v1) %*% v2)
}

#' Combines a set of vectors and applies a kernel function between combinations.
#' @param data The data frame containing vectors to be combined
#' @param kernel_function A function that applies an operation between two
#' vectors and returns a number.
get_kernel_matrix <- function(vectors, kernel_function) {
  print(vectors)
  kernel_matrix <- matrix(0, nrow=nrow(vectors), ncol=nrow(vectors))
  COMBINATION_SIZE <- 2
  combinations <- combn(nrow(vectors), COMBINATION_SIZE)
  
  for (i in 1:ncol(combinations)) {
    idx1 <- combinations[1, i]
    idx2 <- combinations[2, i]
    v1 <- vectors[idx1,]
    v2 <- vectors[idx2,]
    kernel_matrix[idx1, idx2] <- kernel_function(unlist(v1), unlist(v2))
    kernel_matrix[idx2, idx1] <- kernel_matrix[idx1, idx2]
  }
  
  # Fill in diagonal
  for (i in 1:nrow(vectors)) {
    v <- unlist(vectors[i,])
    kernel_matrix[i, i] <- kernel_function(v, v)
  }
  
  return(kernel_matrix)
}
