library(e1071)

source("lib.R")

# Define data
data <- data.frame(
  x1 = c(2, 2, -2, -2, 2, 2, -2, -2, 1, 1, -1, -1),
  x2 = c(2, -2, -2, 2, 2, -2, -2, 2, 1, -1, -1, 1),
  y = c(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1)
)

svmC <- svm(y~., data , kernel="linear")

# 1. Determine support vectors 
supportVectors <- data[svmC$index,1:2]
supportVectors
plot(supportVectors)

# 2. Kernel values 

# Exclude last column, since it is only labels
kernel_matrix <- get_kernel_matrix(data[,-3], dot_product_kernel)
kernel_matrix

# 3. Width of street
wC <- crossprod(as.matrix(supportVectors), svmC$coefs)
widthC = 2/(sqrt(sum((wC)^2)))
widthC

# 4. Vector of weights, normal to hyperplane (W)
wC

# 5. Vector B
bC <- svmC$rho
bC


# 6. Hyperplane equation, negative and possitive support planes
paste(c("[",wC,"]' * x + [",bC,"] = 0"), collapse=" ")
paste(c("[",wC,"]' * x + [",bC,"] = 1"), collapse=" ")
paste(c("[",wC,"]' * x + [",bC,"] = -1"), collapse=" ")

# 7. Classification
print_clasificacion(c(5,6), wC, bC)
print_clasificacion(c(1,-4), wC, bC)