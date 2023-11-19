library(kernlab)
library(e1071)

# Define data
data <- data.frame(
  x1 = c(2, 0, 1),
  x2 = c(0, 0, 1),
  y = c(1, -1, -1)
)

# 1. Determine support vectors 
data$y <- as.factor(data$y)
svmB <- svm(y~., data, kernel="linear")
supportVectors <- data[svmB$index,1:2]
supportVectors

# 2. Kernel values 

# Exclude last column, since it is only labels
kernel_matrix <- get_kernel_matrix(data[,-3], dot_product_kernel)
kernel_matrix

# 3. Width of street
wB <- crossprod(as.matrix(supportVectors), svmB$coefs)
widthB = 2/(sqrt(sum((wB)^2)))
widthB
# 4. Vector of weights, noraml to hyperplane (W)
wB
# 5. Vector B
bB <- svmB$rho
bB
# 6. Hyperplane equation, negative and possitive support planes
paste(c("[",wB,"]' * x + [",bB,"] = 0"), collapse=" ")
paste(c("[",wB,"]' * x + [",bB,"] = 1"), collapse=" ")
paste(c("[",wB,"]' * x + [",bB,"] = -1"), collapse=" ")

# 7. Classification
print_clasificacion(c(5,6), wB, bB)
print_clasificacion(c(1,-4), wB, bB)