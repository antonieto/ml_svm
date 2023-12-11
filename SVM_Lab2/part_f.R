library(kernlab)

source("lib.R")

# Define data
data <- iris
svm <- ksvm(Species~., data, type="C-svc",C = 100, kernel="vanilladot")

# 1. Determine support vectors 
supportVectors <- data[svm@SVindex, -3]
supportVectors

# 2. Kernel values 

# Exclude last column, since it is only labels
kernel_matrix <- get_kernel_matrix(data[,-3], dot_product_kernel)
kernel_matrix

# 3. Width of street
w <- colSums(coef(svm)[[1]] * data[SVindex(svm),])
w <- w[-5]
b <- svm@b

widthB = 2/(sqrt(sum((w)^2)))
widthB

# 4. Vector of weights, normal to hyperplane (W)
w 

# 5. Vector B
b

# 6. Hyperplane equation, negative and possitive support planes
paste(c("[",w,"]' * x + [",b,"] = 0"), collapse=" ")
paste(c("[",w,"]' * x + [",b,"] = 1"), collapse=" ")
paste(c("[",w,"]' * x + [",b,"] = -1"), collapse=" ")
