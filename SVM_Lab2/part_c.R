library(kernlab)

source("lib.R")

# Define data
data <- data.frame(
  x1 = c(2, 2, -2, -2, 2, 2, -2, -2, 1, 1, -1, -1),
  x2 = c(2, -2, -2, 2, 2, -2, -2, 2, 1, -1, -1, 1),
  y = c(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1)
)


svm <- ksvm(y~., data, type="C-svc",C = 100, kernel="vanilladot")

# 1. Determine support vectors 
supportVectors <- data[svm@SVindex, -3]
supportVectors

# 2. Kernel values 

# Exclude last column, since it is only labels
kernel_matrix <- get_kernel_matrix(data[,-3], dot_product_kernel)
kernel_matrix

# 3. Width of street
w <- colSums(coef(svm)[[1]] * data[SVindex(svm),])
w <- w[-3]
b <- svm@b

widthB = 2/(sqrt(sum((w)^2)))
widthB

# 4. Vector of weights, normal to hyperplane (W)
w

# 5. Vector B
b

# Plot
plot(x2 ~ x1, data = data, col = ifelse(y == -1, "red", "blue"), pch = 19, main = "SVM Decision Boundary", xlab = "x1", ylab = "x2") 
cat (-w[1]/w[2],"*x+",-b/w[2],"=1")
cat (-w[1]/w[2],"*x+",-b/w[2],"=-1")
cat (-w[1]/w[2],"*x+",-b/w[2],"=0")

abline(b/w[2],-w[1]/w[2])
abline((b+1)/w[2],-w[1]/w[2],lty=2)
abline((b-1)/w[2],-w[1]/w[2],lty=2)

# 6. Hyperplane equation, negative and possitive support planes
paste(c("[",w,"]' * x + [",b,"] = 0"), collapse=" ")
paste(c("[",w,"]' * x + [",b,"] = 1"), collapse=" ")
paste(c("[",w,"]' * x + [",b,"] = -1"), collapse=" ")

# 7. Classification
print_clasificacion(c(5,6), w, b)
print_clasificacion(c(1,-4), w, b)