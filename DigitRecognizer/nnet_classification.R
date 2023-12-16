library(caret)
library(rpart)
library(nnet)
library(e1071)

set.seed(123)

data_titanic <- read.csv("Titanic_Lab3/train.csv", header=T)

# Eliminar datos faltantes
data_titanic<-na.omit(data_titanic)

# Eliminar columnas no necesarias
data_titanic$Name=NULL
data_titanic$PassengerId=NULL
data_titanic$Ticket=NULL
data_titanic$Cabin=NULL

# Validacion cruzada
set.seed(200)
d_size<-dim(data_titanic)[1]
dtest_size <- ceiling(0.2*d_size)
samples<-sample(d_size, d_size, replace = FALSE)
indexes<-samples[1:dtest_size]
dtrain<-data_titanic[-indexes,]
dtest<-data_titanic[indexes,]

#-------------------------------------------------------------------------------
# Perceptron


# Encontrar mejor valor para size
max_size <- 50
accuracy_vector <- numeric(max_size)

for (size in 1:max_size) {
  perceptronTitanic <- nnet(Survived ~ ., data = dtrain, size = size, trace = FALSE)
  
  # Make predictions on the testing set
  predictions <- predict(perceptronTitanic, newdata = dtest)
  
  # Compare predicted values with actual values
  confusion_matrix <- table(predictions, dtest$Survived)
  
  # Calculate accuracy
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  # Store accuracy in the vector
  accuracy_vector[size] <- accuracy
  
  print(paste("Accuracy size =", size, ":", accuracy))
}

# Plot accuracy vs size
plot(1:max_size, accuracy_vector, type = "l", col = "blue", xlab = "Size", ylab = "Accuracy",
     main = "Accuracy vs Size for Neural Network")

# plan: cada uno entrena un clasificador cambiando los parametros -> jupyter notebook