# Importamos las Librer�as necesarias
library (kernlab)
library(e1071)

# Creamos la funci�n que dir� a que clase pertenece cada punto


## APARTADO A ##################################################################

# Creamos el conjunto de datos
dataA <- data.frame(
  x1 = c(0, 4),
  x2 = c(0, 4),
  y = c(1, -1)
)
# Indicamos que la columna y es la importante
dataA$y <- as.factor(dataA$y)
  
# Creamos el SVM con los datos del A con un kernel lineal
svmA <- svm(y~., dataA , kernel="linear")

#Vectores de soporte
vsA <- dataA[svmA$index,1:2]

# Calculamos los valores del kernel
x1=c(0,0)
x2=c(4,4)
KAA=t (x1) %*% x1
KAB=t (x1) %*% x2
KBB=t (x2) %*% x2

# Vector de pesos normal al hiperplano (W)
# Hacemos el CrosProduct entre los vectores soporte y el coe. de Lagrange
wA <- crossprod(as.matrix(vsA), svmA$coefs)

# Calcular ancho del canal
widthA = 2/(sqrt(sum((wA)^2)))

# Calcular vector B
bA <- -svmA$rho

# Calcular la ecuacion del hiperplano y de los planos de soporte positivo
# y negativo
paste(c("[",wA,"]' * x + [",bA,"] = 0"), collapse=" ")
paste(c("[",wA,"]' * x + [",bA,"] = 1"), collapse=" ")
paste(c("[",wA,"]' * x + [",bA,"] = -1"), collapse=" ")

# Determinamos a la clase que pertenece cada uno
print_clasificacion(c(5, 6),wA, bA)
print_clasificacion(c(1, -4),wA, bA)