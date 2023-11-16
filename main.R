#kernlab

require('kernlab')

kfunction <- function(linear =0, quadratic=0)
{
  k <- function (x,y)
  {
    linear*sum((x)*(y)) + quadratic*sum((x^2)*(y^2))
  }
  class(k) <- "kernel"
  k
}
x = c(0,4)
y = c(0,4)
Z = c(1,2)
x1 = matrix(cbind(x,y),,2)
svp <- ksvm(x1,Z,type="C-svc",C = 100, kernel=kfunction(1,0),scaled=c())



#svp <- ksvm(datos$Z~.,data=datos,type="C-svc",C = 100, kernel=kfunction(1,0),scaled=c())


plot(c(min(x1[,1]), max(x1[,1])),c(min(x1[,2]), max(x1[,2])),type='n',xlab='x1',ylab='x2')
title(main='Linear Separable Features')
ymat <- ymatrix(svp)
points(x1[-SVindex(svp),1], x1[-SVindex(svp),2], pch = ifelse(ymat[-SVindex(svp)] < 0, 2, 1))
points(x1[SVindex(svp),1], x1[SVindex(svp),2], pch = ifelse(ymat[SVindex(svp)] < 0, 17, 16))

# Extract w and b from the model
w <- colSums(coef(svp)[[1]] * x1[SVindex(svp),])
b <- b(svp)

# Draw the lines
abline(b/w[2],-w[1]/w[2])
abline((b+1)/w[2],-w[1]/w[2],lty=2)
abline((b-1)/w[2],-w[1]/w[2],lty=2)