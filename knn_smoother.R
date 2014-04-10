library(FNN)
knnreg <- function(x, y, k, fun, subst=0) 
{
  if (is.vector(x)) x <- matrix(x, ncol=1)
  which <- which(!is.na(y))
  res <- rep(NA_real_, length(y))
  if(length(which) == 0) return(rep(subst, length(y)));
  if(k >= length(which)) return(rep(subst, length(y)));
  idx_predict <- get.knnx(data=x[which, ], query=x[-which, ], k=k)$nn.index
  res[-which] <- apply(idx_predict, 1, function(idxrow, fun) fun(y[which][idxrow]), fun)
  idx_train <- get.knn(data=x[which,], k = k)$nn.index
  res[which] <- apply(idx_train, 1, function(idxrow, fun) fun(y[which][idxrow]), fun)
  return(res)
}
