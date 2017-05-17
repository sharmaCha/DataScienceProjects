rm(list =ls(all=T)); gc()

set.seed(6)
library(FeatureHashing)
library(data.table)
library(Matrix)
library(xgboost)

LogLoss <- function(actual, prediction) {
  epsilon <- .00000001
  yhat <- pmin(pmax(prediction, epsilon), 1-epsilon)
  logloss <- -mean(actual*log(yhat)
                   + (1-actual)*log(1 - yhat))
  return(logloss)
}


test <- fread("/home/test", colClasses='character', header=F, skip = 1)
tmpT = nrow(test)
test$V2 = substr(test$V2,7,8)
test$V1 <- NULL
tmpN = names(test)
setnames(test,tmpN,tmpN)
test = hashed.model.matrix(~., data=test, hash_size=2^24,transpose=FALSE, keep.hashing_mapping=TRUE)
test = as(test, "dgCMatrix")

tmpT0 = 2e6
test0 <- fread("/home/train", colClasses='character', header=F, skip=1, nrows=tmpT0)
test0$V3 = substr(test0$V3,7,8)
test0$V1 <- NULL
tclick0 = as.numeric(test0$V2)
test0$V2 = NULL
tmpN0 = names(test0)
setnames(test0,tmpN0,tmpN)
test0 = hashed.model.matrix(~., data=test0, hash_size=2^24,transpose=FALSE, keep.hashing_mapping=TRUE)
test0 = as(test0, "dgCMatrix")


max.depth = 10
gamma = 1.2
eta = 0.8
nround = 10
min_child_weight = 3

#40,428,968 rows in train
tmpS = seq(0.01,1,0.01)
tmpP = rep(0.16,tmpT)
tmpP0 = rep(0.16, tmpT0)
bestLoss = LogLoss(tclick0, tmpP0)
count = 0
tmpX = seq(20000000,40000000,by=5000000)
for (x in tmpX) {
  train <- fread("/home/train", colClasses='character', header=F, skip=x, nrows=2e6)
  train$V3 = substr(train$V3,7,8)
  train$V1 <- NULL
  tclick = as.numeric(train$V2)
  train$V2 = NULL
  tmpN0 = names(train)
  setnames(train,tmpN0,tmpN)
  
  train = hashed.model.matrix(~., data=train, hash_size=2^24,transpose=FALSE, keep.hashing_mapping=TRUE)
  train = as(train, "dgCMatrix")
  
  
  bst <- xgboost(data = train, label = tclick, max.depth=max.depth, gamma = gamma,
                 eta = eta, nround = nround, min_child_weight = min_child_weight,
                 objective = "binary:logistic", booster = 'gbtree')
  
  #check if this lowers logloss
  bestParam = 0
  tmpPred = predict(bst,test0)
  for (param in tmpS) {
    tmpLoss = LogLoss(tclick0,(1-param)*tmpP0 + param*tmpPred)
    if (tmpLoss < bestLoss) {
      bestLoss = tmpLoss
      bestParam = param
      print(bestLoss)
    }
  }
  tmpP0 = (1-bestParam)*tmpP0 + bestParam*tmpPred
  
  
  tmpP = (1-bestParam)*tmpP +  bestParam*predict(bst,test)
  count = count + 1
  print(count)
}






mat1 <- fread("/home/fast_sol.csv", header=T, colClasses=c("character","numeric"), data.table=F) 
mat1$click = tmpP

write.csv(mat1, file = "/home/xgboost.csv", quote=FALSE, row.names=FALSE)
