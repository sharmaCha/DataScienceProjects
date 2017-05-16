setwd("E://Applications//4th Gen Job//God//Projects//CTR-Prediction-master")
library(caret)
#install.packages("gmp")
library(gmp)
library(data.table)

########################################## For Text File #################
train_file    <- 'SampleTrain.txt'
test_file     <- 'SampleTest.txt'
req_cols      <- c("hour","C1","banner_pos","device_type","device_conn_type",
                   "C14","C15","C16","C17","C18","C19","C20","C21")
output_file   <-  'analysis-output-file.txt'

# ?fread
# Similar to read.table but faster and more convenient.
# All controls such as sep, colClasses and nrows are automatically detected.
input         <- as.data.frame(fread(train_file,head=TRUE,sep=","))
# ## input <- as.data.frame(read.table(train_file,header=T,sep=","))
###########################################################################

############# For CSV Actual File ######################################
## Train file was 5.6 GB, we split into 100000 records ( each file = 15 MB file ) and 
# we again split into 10000 records and take the first file for our training
# train_file    <- 'train_1_1.csv'
# 
# ## Test file was 673 MB, we split into 10000 records ( each file = 1.5 MB file ) and 
# # we take the first file for our testing
# test_file     <- 'test_1.csv'
# req_cols      <- c("hour","C1","banner_pos","device_type","device_conn_type",
#                    "C14","C15","C16","C17","C18","C19","C20","C21")
# output_file   <-  'analysis-output-file.txt'
# input         <- as.data.frame(read.csv(train_file,header=T,sep=","))
# # length(complete.cases(input))
##################################################################

inputdata     <- input[,c("click",req_cols)]
## Sample the training data into Training and Validation data into 70% and 30%
inTrain       <- createDataPartition(y=inputdata$click,p=0.7,list=FALSE)
tr            <- inputdata[inTrain,]
cv            <- inputdata[-inTrain,]
tc            <- trainControl(method="cv",number=3)
tc
#### random Forest #######
rfmodel       <- train(tr$click~.,method="rf",data=tr,trControl=tc,
                       preProcess=c("center","scale"))
saveRDS(rfmodel,"rfcvmodel.RDS")
predrfcv      <- predict(rfmodel,cv[,req_cols])
predrfcv      <- replace(predrfcv, predrfcv<0.5 ,as.integer(0))
predrfcv      <- replace(predrfcv, predrfcv>=0.5 ,as.integer(1))
rfCorrect     <- sum(predrfcv == cv$click)

### Logistic regression ######
glmmodel      <- train(tr$click~.,method="glm",family=gaussian(),
                       data=tr,trControl=tc,preProcess=c("center","scale"))
saveRDS(glmmodel,"glmcvmodel.RDS")

?train
# 
# Value
# 
# A list is returned of class train containing:
#   
# method	- the chosen model.
# modelType	- an identifier of the model type.
# results	- a data frame the training error rate and values of the tuning parameters.
# bestTune	- a data frame with the final parameters.
# call -the (matched) function call with dots expanded
# dots	- a list containing any ... values passed to the original call
# metric	- a string that specifies what summary metric will be used to select the optimal model.
# control	- the list of control parameters.
# preProcess	- either NULL or an object of class preProcess
# finalModel	- an fit object using the best parameters
# trainingData	- a data frame
# resample	 - A data frame with columns for each performance metric. Each row corresponds to each resample. 
# If leave-one-out cross-validation or out-of-bag estimation methods are requested, 
# this will be NULL. The returnResamp argument of trainControl controls how much of the resampled
# results are saved. 
# perfNames - a character vector of performance metrics that are produced by the summary function
# maximize -	a logical recycled from the function arguments.
# yLimits -	the range of the training set outcomes.
# times -	a list of execution times: everything is for the entire call to train, 
# final for the final model fit and, optionally, prediction for the time to predict 
# new samples (see trainControl)


####################################
# The overall complexity of RF is something like ntree???mtry???(# objects)
#log(# objects)ntree???mtry???(# objects)loga(# objects); if you want to speed your computations up, 
#you can try the following:
#   
#   Use randomForest instead of party, or, even better,
#   ranger or Rborist (although both are not yet battle-tested).
#   Don't use formula, i.e. call randomForest(predictors,decision) 
# instead of randomForest(decision~.,data=input).
#   Use do.trace argument to see the OOB error in real-time;
# this way you may detect that you can lower ntree.
#   About factors; RF (and all tree methods) try to find an optimal
#   subset of levels thus scanning 2(# of levels-1)2(# of levels-1) possibilities;
#   to this end it is rather naive this factor can give you so much information 
#   -- not to mention that randomForest won't eat factors with more than 32 levels.
#   Maybe you can simply treat it as an ordered one (and thus equivalent to a normal, 
#                                                    numeric variable for RF) 
#   or cluster it in some groups, splitting this one attribute into several?

predglmcv     <- predict(glmmodel,cv[,req_cols])
predglmcv     <- replace(predglmcv, predglmcv<0.5 ,as.integer(0))
predglmcv     <- replace(predglmcv, predglmcv>=0.5 ,as.integer(1))
glmCorrect    <- sum(predglmcv == cv$click)
length(which(cv$click == 1))
# 53
length(which(cv$click == 0))
# 246

length(which(predrfcv == 1))
# 8
length(which(predrfcv == 0))
# 291

rfCorrect
# 250
glmCorrect
# 246
if(rfCorrect > glmCorrect) best_model  <- rfmodel else best_model  <- glmmodel
?as.bigz
# Large Sized Integer Values
# 
# Description
# 
# Class "bigz" encodes arbitrarily large integers (via GMP). 
# A simple S3 class (internally a raw vector), 
# it has been registered as formal (S4) class (via setOldClass), too.

testing       <- as.data.frame(fread(test_file,head=TRUE,sep=","))
testingAdId   <- as.bigz(testing[,1])
testData      <- testing[,req_cols]
num_testData  <- dim(testData)[1]

num_testData

predcv        <- predict(best_model,testData)
predcv        <- replace(predcv, predcv<0.5 ,as.integer(0))
predcv        <- replace(predcv, predcv>=0.5 ,as.integer(1))

result    <- ""
for(i in 1:num_testData)
    result <- paste(result,paste(testingAdId[i],predcv[i],sep=","),sep="\n")

write(result,file=output_file)
