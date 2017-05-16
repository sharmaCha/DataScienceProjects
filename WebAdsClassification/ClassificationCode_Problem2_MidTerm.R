#install.packages("caret")
library(caret)

#sample model fitting
#80% of the sample size
#Set the seed to make your partition reproductible
set.seed(1)

#Split the data into training and testing
training <- createDataPartition(dataset$classes, p=0.8)
trainData <- dataset[training[[1]], ]
testData <- dataset[-training[[1]], ]

#imputing the numeric 'na' with the mean and 'factor' with median
impute <- function(x) {
  if (class(x) == "numeric") { 
    ifelse(is.na(x), mean(x, na.rm = TRUE), x) 
  }
  else {
    ifelse(is.na(x), as.factor(median(x, na.rm = T)), x)
  }
}

#calling the impute function on train data
trainData <- data.frame(lapply(trainData, impute))
trainData$classes <- as.factor(trainData$classes)

#calling the impute function on test data
testData <- data.frame(lapply(testData, impute))
testData$classes <- as.factor(testData$classes)

#### 1. Logistic Regression##########
######## START OF LOGISTIC REGRESSION #######
#A. Logistic Regression steps
library(ggplot2)
print("Fitting logistic regression")
  
#logit code
lr.fitt <- glm(classes ~ height + width + aratio + local, trainData, family = binomial())

#predicting the value after fitting the model
lm.probs <- predict(lr.fitt, testData, type = "response")
#converting the prediction to binary after checking of threshold
lm.preds <- ifelse(lm.probs > 0.5, 1, 0)

#r2 in logit
#install.packages("pscl")
library(pscl)
pR2(lr.fitt)

#summary of test data, number of ads and non-ads
table(testData$classes)
#summary of prediction data, number of ads and non-ads
table(lm.preds)

print(paste("Logistic regression accuracy estimate: ", mean(lm.preds == testData$classes)))

#a model with good predictive ability should have an AUC closer to 1 (1 is ideal) than to 0.5.
# Plot the performance of the model applied to the evaluation set as
# an ROC curve.
#install.packages("ROCR")
library(ROCR)
pr.classes <- prediction(lm.preds, testData$classes)
prf.classes <- performance(pr.classes, measure = "tpr", x.measure = "fpr")

#plotting ROC Curve
plot(prf.classes, main="ROC curve", colorize=T)

# And then plotting a lift chart
#Any model with lift @ decile above 100% till minimum 3rd decile and maximum 7th decile is a good model
perf.classes <- performance(pr.classes,"lift","rpp")
#install.packages("BCA")
library(BCA)
lift.chart(c("pr.classes"), testData, targLevel = "Yes", type = "cumulative")
plot(perf.classes, main="lift curve", colorize=T)  

#plotting the AUC to see the performance
auc <- performance(pr.classes, measure = "auc")
auc <- auc@y.values[[1]]
auc  ### 79.6%

#getting summary of our model performance
# Function that returns Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}
# Function that returns Mean Absolute Percentage Error
mape <- function(error)  {
  mean(abs(error/as.numeric(testData$classes)) * 100)
}

#error
#convert numeric prediction to factor
error.logistic <- as.numeric(testData$classes) - lm.probs
error.logistic

#rmse - 1.066
rmse.logit <- rmse(error.logistic) 
rmse.logit
#calculating mean square value
classes.rms.logit <- c("RMS", rmse(error.logistic))

#mae - 0.30
mae.logit <- mae(error.logistic)
credit.mae.logit
classes.ma.logit <- c("MAE", mae(error.logistic))

#mape - 
mape.logit <- mape(error.logistic)
mape.logit
classes.map.logit <- c("MAPE", mape(error.logistic))

#performance metrics of model Logit
logit.classes.performance <- NULL
logit.classes.performance <- rbind(classes.rms.logit, classes.ma.logit, classes.map.logit, deparse.level = 0)
logit.classes.performance

#model evaluation 1 - lesser AIC means better model
SSE1 <- sum(modelLogit$residuals^2)
AIC1 <- 2*5 + 30*log(SSE1/30)
AIC1

#Validation of Predicted Values
# Create predicted target value as 1 or 0
library(e1071)
library(caret)
confusionMatrix(data=factor(lm.preds),reference=factor(testData$classes),positive='1')

#plotting the regression
plot(lr.fitt, uniform=TRUE, main="Logistic Regression for classes")

######## END OF LOGISTIC REGRESSION #######
###########################################