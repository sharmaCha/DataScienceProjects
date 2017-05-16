#reading the dataset from the ad.data file
dataset <- read.csv("C:/Midterm/Case2/ad.data", 
                    strip.white = TRUE, na.strings = "?", header = FALSE)
#View(dataset)

#reading the column names from the ad.names file
column.names <-  read.table("C:/Midterm/Case2/ad.names", sep=":", skip = 3, blank.lines.skip= TRUE, comment.char = "|", fill=FALSE, header=FALSE, strip.white=TRUE)
#View(column.names)

#converting column.names$V1 to vector so that assign it to dataset
name <- as.vector(column.names$V1)
#appending the 'classes' column name to vector to complete the dataset
name <- c(name, "classes")
#checking the length of the vector, to make sure vector is complete
length(name)

name[which(colnames(name) %in% c("ancurl*com"))]
length(name[name == "ancurl*com"])

#setting the column names to the dataset
colnames(dataset) <- name
str(dataset)

#changing column height and width to numeric
dataset[,1] <- as.numeric(dataset[,1])
dataset[,2] <- as.numeric(dataset[,2])
dataset[,4] <- as.numeric(dataset[,4])

#changing the value from 
dataset$classes <- ifelse(dataset$classes=="ad.", 1L, 0L) 

#write to csv file
write.csv(dataset, file="/R Workspace/Mid Term/Problem 2/InternetADs.csv", row.names = F)

#install.packages("caret")
library(caret)
#setting the seed
set.seed(1)
#Splitting the dataset into training and test to run the Decision Tree Model
training <- createDataPartition(dataset$classes, p=0.8)
#sample(seq_len(nrow(dataset)), size = floor(0.8 * nrow(dataset)))
trainData <- dataset[training[[1]], ]
testData <- dataset[-training[[1]], ]

#Handling missing values
impute <- function(x) {
  if (class(x) == "numeric") { 
    ifelse(is.na(x), mean(x, na.rm = TRUE), x) 
  }
  else {
    ifelse(is.na(x), as.factor(median(x, na.rm = T)), x)
  }
}

trainData <- data.frame(lapply(trainData, impute))
trainData$classes <- as.factor(trainData$classes)

testData <- data.frame(lapply(testData, impute))
testData$classes <- as.factor(testData$classes)

#Running the decision Tree Model
DT_model <- rpart(classes ~ height + width + aratio + local, data = trainData, method = "class")
plot(DT_model, compress=TRUE)
title(main="Decision Tree For Ad/Non-Ad dataset")
text(DT_model, pretty=0)
summary(DT_model)

#Just a clearer version for Decision Tree
plot(DT_model, uniform=TRUE,main="Decision Tree For Ad/Non-Ad dataset")
text(DT_model, use.n=TRUE, all=TRUE, cex=.8)


# Pruning the tree 
library(tree)
DT_prunedModel<- prune(DT_model, cp= DT_model$cptable[which.min(DT_model$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(DT_prunedModel, uniform=TRUE, main="Pruned Decision Tree for Ad-NonAd dataset")
text(DT_prunedModel, use.n=TRUE, all=TRUE, cex=.8)
#post(DT_prunedModel, file = "E:\\Study Files\\INFO 7390 - ADS\\Assignments\\Mid Term\\Problem 1\\problem1Decisiontree1.ps",title = "Pruned Classification Tree for Credit")
# display the pruned results 
printcp(DT_prunedModel) 
plotcp(DT_prunedModel) 
summary(DT_prunedModel)

#Since there is no effect after pruning the decision tree, we are using the initial model 

table(testData$classes)
#predicting for test data
predicted_Dataset <- predict(DT_model, testData, type = "class")
#View(as.data.frame(DecisionPrediction1))
class(predicted_Dataset)
View(predicted_Dataset)
#Accuracy of Model - Decision Tree - 93%
accuracy <- table(predicted_Dataset, testData[,"classes"])
#Accuracy
sum(diag(accuracy))/sum(accuracy)

#confusion matrix........................Create predicted target value as 1 or 0
library(e1071)
library(caret)
unique(predicted_Dataset)
confusionMatrix(data=factor(predicted_Dataset),reference=factor(testData$classes),positive='1')
table(predicted_Dataset,testData$classes)
error_Prediction <- 1-(sum(predicted_Dataset==testData$classes)/length(testData$classes))
View(error_Prediction)                          #error- 0.068


#Plotting the ROC Curve, Lift Curve
library(ROCR)
roc_pred <- prediction(as.numeric(predicted_Dataset), as.numeric(testData$classes))
plot(performance(roc_pred, measure="tpr", x.measure="fpr"), colorize=TRUE)

#Try this for a lift curve:
plot(performance(roc_pred, measure="lift", x.measure="rpp"), colorize=TRUE)

#Sensitivity/specificity curve and precision/recall curve:
plot(performance(roc_pred, measure="sens", x.measure="spec"), colorize=TRUE)
plot(performance(roc_pred, measure="prec", x.measure="rec"), colorize=TRUE)