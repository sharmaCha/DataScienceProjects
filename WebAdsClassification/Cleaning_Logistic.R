#reading the dataset from the ad.data file
dataset <- read.csv("/R Workspace/Mid Term/Problem 2/ad-dataset/ad.data", 
                    strip.white = TRUE, na.strings = "?", header = FALSE)
#View(dataset)

#reading the column names from the ad.names file
column.names <-  read.table("/R Workspace/Mid Term/Problem 2/ad-dataset/ad.names", sep=":", skip = 3, blank.lines.skip= TRUE, comment.char = "|", fill=FALSE, header=FALSE, strip.white=TRUE)
View(column.names)

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
#View(as.data.frame(dataset$classes))

#view the dataset
#View(dataset)
#View(summary(dataset))

#write to csv file
write.csv(dataset, file="/R Workspace/Mid Term/Problem 2/InternetADs.csv", row.names = F)

#install.packages("caret")
library(caret)
#setting the seed
set.seed(1)
training <- createDataPartition(dataset$classes, p=0.8)
  #sample(seq_len(nrow(dataset)), size = floor(0.8 * nrow(dataset)))
train <- dataset[training[[1]], ]
test <- dataset[-training[[1]], ]

impute <- function(x) {
  if (class(x) == "numeric") { 
    ifelse(is.na(x), mean(x, na.rm = TRUE), x) 
  }
  else {
    ifelse(is.na(x), as.factor(median(x, na.rm = T)), x)
  }
}

train <- data.frame(lapply(train, impute))
train$classes <- as.factor(train$classes)

test <- data.frame(lapply(test, impute))
test$classes <- as.factor(test$classes)

#running the logistic regression on the dataset
library(ggplot2)
if (!exists("lm.fit")) { 
  print("Fitting logistic regression")
  lm.fit <- glm(classes ~ height + width + aratio + local, train, family = binomial())
}

lm.probs <- predict(lm.fit, test, type = "response")
lm.preds <- ifelse(lm.probs > 0.5, 1, 0)

library(pscl)
pR2(lm.fit)

table(test$classes)
table(lm.preds)
print(paste("Logistic regression accuracy estimate: ", mean(lm.preds == test$classes)))

#install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(data=factor(lm.preds),reference=factor(test$classes), positive='1')




#Decision Tree model
#install.packages("rpart")
library(rpart)

classFit1 <- rpart(classes ~ ., data=train,method="class")
summary(classFit1)
plot(classFit1)
text(classFit1, pretty=0)

#install.packages('rattle')
#install.packages('rpart.plot')
install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(classFit1)

#predicting for test data
DecisionPrediction1 <- predict(classFit1, test, type = "class")
#View(as.data.frame(DecisionPrediction1))

#View(test)
#saving my prediction to my dataframe and new file
submitPredictClass1 <- data.frame(X = test$default_payment, default_payment = DecisionPrediction1)


#optimal values
printcp(classFit1) # display the results 
plotcp(classFit1) 
summary(classFit1)

# plot tree 
plot(classFit1, uniform=TRUE,main="Classification Tree for Credit")
text(classFit1, use.n=TRUE, all=TRUE, cex=.8)

# prune the tree 
pfit1<- prune(classFit1, cp= classFit1$cptable[which.min(classFit1$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(pfit1, uniform=TRUE, main="Pruned Classification Tree for Credit")
text(pfit1, use.n=TRUE, all=TRUE, cex=.8)
post(pfit1, file = "E:\\Study Files\\INFO 7390 - ADS\\Assignments\\Mid Term\\Problem 1\\problem1Decisiontree1.ps",title = "Pruned Classification Tree for Credit")
# display the pruned results 
printcp(pfit1) 
plotcp(pfit1) 
summary(pfit1)

