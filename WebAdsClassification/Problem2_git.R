ads_data <- read.csv("/R Workspace/Mid Term/Problem 2/ad-dataset/ad.data", strip.white = TRUE,
                    na.strings = "?", 
                    header = FALSE)
View(ads_data)
ads_data[,1] <- as.numeric(ads_data[,1])
ads_data[,2] <- as.numeric(ads_data[,2])
names(ads_data)[1:3] <- c("height", "width", "aratio")
names(ads_data)[1559] <- "ad"
ads_data$ad <- ifelse(ads_data$ad=="ad.", 1L, 0L)
#View(ads_data)

#install.packages("caret")
library(caret)
set.seed(123)
training_index <- createDataPartition(ads_data$ad, p=0.8)
Xy <- ads_data[training_index[[1]], ]
Xy_vl <- ads_data[-training_index[[1]], ]

impute <- function(x) {
  if (class(x) == "numeric") { 
    ifelse(is.na(x), mean(x, na.rm = TRUE), x) 
  }
  else {
    ifelse(is.na(x), as.factor(median(x, na.rm = T)), x)
  }
}

Xy <- data.frame(lapply(Xy, impute))
Xy$ad <- as.factor(Xy$ad)
Xy_vl <- data.frame(lapply(Xy_vl, impute))
Xy_vl$ad <- as.factor(Xy_vl$ad)

#write to csv
write.csv(Xy_vl, file="/Mid Term/Problem 2/ad.csv", row.names = F)

#View(summary(Xy_vl$ad))
#sum((lr_preds == 0))
#length(rf_probs[rf_probs < 0.01])
#nrow(rf$ad)

library(ggplot2)
if (!exists("lr.fit")) { 
  print("Fitting logistic regression")
  lr.fit <- glm(ad ~ height + width + aratio + V4, Xy, family = binomial())
}
lr_probs <- predict(lr.fit, Xy_vl, type = "response")
lr_preds <- ifelse(lr_probs > 0.5, 1, 0)
table(lr_preds)
table(Xy_vl$ad)

#install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(data=factor(lr_preds),reference=factor(Xy_vl$ad),positive='1')

library(pscl)
pR2(lr.fit)

print(paste("Logistic regression accuracy estimate: ", mean(lr_preds == Xy_vl$ad)))
lr_tpr <- c()
lr_fpr <- c()
for (threshold in seq(1, 0, by = -0.01)) {
  conf_mat <- table(Xy_vl$ad, 
                    factor(ifelse(lr_probs >= threshold, 1, 0), levels = c(0, 1)))
  lr_tpr <- c(lr_tpr, conf_mat[2, 2]/(conf_mat[2, 1] + conf_mat[2, 2]))
  lr_fpr <- c(lr_fpr, conf_mat[1, 2]/(conf_mat[1, 1] + conf_mat[1, 2]))
}
conf_mat




n <- length(lr_fpr)
lr_auc <- diff(lr_fpr) %*% lr_tpr[1:n-1]
print(paste("Logistic regression AUC estimate: ", lr_auc))

install.packages("Cairo")
dyn.load(Cairo)
lr_roc <- qplot(lr_fpr, lr_tpr, geom = "step")
lr_roc <- lr_roc + ggtitle("ROC Curve for Logistic Regression") + 
  labs(x = "False Positive Rate", y = "True Positive Rate")
lr_roc

#svg("roc-03.svg")
plot(lr_roc)
dev.off()

#install.packages("randomForest")
library(randomForest)
if (!exists("rf.fit")) { 
  print("Fitting random forest")
  rf.fit <- randomForest(ad ~ height + width + aratio + V4, Xy, ntree = 1000)
}

print(paste("Random forest accuracy estimate: ", mean(predict(rf.fit, Xy_vl) == Xy_vl$ad)))
rf_probs <- predict(rf.fit, Xy_vl, type = "prob")[, 1]
rf_tpr <- c()
rf_fpr <- c()
for (threshold in seq(1, 0, by = -0.01)) {
  conf_mat <- table(rf_probs,
                    factor(ifelse(rf_probs >= threshold, 1, 0), levels = c(0, 1)))
  rf_tpr <- c(rf_tpr, conf_mat[2, 2]/(conf_mat[2, 1] + conf_mat[2, 2]))
  rf_fpr <- c(rf_fpr, conf_mat[1, 2]/(conf_mat[1, 1] + conf_mat[1, 2]))
}
sum(rf_probs >= 0.01)
rf_preds <- factor(ifelse(rf_probs >= 0.01, 1, 0), levels = c(0, 1))
confusionMatrix(rf_preds, Xy_vl$ad)
n <- length(rf_fpr)
rf_auc <- diff(rf_fpr) %*% rf_tpr[1:n-1]
print(paste("Random forest AUC estimate: ", rf_auc))
rf_roc <- qplot(rf_fpr, rf_tpr, geom = "step")
rf_roc <- rf_roc + ggtitle("ROC Curve for Random Forest") + 
  labs(x = "False Positive Rate", y = "True Positive Rate")
rf_roc
svg("~/MachineLearning/InternetAds/roc-04.svg")
plot(rf_roc)
dev.off()

results <- NULL
results <- cbind(results, as.data.frame(Xy_vl$ad))
results <- cbind(results, lr_preds)
View((results))
