
# read trainDF and test DF
trainDF = readRDS("E:\\trainDF_Final.RDS")
testDF = readRDS("E:\\testDF_Final.RDS")

set.seed(123)
split <- sample.split(trainDF, SplitRatio = 0.8)
train <- subset(trainDF, split == TRUE)
test <- subset(trainDF, split == FALSE)

sum(is.na(train))

train <- na.omit(train)

View(train)

sum(is.na(test))

test <- na.omit(test)

#### ====== Deep Learning using H2O package ====== ###
library(h2o)
localH2O = h2o.init(max_mem_size = '1g',
                    nthreads = -1) 
train_h2o = as.h2o(trainDF)
test_h2o = as.h2o(testDF)

## train model

model =
  h2o.deeplearning(x = 2:401, 
                   y = 1, # labels
                   training_frame = train_h2o,
                   activation = "RectifierWithDropout", 
                   input_dropout_ratio = 0.05,
                   hidden_dropout_ratios = c(0.5, 0.5, 0.5), 
                   balance_classes = TRUE, 
                   hidden = c(300, 300, 300), 
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, 
                   epochs = 5000) # increase the learning rates
h2o.confusionMatrix(model)
#write_csv(h2o.confusionMatrix(model), path = "E:\\h2O_deepLearning_conf.matrix.csv")

## classify test set
h2o_y_test = h2o.predict(model, test_h2o)
preds = as.vector(h2o_y_test$predict)
preds
testPreds = data.frame(testDF$photo_id, preds)
testPreds = testPreds %>% dplyr::rename(photo_id = testDF.photo_id, labels = preds)
write_csv(testPreds, path = "E:\\h2o_dl_output_pred.csv")
#View(testPreds)
# Check performance of classification model.
performance = h2o.performance(model = model)
print(performance)
#100%