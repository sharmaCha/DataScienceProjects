###########################################################
############## Text2Vec - It's a R package built to be used as an API for text analysis and natural language processing (NLP)##########################

# install.packages("text2vec")
# install.packages("deepnet")
install.packages("darch") #one time
install.packages("stringi")
install.packages("Metrics")
library(Metrics)
library(text2vec)
library(forecast)
library(darch)
library(data.table)

## setting the path of the data file which has ids, text and polarity(sentiment)
setwd("E:/INFO Big Data/Final project")
dtnew <- as.data.table(read.csv("datafile.csv"))

# changing datatype for text
dtnew <- dtnew[, text:=as.character(text)]
str(dtnew)
#assigning ids to each rows and creating test and train dataset
setkey(dtnew, id)
set.seed(2016L)
all_ids = dtnew$id
#sampling the dataset
train_ids = sample(all_ids, 700)
test_ids = setdiff(all_ids, train_ids)
train = dtnew[J(train_ids)]
test = dtnew[J(test_ids)]


# creating functions to be used in tokenizing
prep_fun =  tolower
tok_fun = word_tokenizer

#tokenizing for the training dataset 
it_train = itoken(train$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = as.character(train$id), 
                  progressbar = FALSE)

# creating vocabulary for the Training Dataset
vocab = create_vocabulary(it_train)

# vectorization of the vocabulary created
vectorizer = vocab_vectorizer(vocab)
# creating a Document Term Matrix (dtm) on the Train data set from the vectorizer developed
dtm_train = create_dtm(it_train, vectorizer)
###checking the DTM Train dimensions
dim(dtm_train)
#  700 9397
dtm_mat_train <- as.matrix(dtm_train)

#write.csv(dtm_mat_train,"DTM_Matrix_Train.csv")
#### tokenizing for the test data set 
it_test = itoken(test$text, 
                 preprocessor = prep_fun, 
                 tokenizer = tok_fun, 
                 ids = as.character(test$id), 
                 progressbar = FALSE)

# creating vocabulary for the Test Dataset
vocab = create_vocabulary(it_test)

# vectorization of the vocabulary created
vectorizer = vocab_vectorizer(vocab)
# creating a Document Term Matrix (dtm) on the TEST data set from the vectorizer developed
dtm_test = create_dtm(it_test, vectorizer)
dim(dtm_test)
##   300 5310
dtm_mat_test <- as.matrix(dtm_test)
#write.csv(dtm_mat_test,"DTM_Matrix_Test.csv")

#####################################

# ## checking if the DTM has rows, equal to the no. of documents, column should be equal to the no. of unique terms
# identical(rownames(dtm_train), train$id)
# 
# identical(rownames(dtm_test), test$id)

### Pruning vocabulary - We need to remove the words which make no sense to the Text for Analysis. Words like "I", "Me", etc.
## first we do with the train dataset
stop_words = c("he","she","it","them","themselves","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours")
vocab = create_vocabulary(it_train, stopwords = stop_words)

pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)
# create dtm_train with new pruned vocabulary vectorizer

dtm_train  = create_dtm(it_train, vectorizer)
dtm_mat_train <- as.matrix(dtm_train)
dim(dtm_train)
#  700 1026

##Conclusion: we previously had 9397 columns and now we have 1026 columns
##This reduction in columns will help us to both accuracy improvement
##(because we removed “noise”) and reducing the training time.

## next we do with the test dataset

dtm_test   = create_dtm(it_test, vectorizer)
dtm_mat_test <- as.matrix(dtm_test)

dim(dtm_test)
## 300 1026


## latent Dirichlet allocation (LDA) is a generative statistical model that allows 
### sets of observations to be explained by unobserved groups that explain why some parts of the data are similar.
dtm_train_lda <- create_dtm(it_train, vectorizer, type = "lda_c")

lda_model_amazon =   LDA$new(n_topics = 10, vocabulary = pruned_vocab, doc_topic_prior = 0.1, topic_word_prior = 0.01)
doc_topic_distr = lda_model_amazon$fit_transform(dtm_train_lda, n_iter = 1000, convergence_tol = 0.01, 
                                                 check_convergence_every_n = 10)

install.packages("LDAvis")
library("LDAvis")
lda_model_amazon$plot()


dtm_mat_train<- as.data.frame(as.data.table(dtm_mat_train))
  dtm_mat_test<- as.data.frame(as.data.table(dtm_mat_test))

traininginput <-  as.data.frame(as.data.table(dtm_mat_train))
testinginput <- as.data.frame(as.data.table(dtm_mat_test))

training <- cbind(dtm_mat_train,as.factor(train[,train$polarity]))
testing <- cbind(dtm_mat_test,as.factor(test[,test$polarity]))

colnames(training)[1027]<-'polarity'
colnames(testing)[1027]<-'polarity'

##################################################################################33333
#### DNN - Deep Belief Network


traininginput1 <- as.data.frame(training[1:1026])
trainingoutput1 <- as.data.frame(training[1027])
testinginput1 <- as.data.frame(testing[1:1026])
testingoutput1 <- as.data.frame(testing[1027]) 

  
darch <- darch(traininginput1, trainingoutput1,
               rbm.numEpochs = 3,
               rbm.trainOutputLayer = F,
               layers = c(1026,4,1), # 1026 input 4 hidden layers and 1 output neurons
               darch.fineTuneFunction = backpropagation,
               darch.batchSize = 1,
               darch.bootstrap = F,
               darch.learnRateWeights = 1,
               darch.learnRateBiases = 1,
               darch.initialMomentum = .9,
               darch.finalMomentum = .9,
               darch.isBin = T,
               darch.stopClassErr = 0,
               darch.numEpochs = 493,
               gputools = F)

print(darch)

predictions <- predict(darch, newdata=testinginput1, type="class")

numIncorrect <- sum(predictions!= testingoutput1$polarity)
cat(paste0("Incorrect classifications on all examples: ", numIncorrect, " (",
           round(numIncorrect/nrow(testinginput1)*100, 2), "%)\n"))

numCorrect <- sum(predictions == testingoutput1$polarity)
cat(paste0("Correct classifications on all data: ", numCorrect, "
           ,→ (", round(numCorrect / nrow(testinginput1) * 100, 2), "%)\n"))


head(predictions)
#confusion matrix
table(pred=predictions,true=testingoutput1$polarity)

## Create confusion matrix based on actual response variable and predicted value.
caret::confusionMatrix(data=predictions,
                testingoutput1$polarity,
                positive='yes')


######################################################################### SVM
#### SVM ####

#install.packages("RTextTools")
library(RTextTools)


# dtm_train
# 
# dtMatrix_train <- create_matrix(train$text)
# # Configure the training data
# 
# class(train$polarity)

#1st iteration: Apply svm on entire train dataset to get initial values of kernal, cost and gamma
model1 = svm(training$polarity ~ ., data = training, scale = F)
summary(model1)


#2nd iteration: Apply value of cost as 1 and gamma as 0.0009746589
model2 = svm(training$polarity ~ ., kernel = "radial",cost = 1, gamma = 0.0009746589, data = training, scale = F)
summary(model2)

predictions <-  predict(model2, testing[,1:1026])
table(predictions,testing[,1027])
caret::confusionMatrix(data=predictions,
                       testing$polarity,
                       positive='yes')

svm.tune <- best.tune(svm,polarity ~ .,data = training,kernel="radial",scale=false)
print(svm.tune)


# Tried Tuning SVM to find the best cost and gamma but insted of classification regression was happening..
#tired converting data into numeric format for better tuning but it was taking long process time
c= seq(0,3, by=1)

# for tuning the SVM
tuneResult <- tune(svm, Response ~ .,  data = train_svm,ranges = list(gamma = seq(0,0.2,0.01), cost=10^(-1:1)))

svm_tune <- tune(svm, polarity ~ .,data = training, scale = F, kernel="radial", 
                 ranges = list(gamma = seq(0,0.2,0.01), cost=10^(-1:1)))
print(svm_tune)
train_dateset[10]
svm_model_after_tune <- svm(polarity ~ ., data=training, kernel="radial", cost=10, gamma=0.01, scale = F)
summary(svm_model_after_tune)


predictions <-  predict(svm_model_after_tune, testing[,1:1026])
table(predictions,testing[,1027])
#confusion matrix for the final tuned SVM model
caret::confusionMatrix(data=predictions,
                       testing$polarity,
                       positive='yes')
