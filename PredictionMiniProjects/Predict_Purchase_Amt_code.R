'''
Removing user id & product id for the time being
Gender, City Category, Maritial Status to be converted to 1-Hot encoded form
Age (in bins) to be mapped to intergers starting from 1,2...
Occupation? Need to figure out something or keep as it is
Stay_In_Current_City_Years: 4+ change to 4 and lets try (capping)
Other fileds lets keep as it as.

Product_Category_1 > Product_Category_2 > Product_Category_3 always.
t-test, ANOVA for categorial independent variables like product category we can find out their impact on the dependent variable
'''

train_data<-read.csv("E:\\train.csv")
test_data<-read.csv("E:\\test-comb.csv")

one_hot_gender<-with(train_data,data.frame(model.matrix(~Gender-1,train_data),User_ID,Product_ID))
table(train_data$Age)
age_recoded <- as.numeric(train_data$Age)
#0-17  18-25  26-35  36-45  46-50  51-55    55+
table(train_data$Occupation) #Let's keep occupation as it is

table(train_data$City_Category)#Let's convert it into one-hot encoded form
one_hot_city<-with(train_data,data.frame(model.matrix(~City_Category-1,train_data),User_ID,Product_ID))

Stay_In_Current_City_Years_capped<-ifelse(train_data$Stay_In_Current_City_Years=="4+",4,train_data$Stay_In_Current_City_Years)
train_data$Stay_In_Current_City_Years
typeof(Stay_In_Current_City_Years_capped)

one_hot_Marital_Status<-with(train_data,data.frame(model.matrix(~Marital_Status-1,train_data),User_ID,Product_ID))
table(train_data$Product_ID)

new_train_data<-cbind(one_hot_gender,age_recoded,train_data$Occupation,one_hot_city,Stay_In_Current_City_Years_capped,one_hot_Marital_Status,train_data$Product_Category_1,train_data$Product_Category_2,train_data$Product_Category_3,train_data$Purchase)
ready_train_data<-new_train_data[,-c(10,11,14,15)]



#####DID NOT WORK BEGIN####
library(DMwR)
knnOutput <- knnImputation(ready_train_data)

library(mice)
#Lets treat the missing values using mice
miceMod <- mice(ready_train_data[,12:13], method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
#t-test, ANOVA for categorial independent variables like product category we can find out their impact on the dependent variable
aov(as.numeric(ready_train_data$`train_data$Purchase`)~ready_train_data$GenderF,data=ready_train_data)
#####DID NOT WORK END####

#Replacing my mode
##ready_train_data$`train_data$Product_Category_2`[is.na(ready_train_data$Product_Category_2)]<-8
ready_train_data[,13][is.na(ready_train_data[,13])]<-8
ready_train_data[,14][is.na(ready_train_data[,14])]<-16






one_hot_gender<-with(test_data,data.frame(model.matrix(~Gender-1,test_data),User_ID,Product_ID))
##NEED to understand what is t-test & annova

table(test_data$Age)
age_recoded <- as.numeric(test_data$Age)
#0-17  18-25  26-35  36-45  46-50  51-55    55+
table(test_data$Occupation) #Let's keep occupation as it is

table(test_data$City_Category)#Let's convert it into one-hot encoded form
one_hot_city<-with(test_data,data.frame(model.matrix(~City_Category-1,test_data),User_ID,Product_ID))

Stay_In_Current_City_Years_capped<-ifelse(test_data$Stay_In_Current_City_Years=="4+",4,test_data$Stay_In_Current_City_Years)
test_data$Stay_In_Current_City_Years
typeof(Stay_In_Current_City_Years_capped)

one_hot_Marital_Status<-with(test_data,data.frame(model.matrix(~Marital_Status-1,test_data),User_ID,Product_ID))
table(test_data$Product_ID)

new_test_data<-cbind(one_hot_gender,age_recoded,test_data$Occupation,one_hot_city,Stay_In_Current_City_Years_capped,one_hot_Marital_Status,test_data$Product_Category_1,test_data$Product_Category_2,test_data$Product_Category_3)
ready_test_data<-new_test_data[,-c(10,11,14,15)]



##ready_test_data$`test_data$Product_Category_2`[is.na(ready_test_data$Product_Category_2)]<-8
##ready_test_data$`test_data$Product_Category_3`[is.na(ready_test_data$Product_Category_3)]<-16
ready_test_data[,13][is.na(ready_test_data[,13])]<-8





#Modelling with h2o as done by winning solution on this modifies dataset
library(h2o)
library(data.table)
library(dplyr)
#Read in the data. Using data.table significantly improves performance in all kind of 
# data munging activities
train <- ready_train_data
test <- ready_test_data

#Initialise H2o server with 3 threads. I have a 4 core processor, so keeping one for other
#activities (like browsing & FB!). Though not necessary for this data set :) 
#Download and run H2O (http://h2o.ai/)
h2o.server <- h2o.init( nthreads= -1)

## Preprocessing the training data

#Converting all columns to factors
#?# selCols = names(train)[1:11]
#?# train = train[,(selCols) := lapply(.SD, as.factor), .SDcols = selCols]

#Converting to H2o Data frame & splitting
train.hex <- as.h2o(train)

#features=names(train)[!names(train) %in% c("Purchase")]
features=c("User_ID","Product_ID")


gbmF_model_1 = h2o.gbm( x=features,
                        y = "train_data$Purchase",
                        training_frame =train.hex ,
                        #validation_frame =testHex ,
                        max_depth = 3,
                        distribution = "gaussian",
                        ntrees =500,
                        learn_rate = 0.05,
                        nbins_cats = 5891
)

gbmF_model_2 = h2o.gbm( x=features,
                        y = "train_data$Purchase",
                        training_frame =train.hex ,
                        #validation_frame =testHex ,
                        max_depth = 3,
                        distribution = "gaussian",
                        ntrees =430,
                        learn_rate = 0.04,
                        nbins_cats = 5891
)


dl_model_1 = h2o.deeplearning( x=features,
                               y = "train_data$Purchase",
                               training_frame =train.hex ,
                               #validation_frame =testHex ,
                               activation="Rectifier",
                               hidden=6,
                               epochs=60,
                               adaptive_rate =F
)


dl_model_2 = h2o.deeplearning( x=features,
                               # x=features,
                               y = "train_data$Purchase",
                               training_frame =train.hex ,
                               #validation_frame =testHex ,
                               activation="Rectifier",
                               hidden=60,
                               epochs=40,
                               adaptive_rate =F
)


dl_model_3 = h2o.deeplearning( x=features,
                               y = "train_data$Purchase",
                               training_frame =train.hex ,
                               #validation_frame =testHex ,
                               activation="Rectifier",
                               hidden=6,
                               epochs=120,
                               adaptive_rate =F
)



MySubmission = test[, c("User_ID", "Product_ID"), with = FALSE]
#test = test[,c("User_ID", "Product_ID") := NULL, with = FALSE]

#Converting all columns to factors
#?#selCols = names(test)
#?#test = test[,(selCols) := lapply(.SD, as.factor), .SDcols = selCols]

# Converting to H2o.DataFrame
test.hex  = as.h2o(test)

#Making the predictions
testPurchase_gbm_1 = as.data.frame(h2o.predict(gbmF_model_1, newdata = test.hex) )
testPurchase_gbm_2 = as.data.frame(h2o.predict(gbmF_model_2, newdata = test.hex) )

testPurchase_dl_model_1 = as.data.frame(h2o.predict(dl_model_3, newdata = test.hex) )
testPurchase_dl_model_2 = as.data.frame(h2o.predict(dl_model_3, newdata = test.hex) )
testPurchase_dl_model_3 = as.data.frame(h2o.predict(dl_model_3, newdata = test.hex) )

testPurchase_gbm_1$predict=ifelse(testPurchase_gbm_1$predict<0,0,testPurchase_gbm_1$predict)
testPurchase_gbm_2$predict=ifelse(testPurchase_gbm_2$predict<0,0,testPurchase_gbm_2$predict)

testPurchase_dl_model_1$predict=ifelse(testPurchase_dl_model_1$predict<0,0,testPurchase_dl_model_1$predict)
testPurchase_dl_model_2$predict=ifelse(testPurchase_dl_model_2$predict<0,0,testPurchase_dl_model_2$predict)
testPurchase_dl_model_3$predict=ifelse(testPurchase_dl_model_3$predict<0,0,testPurchase_dl_model_3$predict)


ans=0.3*(testPurchase_dl_model_1$predict)+
  0.15*(testPurchase_dl_model_2$predict)+
  0.25*(testPurchase_dl_model_3$predict)+
  0.1*(testPurchase_gbm_1$predict)+
  0.2*(testPurchase_gbm_2$predict)



#Final Submission
MySubmission$Purchase = ans


write.csv(ans, "E:\\submit.csv", row.names = F)
