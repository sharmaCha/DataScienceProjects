setwd("/Downloads")
train<-read.csv("train.csv",stringsAsFactors = F)
test<-read.csv("test.csv",stringsAsFactors = F)

table(train$Surge_Pricing_Type) #Number_of_instances_in_each_category

test$Surge_Pricing_Type<-NA
train_test<-rbind(train,test) #Combining train & test to make preprocessing easier

summary(train_test)
##Treating categorical variables
train_test$Type_of_Cab<-as.factor(train_test$Type_of_Cab) #5categories
train_test$Confidence_Life_Style_Index<-as.factor(train_test$Confidence_Life_Style_Index)#3categories
train_test$Destination_Type<-as.factor(train_test$Destination_Type) #18categories
train_test$Gender<-as.factor(train_test$Gender)#2categories
#One hot encoding
library(dummies)
df <- dummy.data.frame(train_test, names=c("Gender"), sep="_")

##Oulier treatment
boxplot(train_test[,5])
hist(train_test[,5])
#2,4,**5 [2.2-3.5],8,**9 [max 4], **10[max 125], **11[max 65], **12[max 120]
train_test[,9]<-ifelse(train_test[,9]>4,4,train_test[,9]) #Capping
#train_test[,5]<-ifelse(train_test[,5]<2.2,2.2,train_test[,5]) #Capping_under

##Missing Value Treatment
apply(train_test, 2, function(x) any(is.na(x))) #Checking missing values in each features
#Type of Cab,  #Customer_Since_Months, #Life_Style_Index, #Var1
#By {Mean, Mode} OR {use gbm & rf to deal with these [BETTER]}
train_test$Type_of_Cab[train_test$Type_of_Cab==""]="B" #Replacing by mode
train_test$Customer_Since_Months[is.na(train_test$Customer_Since_Months)]=mean(train_test$Customer_Since_Months,na.rm=T)
train_test$Life_Style_Index[is.na(train_test$Life_Style_Index)]=mean(train_test$Life_Style_Index,na.rm=T)
train_test$Var1[is.na(train_test$Var1)]=mean(train_test$Var1,na.rm=T)

#Feature Engineering
  #Think + Google: when does ola/uber start surge pricing
     #1) Place of Booking
     #2) Time of Booking
     #3) Day of booking (Week Day)
     #4) Weather/Temperature
     #5) Peak Hours
     #6) Destination [Already covered] -> For destinations like airport, railway station candidate has to go so even if surge, no effect
     #7) Holidays/Events (Seasonality)
     #8) Location of pickup (Empty road will have surge pricing to attract more drivers)
  ##SurgePricing is DemandSupply Economics

#T#Correlation between categorical variables =>
#Categorical VS Categorical
#chi2 = chisq.test(train_test[,c(3,6,7,13)], correct=F)
#c(chi2$statistic, chi2$p.value)
#sqrt(chi2$statistic / sum(train_test[,c(3,6,7,13)]))

#Categorical VS Nueric
#aov1 = aov(train_independent_data ~ dependent_response)
#summary(aov1)

##Checking correlation between numeric variables
library(corrplot)
train_rm_na<-na.omit(train_test) #Removing test test and those instances of train set having NA
M <- cor(train_rm_na[,c(2,4,5,8,9,10,11,12)])#,14)]) #Only numeric variables are considered
corrplot(M,method="number")
##Var-2, Var-3 correlated .68
##Trip_Distance, LifeStyleIndex .47


train_test$Trip_ID=NULL
train_tot_df<-train_test[1:nrow(train),]
test_df<-train_test[(nrow(train)+1):nrow(train_test),]

##Sampling data into train_total into  train & validation set
library(caTools)
set.seed(101) 
sample = sample.split(train_tot_df, SplitRatio = .75)
train_df = subset(train_tot_df, sample == TRUE)
valid_df = subset(train_tot_df, sample == FALSE)



library(h2o)
library(data.table)
library(dplyr)

h2o.server <- h2o.init( nthreads= -1)

## Preprocessing the training data
#Converting all columns to factors
selCols = names(train_df)
#train_1 = train_df[,(selCols) := lapply(.SD, as.factor), .SDcols = selCols]

testHex = as.h2o(test_df)
train_score1=as.factor(train_df$Surge_Pricing_Type)
train_1 = cbind(train_df,Y=train_score1)
valid_score1=as.factor(valid_df$Surge_Pricing_Type)
valid_1 = cbind(valid_df,Y=valid_score1)
#Converting to H2o Data frame & splitting
train.hex1 = as.h2o(train_1)
validHex1 = as.h2o(valid_1)
features=names(train.hex1)[-c(13,14)]#Removing Surge_Pricing_Type,Y i.e. the dependent variables

gbmF_model_1 = h2o.gbm( x=features,
                        y = "Y",
                        training_frame =train.hex1 ,
                        validation_frame =validHex1 ,
                        max_depth = 5,
                        #distribution = "bernoulli",
                        ntrees =1000,
                        learn_rate = 0.05
                        #,nbins_cats = 5891
)

summary(gbmF_model_1)
#Variable Importances, RMSE from Validation set: Obtained from here


rf_model_1 =h2o.randomForest(x=features,y="Y",training_frame =train.hex1,
                             validation_frame =validHex1,
                             ntrees=3000,max_depth = 6)
summary(rf_model_1)
#Variable Importances, RMSE from Validation set: Obtained from here


dl_model_1 = h2o.deeplearning( x=features,
                               # x=features,
                               y = "Y",
                               training_frame =train.hex1 ,
                               validation_frame =validHex1 ,
                               activation="Rectifier",
                               hidden=80,
                               epochs=50,
                               adaptive_rate =F
)

summary(dl_model_1)
#Variable Importances, RMSE from Validation set: Obtained from here


test_pred_score1 = as.data.frame(h2o.predict(gbmF_model_1, newdata =testHex ,type="") )
pred1_1 = test_pred_score1
###ans<-lapply(pred1_1,function(x) ifelse(x<0,mean(train$total_sales),x)) #Only when some negatives are predicted
answ<-pred1_1$predict
fl<-cbind(test$Trip_ID,answ)
colnames(fl)<-c("Trip_ID","Surge_Pricing_Type")
write.csv(file="test_submission_v5.csv",x=fl,row.names = F)


#ENSEMBLING models created so far
pred_gbm=as.data.frame(h2o.predict(gbmF_model_1, newdata =testHex ,type=""))
pred_gbm_res=pred_gbm$predict
pred_rf=as.data.frame(h2o.predict(rf_model_1, newdata =testHex ,type=""))
pred_rf_res=pred_rf$predict
pred_dl=as.data.frame(h2o.predict(dl_model_1, newdata =testHex ,type=""))
pred_dl_res=pred_dl$predict
ensem<-cbind(pred_gbm_res,pred_rf_res,pred_dl_res)
colnames(ensem)<-c("gbm","rf","dl")
predictions_majority_vote<-apply(ensem,1, function(x) tail(names(sort(table(x))), 1))

#Seperate model for each of two types and ensembing them  
train_df_12<-train[train$Surge_Pricing_Type %in% c("1","2"),]
train_df_23<-train[train$Surge_Pricing_Type %in% c("2","3"),]
train_df_13<-train[train$Surge_Pricing_Type %in% c("1","3"),]