Loan.Train <- read.csv("/Loan prediction 3/Loan-Train.csv")
View(Loan.Train)
attach(Loan.Train)

for (i in 1:614) {
  Loan.Train$TotalIncome[i]=Loan.Train$ApplicantIncome[i]+Loan.Train$CoapplicantIncome[i]  
}
hist(Loan.Train$TotalIncome)

TotalIncome1=log(Loan.Train$TotalIncome)
hist(TotalIncome1)
Loan.Train$TotalIncome=TotalIncome1

LoanAmount1=Loan.Train$LoanAmount[which(!is.na(Loan.Train$LoanAmount))]
ApplicantIncome1=Loan.Train$ApplicantIncome
CoapplicantIncome1=Loan.Train$CoapplicantIncome
Loan_Amount_Term1=Loan.Train$Loan_Amount_Term[which(!is.na(Loan.Train$Loan_Amount_Term))]

plot(Loan.Train$LoanAmount,ApplicantIncome1)
plot(Loan.Train$LoanAmount,CoapplicantIncome1)
plot(log(Loan.Train$LoanAmount),Loan.Train$TotalIncome)
plot((Loan.Train$LoanAmount),exp(Loan.Train$TotalIncome))

Loan.Train$Income=exp(Loan.Train$TotalIncome)
View(Loan.Train)

indicator=function(t) {
  Loan.Train$Income=dim(length(t))
  Loan.Train$Income[which(!is.na(t))]=1
  Loan.Train$Income[which(is.na(t))]=0
  return(Loan.Train$Income)
}
Loan.Train$Ind=indicator(Loan.Train$LoanAmount)

regr=lm(Loan.Train$LoanAmount~Loan.Train$Income, data = Loan.Train)
summary(regr)

for (i in 1:614) {
  if (Loan.Train$Ind[i]==0)
  {
    Loan.Train$LoanAmount[i]=8.872e+01+8.186e-03*Loan.Train$Income[i]
  }
}

cor(Loan.Train$Income,Loan.Train$LoanAmount)
hist(Loan.Train$LoanAmount)
LogLoanAmount=log(Loan.Train$LoanAmount)
hist(LogLoanAmount)

Loan.Train$Amount= LogLoanAmount 

View(Loan.Train)

summary(Loan.Train$Credit_History)
Loan.Train$Credit_History[which(is.na(Loan.Train$Credit_History))]=median(Loan.Train$Credit_History)

summary(Loan.Train$Loan_Amount_Term)
Loan.Train$Loan_Amount_Term[which(is.na(Loan.Train$Loan_Amount_Term))]= 360

summary(Loan.Train$Self_Employed)
Loan.Train$Self_Employed[Loan.Train$Self_Employed==""] <- NA
Loan.Train$Self_Employed[which(is.na(Loan.Train$Self_Employed))]='No'

Loan.Train$Dependents[Loan.Train$Dependents=="3+"] <- 3
Loan.Train$Dependents[which(is.na(Loan.Train$Dependents))]= 3
Loan.Train$Dependents[Loan.Train$Dependents==""] <- 0
summary(Loan.Train$Dependents)
Loan.Train$Dependents[is.na(Loan.Train$Dependents)]
Loan.Train$Dependents[Loan.Train$Dependents=="<NA>"] <- '3+'
Loan.Train$Dependents[Loan.Train$Dependents != (0 | 1 | 2)] <- 3

summary(Loan.Train$Married)
Loan.Train$Married[Loan.Train$Married==""] <- NA
Loan.Train$Married[which(is.na(Loan.Train$Married))]= 'Yes'

summary(Loan.Train$Gender)
Loan.Train$Gender[Loan.Train$Gender==""] <- NA
Loan.Train$Gender[which(is.na(Loan.Train$Gender))]= 'Male'

write.table(Loan.Train, "d:/Loan_Train.txt", sep=",")

Training_Loan <- read.csv("/Downloads/AV/Loan prediction 3/Training_Loan.csv")
View(Training_Loan)
attach(Training_Loan)
summary(Training_Loan)

#Random Forests-81.3% accuracy in validation

install.packages("randomForest")
library(randomForest)

#Define the data frame
df =data.frame(Training_Loan)

#Building and Applying the Model
#set random seed
set.seed(123)

attach(df)

#Partition the data into training and testing 
train=sample (1:nrow(df), 400)

#Define the test frame
df.validation=df[-train,]

Loan_Status.validation=Loan_Status[-train] 

rf <- randomForest(factor(Loan_Status)~.,data=df,subset = train, n.trees=50,interaction.depth=5, importance = T)
rf <- randomForest(factor(Loan_Status)~.,data=df,subset = train, n.trees=75,interaction.depth=6, importance = T)
rf <- randomForest(factor(Loan_Status)~.,data=df,subset = train, n.trees=100,interaction.depth=5, importance = T)

tree.pred=predict(rf,df.validation,type="class")

table(tree.pred,Loan_Status.validation) #confusion matrix

#Using the importance() function, we can view the importance of each variable
importance (rf) 

varImpPlot (rf)

#Logistic Regression-  accuracy in Validation
library("LOGIT", lib.loc="~/R/win-library/3.3")
library("glm2", lib.loc="~/R/win-library/3.3")
#Train the model using the training sets and check
#score
model.glm <-glm(Loan_Status~., data =Training_Loan, family = binomial)

logreg <- glm(train$Loan_Status ~ ., data = train,family='binomial')
summary(model.glm)
#Predict Output
predicted= predict(model.glm,df.validation)

predicted