Testing_Loan <- read.csv("/Loan prediction 3/Testing_Loan.csv")
View(Testing_Loan)
attach(Testing_Loan)

for (i in 1:367) {
  Testing_Loan$TotalIncome[i]=Testing_Loan$ApplicantIncome[i]+Testing_Loan$CoapplicantIncome[i]  
}
hist(Testing_Loan$TotalIncome)
Testing_Loan$TotalIncome=log(Testing_Loan$TotalIncome)

Testing_Loan$Income=exp(Testing_Loan$TotalIncome)

indicator=function(t) {
  Testing_Loan$Income=dim(length(t))
  Testing_Loan$Income[which(!is.na(t))]=1
  Testing_Loan$Income[which(is.na(t))]=0
  return(Testing_Loan$Income)
}
Testing_Loan$Ind=indicator(Testing_Loan$LoanAmount)

regr=lm(Testing_Loan$LoanAmount~Testing_Loan$Income, data = Testing_Loan)
summary(regr)

for (i in 1:367) {
  if (Testing_Loan$Ind[i]==0)
  {
    Testing_Loan$LoanAmount[i]=9.628e+01+6.288e-03*Testing_Loan$Income[i]
  }
}

cor(Testing_Loan$Income,Testing_Loan$LoanAmount)
hist(Testing_Loan$LoanAmount)
LogLoanAmount=log(Testing_Loan$LoanAmount)
hist(LogLoanAmount)

Testing_Loan$LoanAmount= LogLoanAmount 

View(Testing_Loan)

colnames(Testing_Loan)[colnames(Testing_Loan)=="LoanAmount"] <- "Amount"

Testing_Loan <- subset(Testing_Loan, select = -c(6,7))
Testing_Loan <- subset(Testing_Loan, select = -c(11,12))

write.table(Testing_Loan, "d:/Testing_Loan.txt", sep=",")