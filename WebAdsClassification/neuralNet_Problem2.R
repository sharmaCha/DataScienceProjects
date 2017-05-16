#install.packages("MASS")
#install.packages("grid")
#install.packages("neuralnet")
install.packages("nnet")
install.packages("clusterGeneration")

library (MASS)
library (grid)
library (neuralnet)
library(nnet)


neural_dataset <- read.csv("/best_dataset.csv")


#neural_dataset$Account <- as.numeric(neural_dataset$Account)
#neural_dataset$Date <- as.numeric(neural_dataset$Date)
neural_dataset$month <- as.numeric(neural_dataset$month)
neural_dataset$day <- as.numeric(neural_dataset$day)
neural_dataset$year <- as.numeric(neural_dataset$year)
neural_dataset$Day.of.Week <- as.numeric(neural_dataset$Day.of.Week)
neural_dataset$Weekday <- as.numeric(neural_dataset$Weekday)
neural_dataset$hour <- as.numeric(neural_dataset$hour)
neural_dataset$Peakhour <- as.numeric(neural_dataset$Peakhour)
neural_dataset$Temp <- as.numeric(neural_dataset$Temp)
neural_dataset$kWh <- as.numeric(neural_dataset$kWh)

View(neural_dataset)

#Sampling the data
#normalizing the data
normalize2 <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

dfnnet <- subset(neural_dataset, select=-c(1,2,3,6))
#View(dfnnet)

#selecting my new subset
datannet_n <- as.data.frame(lapply(dfnnet, normalize2))
#View(datannet_n)

train <- sample(1:nrow(datannet_n),round(0.75*nrow(datannet_n)))
traindata <- datannet_n[train,]
testdata <- datannet_n[-train,]
View(testdata)
####################################################################
#nn <- neuralnet(
#       case~age+parity+induced+spontaneous,
#      data=infert, hidden=2, err.fct="ce",
#       linear.output=FALSE)


#n <- names(train)
#f <- model.matrix(paste("kWh ~", paste(n[!n %in% c("kWh","Day of Week")], collapse = " + ")))

#neuralnet(f,data=train,hidden=c(5,3),linear.output=T)
#View(traindata)

#train <- sample(1:nrow(neural_dataset),round(0.75*nrow(neural_dataset)))
#traindata <- neural_dataset[train,]
#testdata <- neural_dataset[-train,]

net.sqrt <- neuralnet(kWh ~ Peakhour + Day.of.Week  + hour + Weekday + Temp + day + month, data=traindata, hidden=c(7,5,4), threshold=0.9, linear.output = F)


#net.sqrt = neuralnet( kWh ~ traindata$Peakhour +  traindata$Temp + traindata$month + traindata$day + traindata$Day.of.Week, traindata, hidden=5, linear.output = FALSE ,threshold=0.01) 
#net.sqrt <- neuralnet(f,neural_dataset, hidden=c(5,5), threshold=0.1)
print(net.sqrt)

#Plot the neural network
plot(net.sqrt)

#Test the neural network on some training data
#testdata <- as.data.frame((1:10)^2) #Generate some squared numbers
net.results <- compute(net.sqrt, testdata[,-c(5)]) #Run them through the neural network
View(as.data.frame(net.results))

#Lets see what properties net.sqrt has
ls(net.results)

#Lets see the results
print(net.results$net.result)

#Lets display a better version of the results
cleanoutput <- cbind(testdata,sqrt(testdata),
                     as.data.frame(net.results$net.result))
colnames(cleanoutput) <- c("Input","Expected Output","Neural Net Output")
print(cleanoutput)
