#Data Wrangling of cleanning Dataset 1 - Energy
energy <- read.csv("E:\\Projects\\EnerNoc Energy Project\\rawdata.csv")
#View(energy)

df <- NULL
# ?sapply
### we check for the kwh values in the column
# 
# Description
# 
# lapply returns a list of the same length as X, each element of which is the result of applying FUN to the corresponding element of X.
# 
# sapply is a user-friendly version and wrapper of lapply by default returning a vector, matrix or, if simplify = "array", an array if appropriate, by applying simplify2array(). sapply(x, f, simplify = FALSE, USE.NAMES = FALSE) is the same as lapply(x, f).
# 
# vapply is similar to sapply, but has a pre-specified type of return value, so it can be safer (and sometimes faster) to use.
# 
# replicate is a wrapper for the common use of sapply for repeated evaluation of an expression (which will usually involve random number generation).
# 
# simplify2array() is the utility called from sapply() when simplify is not false and is similarly called from mapply().
# 
# Usage
# 
# lapply(X, FUN, ...)
# 
# sapply(X, FUN, ..., simplify = TRUE, USE.NAMES = TRUE)
# 
# vapply(X, FUN, FUN.VALUE, ..., USE.NAMES = TRUE)
# 
# replicate(n, expr, simplify = "array")
# 
# simplify2array(x, higher = TRUE)

kWHList <- sapply (energy$Units, function(x) x == 'kWh')
head(kWHList)
## for the kwh values, we keep the rows
df <- energy[kWHList,]

nrow(df)
View(df)

#Transforming and changing
x <- df
#View(x)
## we want each day data in one column
trans <- t(x[4:292])
View(trans)

#install.packages("reshape")
library("reshape")
?melt
# Melt
# 
# Description
# 
# Melt an object into a form suitable for easy casting.
# 
# Usage
# 
# melt(data, ...)
## first we put all the 00:05 time first then 00:10 time second....
md <- melt(df, id=(c("Account", "Date", "Channel", "Units")))
## now we order by date, so first 12 * 24 records are for date 1, next 12 * 24 for Date 2...
md <- md[order(md$Date),]
View(md)

#105120
#8760
#hourlySum$kWh <- aggregate(md$value, by=list(), sum, na.rm = FALSE)

k <- 1
hour <- 0
hourlySum <- NULL
modified <- NULL
## now we will calcualte the kwh values hourly basis, hour, peakhour
for(i in seq(from=1, to=nrow(md), by=12)) {
  
  j <- i+11;
  
  if(hour == 24)
    hour = 0;
  
  if(hour > 6 && hour < 20)
    hourlySum$Peakhour[k] <- 1
  else
    hourlySum$Peakhour[k] <- 0
  
  hourlySum$Hour[k] <- hour;
  hourlySum$kWh[k] <- sum(md$value[i:j])
  #hold the requried rows and delete the rest
  modified <- rbind(modified, md[i,])
  
  hour <- hour + 1;
  k <- k + 1;
}

View(hourlySum)
View(modified)

#Consolidating the dataset hourly data with other fields
energy <- cbind(modified, hourlySum)
View(energy)
head(energy)
# Account     Date         Channel Units variable value Peakhour Hour    kWh
# 1     26435791004 1/1/2014 605105283 1 kWh   kWh    X0.05 7.440        0    0 81.285
# 4381  26435791004 1/1/2014 605105283 1 kWh   kWh    X1.05 6.975        0    1 81.705
# 8761  26435791004 1/1/2014 605105283 1 kWh   kWh    X2.05 7.395        0    2 81.510
# 13141 26435791004 1/1/2014 605105283 1 kWh   kWh    X3.05 6.945        0    3 83.115
# 17521 26435791004 1/1/2014 605105283 1 kWh   kWh    X4.05 7.320        0    4 83.835
# 21901 26435791004 1/1/2014 605105283 1 kWh   kWh    X5.05 7.365        0    5 85.530


#splitting the date
library(lubridate)
energy$Date <- as.Date(energy$Date, "%m/%d/%Y")
head(energy$Date)
energy$month <- month(energy$Date)
energy$day <- day(energy$Date)
energy$year <- year(energy$Date)
head(energy)
### For the day of the weekday
energy["Day of Week"] <- wday(energy$Date) - 1;

#Weekday
#install.packages("timeDate")
library("timeDate")
# if a day is a weekday (1) or not (0)
weekdayList <- sapply(energy$Date, function(x) {
  if(isWeekday(x, wday=1:5)) {
    energy["Weekday"] <- 1
  } else  {
    energy["Weekday"] <- 0 
  }
})
energy$Weekday <- weekdayList
#View(energy$Weekday)

#energy["Weekday"] <- NULL
#energy["Weekday"] <- factor(isWeekday(energy$Date, wday = 1:5), labels = c("1", "0"), levels = c(TRUE, FALSE))
head(energy)
# Account       Date         Channel Units variable value Peakhour Hour    kWh month day year
# 1     26435791004 2014-01-01 605105283 1 kWh   kWh    X0.05 7.440        0    0 81.285     1   1 2014
# 4381  26435791004 2014-01-01 605105283 1 kWh   kWh    X1.05 6.975        0    1 81.705     1   1 2014
# 8761  26435791004 2014-01-01 605105283 1 kWh   kWh    X2.05 7.395        0    2 81.510     1   1 2014
# 13141 26435791004 2014-01-01 605105283 1 kWh   kWh    X3.05 6.945        0    3 83.115     1   1 2014
# 17521 26435791004 2014-01-01 605105283 1 kWh   kWh    X4.05 7.320        0    4 83.835     1   1 2014
# 21901 26435791004 2014-01-01 605105283 1 kWh   kWh    X5.05 7.365        0    5 85.530     1   1 2014
# Day of Week Weekday
# 1               3       1
# 4381            3       1
# 8761            3       1
# 13141           3       1
# 17521           3       1

#Removing unneccesary column
energy <- energy[ , -which(names(energy) %in% c("Channel","Units", "variable", "value"))]

#Finding and replacing NA values with Zero from kWh
for(i in 1:nrow(energy)) {
  a <- is.na(energy$kWh)
  
  if(is.na(energy$kWh[i]))
    energy$kWh[i] <- energy$kWh[i-1];
}

#Detecting the outliers in kWH
sd(energy$kWh)
for(i in 1:nrow(energy)) {
  if((energy$kWh[i]) < (mean(energy$kWh) - (1.5)*sd(energy$kWh)) &&
     (energy$kWh[i]) > (mean(energy$kWh) + (1.5)*sd(energy$kWh))) {
    
    b <- energy[i,]
    #energy$kWh[i] <- mean(energy$kWh);
    energy$kWh[i] <- energy$kWh[i-1];
  }
}

#finding the minimum but greater tha zero
minkWh <- energy[energy$kWh > min(energy$kWh),]
minkWh <- min(minkWh$kWh)

max(energy$kWh)

#Log Transformation for dealing with thousands of zero's
energy$kWh <- sapply(energy$kWh, function(x) {if(x == 0) { x <-log1p(minkWh) } else { x <- x}})


#Display energy dataset
View(energy)
View(summary(energy))

#write clean energy dataset to csv file
#write.csv(energy, file="Output/energy.csv")


##Dataset 2 - Temperature (Pulling from weatherData)
install.packages("weatherData")
#remove.packages("weatherData")
install.packages("devtools")
install_github("ozagordi/weatherData")
library("devtools")
library(weatherData)
getStationCode("Boston")

tempData <- getWeatherForDate("KBOS", start_date=min(energy$Date),
                              end_date = max(energy$Date),
                              opt_detailed=T,
                              opt_custom_columns=T, custom_columns=c(2))

#View(tempData)
temperature <- tempData
#View(temperature)

tempData <- NULL
tempData <- temperature

library("lubridate")
tempData$Date <- NULL
tempData$hour <- NULL
tempData$Date <- as.Date(tempData$Time, "%Y-%m-%d", tz = "EST")
tempData$hour <- hour(tempData$Time)
tempData <- tempData[order(tempData$Date),]
#View(tempData)

aggdata <- aggregate(tempData$TemperatureF, by=list(tempData$Date, tempData$hour), mean, na.rm = FALSE)
aggdata <- aggdata[order(aggdata$Group.1),]
View(aggdata)
#displaying the dataset
#View(aggdata)
#View(summary(aggdata))

#removing the outlier
#10th October 2014, 4th Hour
aggdata[which(aggdata$x < 0),]
aggdata <- aggdata[-which(aggdata$x < 0),]

#aggdata <- reviseTemp

View(summary(aggdata))

#Finding and replacing NA values with Zero from kWh
#naList <- sapply (aggdata$x, function(x) is.na(x))
#aggdata[naList,]

for(i in 1:nrow(aggdata)) {
  a <- is.na(aggdata$x)
  
  if(is.na(aggdata$x[i]))
    aggdata$x[i] <- aggdata$x[i-1];
}

#Detecting the outliers in Temperature
for(i in 1:nrow(aggdata)) {
  if((aggdata$x[i]) < (mean(aggdata$x) - (1.5)*sd(aggdata$x)) &&
     (aggdata$x[i]) > (mean(aggdata$x) + (1.5)*sd(aggdata$x))) {
    b <- aggdata[i,]
    aggdata$x[i] <- aggdata$x[i-1];
  }
}

#renaming the columns to the desired value
colnames(aggdata) <- c("Date", "Hour", "Temp");

#View(aggdata)

#write clean temp to csv file
#write.csv(aggdata, file="Output/temperature.csv")


##Merging of Energy and Temperature
library(dplyr)
consolidate <- left_join(energy,aggdata, by = c("Date"="Date", "Hour"= "Hour"))

#plot(consolidate$kWh, consolidate$Temp)

## Working on energy for outlier
#View(energy)

#Outlier detected for 9th March, 2nd Hour
#energy[which(energy$kWh == ''),]

#Finding and replacing NA values with Zero from kWh
for(i in 1:nrow(consolidate)) {
  #a <- is.na(consolidate$kWh)
  if(is.na(consolidate$kWh[i]))
    consolidate$kWh[i] <- consolidate$kWh[i-1]
  
  if(is.na(consolidate$Temp[i])) {
    #naMonth <- month(consolidate$Date[i])
    #consolidate %>% group_by(month) %>% filter(month == naMonth) %>% filter(kWh > 350)
    #consolidate$Temp[i] <- mean(consolidate$Temp, na.rm = TRUE)
    consolidate$Temp[i] <- consolidate$Temp[i-1];
  }
}

#Detecting the outliers in kWH
for(i in 1:nrow(consolidate)) {
  if((consolidate$kWh[i]) < (mean(consolidate$kWh) - (1.5)*sd(consolidate$kWh)) &&
     (consolidate$kWh[i]) > (mean(consolidate$kWh) + (1.5)*sd(consolidate$kWh))) {
    #a <- consolidate[i,]
    #consolidate[i,5] <- mean(consolidate$kWh);
    consolidate$kWh[i] <- consolidate$kWh[i-1];
  }
}

#Detecting the outliers in Temperature
for(i in 1:nrow(consolidate)) {
  if((consolidate$Temp[i]) < (mean(consolidate$Temp) - (1.5)*sd(consolidate$Temp)) &&
     (consolidate$Temp[i]) > (mean(consolidate$Temp) + (1.5)*sd(consolidate$Temp))) {
    #b <- consolidate[i,]
    #consolidate[i,11] <- mean(consolidate$Temp);
    consolidate$Temp[i] <- consolidate$Temp[i-1];
  }
}

#Displaying the summary and dataset
View(consolidate)
View(summary(consolidate))

#Writing it to csv
write.csv(consolidate, file = "Output/merge.csv")

########
#install.packages("tree")
#install.packages("MASS")
#install.packages("ISLR")
#install.packages("mcclust")
#install.packages("readxl")
#install.packages("MASS")
#install.packages("grid")
#install.packages("neuralnet")
#install.packages("nnet")
#install.packages("clusterGeneration")

library (tree)
library (MASS)
library (ISLR)
library(mcclust)
library(readxl)
library (grid)
library (neuralnet)

#A. Regression Tree for prediction
#Reading the best "Filled Dataset"
best_dataset <- read.csv("/Users/hpanjwani/R Workspace/Assignment 2/Output/Hourly_filled_data.csv")
#View(best_dataset)

set.seed (1)
train = sample (1:nrow(best_dataset), nrow(best_dataset)/2)

tree.best_dataset = tree(formula = kWh ~   as.numeric(month) + as.numeric(day)  + as.numeric(Day.of.Week) + as.numeric(Weekday) + as.numeric(hour) + as.numeric(Peakhour) + as.numeric(Temp),best_dataset,subset=train)
summary (tree.best_dataset)
plot (tree.best_dataset)
text (tree.best_dataset, pretty = 0)
cv.boston = cv.tree (tree.best_dataset)
plot (cv.boston$size, cv.boston$dev, type='b')

yhat=predict(tree.best_dataset, newdata = best_dataset [-train,])
#View(as.data.frame(yhat))
newdata <- best_dataset [-train,"kWh"]
accuracy(yhat,newdata)

boston.test=best_dataset [-train,"kWh"]
plot(yhat,boston.test)
abline (0,1)
mean((yhat -boston.test)^2)

#not to be use without pruning best results are coming
#pruning of the tree
prune.boston =prune.tree(tree.best_dataset, best = 8)
plot(prune.boston)
text(prune.boston, pretty = 0)
cv.prune = cv.tree (prune.boston)
plot (cv.prune$size, cv.prune$dev, type='b')

yhat=predict (prune.boston, newdata =best_dataset [-train,])
boston.test=best_dataset [-train,"kWh"]
plot(yhat,boston.test)
abline (0,1)
mean((yhat -boston.test)^2)

#B. Neural Network for Prediction
neural_dataset <- read.csv("/Users/hpanjwani/R Workspace/Assignment 2/Output/Hourly_filled_data.csv")

#neural_dataset$Account <- as.numeric(neural_dataset$Account)
#neural_dataset$Date <- as.numeric(neural_dataset$Date)
#neural_dataset$year <- as.numeric(neural_dataset$year)
neural_dataset$month <- as.numeric(neural_dataset$month)
neural_dataset$day <- as.numeric(neural_dataset$day)
neural_dataset$Day.of.Week <- as.numeric(neural_dataset$Day.of.Week)
neural_dataset$Weekday <- as.numeric(neural_dataset$Weekday)
neural_dataset$hour <- as.numeric(neural_dataset$hour)
neural_dataset$Peakhour <- as.numeric(neural_dataset$Peakhour)
neural_dataset$Temp <- as.numeric(neural_dataset$Temp)
neural_dataset$kWh <- as.numeric(neural_dataset$kWh)
#View(neural_dataset)

drops <- c("Account","Date", "year")
dfnnet <- neural_dataset[ , !(names(neural_dataset) %in% drops)]
#View(dfnnet)

#Sampling the data
#normalizing the data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

#selecting my new subset
datannet_n <- as.data.frame(lapply(dfnnet, normalize))
#View(datannet_n)

train <- sample(1:nrow(datannet_n),round(0.75*nrow(datannet_n)))
traindata <- datannet_n[train,]
testdata <- datannet_n[-train,]
View(testdata)

#training neural net for th model
net.sqrt <- neuralnet(kWh ~ Peakhour + Day.of.Week  + hour + Weekday + Temp + day + month, data=traindata, hidden=c(7,5,5), threshold=0.5, linear.output = F)

#print the error result
print(net.sqrt)

#Plot the neural network
plot(net.sqrt)

#Test the neural network on some training data
acutal <- testdata$kWh
drops <- c("kWh")
testdata <- testdata[ , !(names(testdata) %in% drops)]
net.results <- compute(net.sqrt, testdata) #Run them through the neural network
#View(net.results)

#accuracy of the algorithm
accuracy(net.sqrt, acutal)

#Lets see the results
print(net.results$net.result)

#changing matrix to numeric
net.results$net.result <- as.numeric(net.results$net.result)

#denormalized the data
denormalize <- function(x) {
  #return (((x - min(x)) / (max(x) - min(x)))*((max(x)-min(x))+min(x)))
  return ((x * (max(neural_dataset$kWh) - min(neural_dataset$kWh))) + min(neural_dataset$kWh))
}

results <- as.data.frame(sapply(net.results$net.result, denormalize))
#View(results)


#computing performance of the algorithm
results <- as.numeric(results$`sapply(net.results$net.result, denormalize)`)
error = (results - neural_dataset$kWh)

# Function that returns Root Mean Squared Error
rmse <- function(error) {
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)  {
  mean(abs(error))
}

# Function that returns Mean Absolute Percentage Error
mape <- function(error)  {
  mean(abs(error/neural_dataset$kWh) * 100)
}

#calculating mean square value
rmse(error)
rms <- c("RMS", rmse(error))
mae(error)
ma <- c("MAE", mae(error))
mape(error)
map <- c("MAPE", mape(error))

neural_performance <- NULL
neural_performance <- rbind(rms, ma, map, deparse.level = 0)
neural_performance

#writing prediction performance metrics to csv files
account <- c("Account No", unique(neural_dataset$Account))
write.table(t(account), file="/Output/PredictionPerformanceMetrics.csv", row.names = F, col.names = F, qmethod = "double")

write.table("\n", file="/Output/PredictionPerformanceMetrics.csv", append = T, row.names = F, col.names = F)
write.table("Regression Tree", file="/Output/PredictionPerformanceMetrics.csv", append = T, row.names = F, col.names = F)

write.table(t(accuracy(yhat,newdata)), file="/Output/PredictionPerformanceMetrics.csv", append = T, col.names = F, qmethod = "double")

write.table("\n", file="/Output/PredictionPerformanceMetrics.csv", append = T, row.names = F, col.names = F)
write.table("Neural Network", file="/Output/PredictionPerformanceMetrics.csv", append = T, row.names = F, col.names = F)

write.table(neural_performance, file="/Output/PredictionPerformanceMetrics.csv", append = T, row.names = F, col.names = F, qmethod = "double")
