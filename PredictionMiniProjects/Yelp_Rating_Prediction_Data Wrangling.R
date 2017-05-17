#install.packages("rjson")
#install.packages("irlba")
#install.packages("RJSONIO")

library("rjson")
library("irlba")
library(RJSONIO)
library(tidyr)

setwd("/Yelp Dataset/")

# extracting Business JSON data into a dataset
Lines <- readLines("business.json") #
business <- as.data.frame(t(sapply(Lines, fromJSON)))
business <- as.data.frame(business)
#Replacing index nos by appropriate nos
rownames(business) <- 1:nrow(business)
#View(as.data.frame(business))

#Splitting the Address Filed to StreetName and (State Zipcode)
businessData <- separate(data= business, col = full_address, into = c("streetName","StateFrmAddress"), sep = ",")

#Extracting ZipCode from StteFrmAddress and integrating it with the dataset
x <- businessData$StateFrmAddress
matches <- regmatches(x, gregexpr("[[:digit:]]+", x))
#matches
businessData$zipCode <- matches
businessData <- subset(businessData, nchar(zipCode)==5 )
nrow(businessData)

# extracting Business JSON data into a dataset
Lines <- readLines("user.json") #
userData <- as.data.frame(t(sapply(Lines, fromJSON)))
rownames(userData) <- 1:nrow(userData)
#View(as.data.frame(userData))

# extracting Business JSON data into a dataset
Lines <- readLines("review.json") #
reviewData <- as.data.frame(t(sapply(Lines, fromJSON)))
rownames(reviewData) <- 1:nrow(reviewData)
#View(as.data.frame(reviewData))

# extracting Checkin JSON data into a dataset
Lines <- readLines("checkin.json") #
checkinData <- as.data.frame(t(sapply(Lines, fromJSON)))
rownames(checkinData) <- 1:nrow(checkinData)
#View(as.data.frame(checkinData))

#unlist the category from the business data
categ <- unlist(businessData$categories, recursive = T)
categ <- as.data.frame(categ)

#calculated the frequency of the category
freq <- as.data.frame(table(categ))
#View(freq)

#we have more around 600 categories
#remove the categories which has frequency less than 25
vec <- freq[which(freq$Freq > 25),]
vec <- vec$categ

#dropping the levels of the category
vec <- droplevels(vec)
vec <- as.vector(vec)
#View(as.data.frame(vec))
#class(vec)

#list of important categories based on which prediction model will be trained
vec <- c("Restaurants", "Food", "Mexican", "Nightlife", "Active Life", "Bars", "Sandwiches", "American (Traditional)",
         "Fast Food", "Pizza", "American (New)", "Coffee & Tea", "Burgers", "Italian", "Chinese", "Hotels", "Breakfast & Brunch", 
         "Ice Cream & Frozen Yogurt", "Specialty Food", "Bakeries", "Convenience Stores", "Delis", "Sports Bars", "Japanese", 
         "Sushi Bars", "Mediterranean", "Steakhouses", "Barbeque", "Desserts", "Tex-Mex", "Cafes", "Chicken Wings", "Seafood",
         "Beer, Wine & Spirits", "Thai", "Greek", "Donuts", "Hot Dogs", "Music Venues", "Juice Bars & Smoothies", "Diners", "Bagels",
         "Vietnamese", "Salad", "Wine Bars", "Pubs", "Dive Bars", "Vegetarian", "Local Flavor", "Gluten-Free", "Middle Eastern", "Ethnic Food")

#keeping the row number count which business belong to
#the the categories which we have mentioned above
indx <- c()
for(i in 1:nrow(businessData)) {
  a <- businessData$categories[i]
  
  indx <-  c(indx, any(vec %in% a[[1]]))
}

#View(as.data.frame(indx))
#indx[which(indx == FALSE)]

#creating a matrix which has those categories as column
trans <- matrix(0, ncol = 53, nrow = nrow(businessData))
colnames(trans) <- c(vec, "Type")
trans <- as.data.frame(trans)
#View(trans)

#put values as 1 for those business who has the categories 
#mentioned in the list
for(i in 1:nrow(businessData)) {
  temp <- businessData$categories[i]
  x <- which(vec %in% temp[[1]])
  
  for(j in 1:length(x)) {
    trans[i, x[j]] = 1
  }
  
  trans$Type[i] = length(x)
}

#started the cleaning of userData
userData$funny <- NULL
userData$useful <- NULL
userData$cool <- NULL

#the votes categorization is changed from list to columns
#there are three categories funny, useful and cool
for(i in 1:nrow(userData)) {
  a <- userData$votes[i]
  userData$funny[i] <- a[[1]]["funny"]
  userData$useful[i] <- a[[1]]["useful"]
  userData$cool[i] <- a[[1]]["cool"]
}

#drop the columns like count, name of the user and how many friends user have
#as these columns are not so useful in training the model
drop.names <- c("votes", "name", "friends", "type", "compliments", "elite", "yelping_since")
userData <- userData[,!(names(userData) %in% drop.names)] 

#start cleaning of the review data
reviewData$funny <- NULL
reviewData$useful <- NULL
reviewData$cool <- NULL

#the votes categorization is changed from list to columns
#there are three categories funny, useful and cool
for(i in 1:nrow(reviewData)) {
  a <- reviewData$votes[i]
  reviewData$funny[i] <- a[[1]]["funny"]
  reviewData$useful[i] <- a[[1]]["useful"]
  reviewData$cool[i] <- a[[1]]["cool"]
}

#date of reviewData has been divided 
#into month, date and yaer
reviewData$date <- as.character(reviewData$date)
reviewData$date <- ymd(reviewData$date)
reviewData$Day <- (format(reviewData$date, "%d"))
reviewData$Year <- format(reviewData$date, "%Y")
reviewData$months <- (format(reviewData$date, "%m"))

#votes and date column are removed from the review
reviewData <- reviewData[,!(names(reviewData) %in% c("votes", "date"))]

#dropping the columns from business data
drop.names <- NULL
drop.names <- c("streetName", "StateFrmAddress", "hours", "categories", "neighborhoods", "state", "attributes", "type")
businessData <- businessData[,!(names(businessData) %in% drop.names)] 

#convert user and review id as character
userData$user_id <- as.character(userData$user_id)
reviewData$user_id <- as.character(reviewData$user_id)

#merge user and review file
user_review <- merge(x = userData, y = reviewData, by = "user_id", all = TRUE)

#convert user and merged dataset id as character
businessData$business_id <- as.character(businessData$business_id)
user_review$business_id <- as.character(user_review$business_id)

#merge user, review and business dataset
business_user <- merge(x = user_review, y = businessData, by = "business_id", all.y = TRUE)

NewBusiness <- read.csv("NewBusiness.csv")

#removing outliers
#plotting boxplot for checkin_count
boxplot(NewBusiness$checkin_count, main="Plot of Check-in Count")

#247 rows has been removed
NewBusiness <- NewBusiness[-which(NewBusiness$checkin_count > 6000),]

#plotting boxplot for review_count
boxplot(NewBusiness$review_count, main="Plot of Review Count")

#plotting boxplot for Latitude
boxplot(NewBusiness$latitude, main="Plot of Latitude")

#plotting boxplot for Longitude
boxplot(NewBusiness$longitude, main="Plot of Longitude")

#plotting boxplot for stars
boxplot(NewBusiness$stars, main="Plot of Stars")

