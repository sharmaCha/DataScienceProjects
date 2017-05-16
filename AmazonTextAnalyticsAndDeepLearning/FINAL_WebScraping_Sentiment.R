#### FINAL_WebScraping_Sentiment.R ##############
######### This Script has the Web Scraping and the Sentimental Analysis (Polarity and 
##Multi-level emotions analysis)

### we are using pacman package which has Rvest and other useful packages
#Parse html pages for amazon product reviews

web_scraper <- function(doc){
  
  if(!"pacman" %in% installed.packages()[,"Package"]) install.packages("pacman")
  pacman::p_load_gh("trinker/sentimentr")
  pacman::p_load(RCurl, XML, dplyr, stringr, rvest, audio)
  
  sec = 0
  
  #Remove all white space
  trim <- function (x) gsub("^\\s+|\\s+$", "", x)
  
  #Using CSS Selectorgadget figuring out which selectors has what data and getting that data now
  title <- doc %>%
    html_nodes("#cm_cr-review_list .a-color-base") %>%
    html_text()
  
  author <- doc %>%
    html_nodes(".review-byline .author") %>%
    html_text()
  
  date <- doc %>%
    html_nodes("#cm_cr-review_list .review-date") %>%
    html_text() %>% 
    gsub(".*on ", "", .)
  
  ver.purchase <- doc%>%
    html_nodes(".review-data.a-spacing-mini") %>%
    html_text() %>%
    grepl("Verified Purchase", .) %>%
    as.numeric()
  
  format <- doc %>% 
    html_nodes(".review-data.a-spacing-mini") %>% 
    html_text() %>%
    gsub("Color: |\\|.*|Verified.*", "", .)
  #if(length(format) == 0) format <- NA
  
  stars <- doc %>%
    html_nodes("#cm_cr-review_list  .review-rating") %>%
    html_text() %>%
    str_extract("\\d") %>%
    as.numeric()
  
  comments <- doc %>%
    html_nodes("#cm_cr-review_list .review-text") %>%
    html_text() 
  
  helpful <- doc %>%
    html_nodes(".cr-vote-buttons .a-color-secondary") %>%
    html_text() %>%
    str_extract("[:digit:]+|One") %>%
    gsub("One", "1", .) %>%
    as.numeric()
  
  df <- data.frame(title, author, date, ver.purchase, format, stars, comments, helpful, stringsAsFactors = F)
  
  return(df)
}


#install.packages("pacman")
pacman::p_load(XML, dplyr, stringr, rvest, audio)

#Remove all white space
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

#product code provided whose reviews are considered
#can be changed for any other product reviews
prod_code = "B00KC6I06S"
url <- paste0("https://www.amazon.com/dp/", prod_code)
doc <- read_html(url)

#obtain the text(title of the product) in the node, remove "\n" from the text, and remove white space
prod <- html_nodes(doc, "#title") %>% html_text() %>% gsub("\n", "", .) %>% trim()
prod
## [1] "Fire HD 6 Tablet, 6\" HD Display, Wi-Fi, 8 GB - Includes Special Offers, Black"

# since we are web scraping, we have to consider that the reviews are distributed in html pages.
#so we can decide on the number of pages (proportional to the number of reviews) to be scraped.
#since each page has 10 reviews, so if we scrape like 100 pages, it will be 1000 pages.
#Like this, we set the pages value.

pages <- 100

#getting all the required data from amazon website page by page
reviews_all <- NULL
for(page_num in 1:pages){
  url <- paste0("http://www.amazon.com/product-reviews/",prod_code,"/?pageNumber=", page_num)
  doc <- read_html(url)
  
  reviews <- web_scraper(doc)
  reviews_all <- rbind(reviews_all, cbind(prod, reviews))
}
#reviews_all

##All the reviews
# 
# title
# 1                                                                                Unreal performance for $99!  I couldn't be happier.
# 2                                         The single best value Fire in the entire line. Skip the Fire 7 and get the Fire 6 instead!
# 3                                                                                                  HD Fire 6 is great for the price!
# 4                                                                                                 I was skeptical as I like the feel
# 5                                        BEWARE - You don't really get that 4.5 GB of storage available.  You get considerably less.
# 6                                                                                     Buyer Beware, NOT as Durable as Amazon Claims!
#   7                                                                                                                  Ridiculously cute
# 8                                                                                                                         Five Stars
# 9                                                                                                          Amazon Blocks Chromecast!
#   10                                                                Incredible Price Performance Partner for Prime -- Well Done Amazon
# 11                                                                               Great Tablet for a Pocket; Not Expensive to Replace
# 12                                                                                   Not as "tough" as they would like you to think.
# 13                                                                                                      This tablet junkie loves it!
#   14                                                                                                                        Five Stars
# 15                                                                                                                      My Fire HD 6
# 16                          I have owned every Kindle since the original one and loved them all but this one greatly disappointed me
# 17                                                                                           If you love Kindles you will love the 6
# 18                                                                  Worth it for offline viewing of Amazon Prime TV Shows and Movies
# 19                                                                                             Great for the kids first kindle fire.
# 20                                                                   I love it! The size is perfect, very fast. I'm a happy customer
# 21                                                                                                                         fire hd 6
# 22                                                                                                                          Love it!
# 23                                                                                                            kindle fire hd 6 16 GB
# 24                                                                            love this product to the end, this is wh i love Amazon
# 25                                                                                                 I just can't stop playing with it
# 26                                                                                               Great upgrade from original fire HD
# 27                                        For a stand alone reader that would be fine. When a major selling point is full access ...


#Web Scraping Finished#


#############################################

library(plyr)
library(stringr)
library(ggplot2)
library (plyr)
library (stringr)
library(tm)
library(Rstem)
library(sentiment)
library(wordcloud)
library(RColorBrewer)

write.csv(reviews_all, file = "reviews_all.csv")

#Sentiment Analysis function that takes sentences and positive/negative words from dictionary and match them 
#together to obtain score by subtracting the sum of negative matches from the sum of positive matches
score.sentiment = function(sentences, PWords, NWords, .progress='none')  
{  
  
  #function that returns an array of scores, we use laply() function
  scores = laply(sentences, function(sentence, PWords, NWords) {  
    #data cleaning using gsub()  
    sentence = gsub('[[:punct:]]', '', sentence)  
    sentence = gsub('[[:cntrl:]]', '', sentence)  
    sentence = gsub('\\d+', '', sentence)  
    #enforce sentences to be in lower case and get rid of undefined characters
    sentence <- sapply(sentence,function(row) iconv(row, "latin1", "ASCII", sub=""))
    sentence = tolower(sentence)  
    #str_split to split words in reviews  
    word.list = str_split(sentence, '\\s+')  
    #unlist words, because list() is one level of hierarchy 
    words = unlist(word.list)  
    #compare with dictionary(positive and negative)  
    PMatch = match(words, PWords)  
    NMatch = match(words, NWords)  
    #matching words with both dictionaries
    PMatch = !is.na(PMatch)  
    NMatch = !is.na(NMatch)  
    #true-false will be treated as 1/0 by the function sum():  
    score = sum(PMatch) - sum(NMatch)  
    return(score)  
  }, PWords, NWords, .progress=.progress )  
  scores.df = data.frame(score=scores, text=sentences)  
  return(scores.df)  
} 

#scoring product reviews
setwd("C:/Users/Bashaer/Desktop")
#import word lists
PosDictionary = scan('positive_words.txt', what='character', comment.char=';')
NegDictionary = scan('negative_words.txt', what='character', comment.char=';')
#include some words to the both positive/negative dictionaries
PWords = c(PosDictionary, 'upgrade')
NWords = c(NegDictionary, 'wait','waiting')

#we take all the comments column and turn it into a factor
reviews_all$comments<-as.factor(reviews_all$comments)
#Score all product reviews by using the function score.sentiment
Reviews.scores = score.sentiment(reviews_all$comments, PWords,NWords, .progress='text')
path<-"C:/Users/Bashaer/Desktop"
write.csv(Reviews.scores,file="ReviewsScores.csv")
View(Reviews.scores)


#plotting a histogram for each review and the obtained scores
hist(Reviews.scores$score)

#categorization be emotions and polarity
#include all the product reviews
Reviews_txt<- reviews_all$comments
Reviews_txt
#prepare the reviews by getting rid of any punctuation and empty spaces and undefined symbols
Reviews_txt = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", Reviews_txt)
Reviews_txt = gsub("@\\w+", "", Reviews_txt)
Reviews_txt = gsub("[[:punct:]]", "", Reviews_txt)
Reviews_txt = gsub("[[:digit:]]", "", Reviews_txt)
Reviews_txt = gsub("http\\w+", "", Reviews_txt)
Reviews_txt = gsub("[ \t]{2,}", "", Reviews_txt)
Reviews_txt = gsub("^\\s+|\\s+$", "", Reviews_txt)

#function to handle any errors
try.error = function(x)
{  
  #dealing with missing values
  y = NA
  #tryCatch error
  try_error = tryCatch(tolower(x), error=function(e) e)
  if (!inherits(try_error, "error"))
    y = tolower(x)
  return(y)
}

#use the function try.error() with sapply to enforce lower case 
Reviews_txt = sapply(Reviews_txt, try.error)
#remove NAs from product reviews
Reviews_txt = Reviews_txt[!is.na(Reviews_txt)]
names(Reviews_txt) = NULL

#categorization by emotions and polarity
#calculate emotion of product reviews, we use classify_emotion function from Sentiment package
EmoClassification = classify_emotion (Reviews_txt, algorithm="bayes", prior=1.0)
#Emotion best fit
emotion = EmoClassification[,7]
#replace all NA's by the word "unknown"
emotion[is.na(emotion)] = "unknown"

#calculate polarity of product reviews, we use classify_polarity function from Sentiment package
PolClassification = classify_polarity (Reviews_txt, algorithm="bayes")
#polarity best fit
polarity = PolClassification[,4]

#results of both classifications are combined in a data frame
results.df = data.frame(text=Reviews_txt, emotion=emotion, polarity=polarity, stringsAsFactors=FALSE)
View(results.df)
write.csv(results.df, file="emotion-polarity.csv")


#plotting sentiment analysis on product reviews
#plot the emotions of each reviews - "joy", "sad" etc.
ggplot(results.df, aes(x=emotion)) +
  geom_bar(aes(y=..count.., fill=emotion)) +
  scale_fill_brewer(palette="Spectral") +
  labs(x="emotion categories", y="number of reviews", 
       title = "Sentiment Analysis of Reviews of A Product\n(classification by emotion)",
       plot.title = element_text(size=12))

#plot polarity of reviews
ggplot(results.df, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="Set1") +
  labs(x="polarity categories", y="number of reviews",
       title = "Sentiment Analysis of Reviews of A Product\n(classification by polarity)",
       plot.title = element_text(size=12))



### This Script has Web Scraping and Sentimental Analysis  Part of the Final Project
#### The Next Script (Final_Text2Vec_Models.R) contains the Code for the Documents to Vectorization
### conversion and the 3 different Models - 1. DBN/DNN, 2. LDA - Topic Modelling, 3. SVM
# Also the next script has the Performance Evaluation for each model.