setwd("E:/CTR/")
library(Hmisc); library(xgboost)

# For most variables, 5m train sample will be used to calcuate CTRs,
# Before using R "train5m.csv" needs to be created via perl or equivalent
# see later for 'key' site/app vars where we get CTRs of all train
df1 <- read.csv("Data/train5m.csv", head = TRUE,
                stringsAsFactors = FALSE, nrows = 5e6)

df2 <- read.csv("Data/test.csv", head = TRUE,
                stringsAsFactors = FALSE, nrows = 5e6)

train <- c(1:nrow(df1))
df2$click <- 0; df1$id <- NULL; df2$id <- NULL
df <- rbind(df1, df2); rm(df1, df2)
test <- c((max(train)+1):nrow(df))

# Basic variable manipulation/removal, both test and train,
# may not be optimal choices
df$type <- ifelse(df$app_id == 'ecad2386', 1, 0)
df$domain <- with(df, paste0(app_domain, site_domain))
df$category <- with(df, paste0(app_category, site_category))
df$as_id <- with(df, paste0(app_id, site_id))

df$size <- with(df, paste(C15, C16, sep = "x"))
df$UTCday <- as.integer(substr(df$hour, 5, 6))
df$UTCday <- ifelse(df$UTCday < 28, df$UTCday, df$UTCday - 7)
df$UTChour <- substr(df$hour, 7, 8)

df$app_id <- df$app_domain <- df$app_category <- NULL
df$site_id <- df$site_domain <- df$site_category <- NULL
df$C15 <- df$C16 <- NULL; df$id <- df$device_ip <- df$hour <- NULL

click <- df$click; df$click <- NULL

# Set all to factors; reduce number of levels for 'general' vars
for(v in c(1:ncol(df))) {df[, v] <- as.factor(df[, v])}
key <- match(c("domain", "as_id"), names(df))
gen <- setdiff(1:ncol(df), key)
for(v in gen) {df[, v] <- combine.levels(df[, v], 0.01)}

# Load CTR tables for 'key' variables, derived from all train data
load("Data/domain_table_100.RData")
load("Data/as_id_table_100.RData")

# Calculate CTRs by each factor level of training set,
# use two dataframes: train factors, train CTR; var names identical
train_fact <- df[train, ]

train_CTR <- train_fact
for(i in gen) {
train_CTR[ , i] <- ave(click[train], train_fact[ , i])
}

train_CTR$domain <- domain_table_100$CTR[match(train_fact$domain,
                                         domain_table_100$domain)]
train_CTR$as_id <- as_id_table_100$CTR[match(train_fact$as_id,
                                       as_id_table_100$as_id)]

# Lookup CTRs for each factor level of each test feature
test_fact <- df[test, ]

test_CTR <- test_fact
for(i in gen) {
test_CTR[, i] <- train_CTR[match(test_fact[, i], train_fact[, i]), i]
}

test_CTR$domain <- domain_table_100$CTR[match(test_fact$domain,
                                        domain_table_100$domain)]
test_CTR$as_id <- as_id_table_100$CTR[match(test_fact$as_id,
                                      as_id_table_100$as_id)]

# Fill most obvious NA value gaps with average CTR
test_CTR$C17[is.na(test_CTR$C17)] <- 0.17
test_CTR$C21[is.na(test_CTR$C21)] <- 0.17
test_CTR$domain[is.na(test_CTR$domain)] <- 0.17
test_CTR$as_id[is.na(test_CTR$as_id)] <- 0.17

# Specify logloss evaluation measure
llfun <- function(actual, prediction) {
  epsilon <- .000000000000001
  yhat <- pmin(pmax(prediction, epsilon), 1-epsilon)
  logloss <- -mean(actual*log(yhat) + (1-actual)*log(1 - yhat))
  return(logloss)
}

# Prepare to train xgboost model using subset of train_CTR table
t2m <- sample(nrow(train_CTR), 2e6)
dtrain <- xgb.DMatrix(as.matrix(train_CTR[t2m,]), label = click[t2m])
dtest <- xgb.DMatrix(as.matrix(test_CTR), label = click[test])

logregobj <- function(preds, dtrain){
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain){
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0))) / length(labels)
  err <- round(llfun(labels, exp(preds)/(1+exp(preds))), 4)
  return(list(metric = "error", value = err))
}

# Some overfitting with below parameters...
param <- list(max.depth = 10, eta = 0.05, colsample_bytree = 1,
              subsample = 0.5, base_score = -1.7, silent = 1)
watchlist <- list(train = dtrain)
n <- 140 # n of 80 almost as effective

# Main training for model m1
m1 <- xgb.train(param, dtrain, n, watchlist, logregobj, evalerror)
pred <- predict(m1, dtest)
pred <- exp(pred)/(1+exp(pred))
pred <- ifelse(pred < 0.02, 0.02, pred)

sub <- read.csv("Data/sampleSubmission.csv", head = TRUE, nrows = 5e6,
                colClasses = c("character", "numeric"))
sub$click <- pred
write.csv(sub, file = "sub_xgboost.csv", row.names = F)
