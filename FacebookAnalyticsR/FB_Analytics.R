install.packages("Rfacebook")
install.packages("RCurl")
library(Rfacebook)
library(RCurl)
fb_oauth <- fbOAuth(app_id="000000000000", app_secret="1111111111111111111111111",extended_permissions = TRUE)
me <- getUsers("me",token=fb_oauth, private_info=TRUE)
# got error so ran the below code
fbOAuth <- function(app_id, app_secret, extended_permissions=FALSE, legacy_permissions=FALSE, scope=NULL)
{
  ## getting callback URL
  full_url <- oauth_callback()
  full_url <- gsub("(.*localhost:[0-9]{1,5}/).*", x=full_url, replacement="\\1")
  message <- paste("Copy and paste into Site URL on Facebook App Settings:",
                   full_url, "\nWhen done, press any key to continue...")
  ## prompting user to introduce callback URL in app page
  invisible(readline(message))
  ## a simplified version of the example in httr package
  facebook <- oauth_endpoint(
    authorize = "https://www.facebook.com/dialog/oauth",
    access = "https://graph.facebook.com/oauth/access_token") 
  myapp <- oauth_app("facebook", app_id, app_secret)
  if (is.null(scope)) {
    if (extended_permissions==TRUE){
      scope <- c("user_birthday", "user_hometown", "user_location", "user_relationships",
                 "publish_actions","user_status","user_likes")
    }
    else { scope <- c("public_profile", "user_friends")}
    
    if (legacy_permissions==TRUE) {
      scope <- c(scope, "read_stream")
    }
  }
  
  if (packageVersion('httr') < "1.2"){
    stop("Rfacebook requires httr version 1.2.0 or greater")
  }
  
  ## with early httr versions
  if (packageVersion('httr') <= "0.2"){
    facebook_token <- oauth2.0_token(facebook, myapp,
                                     scope=scope)
    fb_oauth <- sign_oauth2.0(facebook_token$access_token)
    if (GET("https://graph.facebook.com/me", config=fb_oauth)$status==200){
      message("Authentication successful.")
    }
  }
  
  ## less early httr versions
  if (packageVersion('httr') > "0.2" & packageVersion('httr') <= "0.6.1"){
    fb_oauth <- oauth2.0_token(facebook, myapp,
                               scope=scope, cache=FALSE) 
    if (GET("https://graph.facebook.com/me", config(token=fb_oauth))$status==200){
      message("Authentication successful.")
    } 
  }
  
  ## httr version from 0.6 to 1.1
  if (packageVersion('httr') > "0.6.1" & packageVersion('httr') < "1.2"){
    Sys.setenv("HTTR_SERVER_PORT" = "1410/")
    fb_oauth <- oauth2.0_token(facebook, myapp,
                               scope=scope, cache=FALSE)  
    if (GET("https://graph.facebook.com/me", config(token=fb_oauth))$status==200){
      message("Authentication successful.")
    } 
  }
  
  ## httr version after 1.2
  if (packageVersion('httr') >= "1.2"){
    fb_oauth <- oauth2.0_token(facebook, myapp,
                               scope=scope, cache=FALSE)  
    if (GET("https://graph.facebook.com/me", config(token=fb_oauth))$status==200){
      message("Authentication successful.")
    } 
  }
  
  ## identifying API version of token
  error <- tryCatch(callAPI('https://graph.facebook.com/pablobarbera', fb_oauth),
                    error = function(e) e)
  if (inherits(error, 'error')){
    class(fb_oauth)[4] <- 'v2'
  }
  if (!inherits(error, 'error')){
    class(fb_oauth)[4] <- 'v1'
  }
  
  return(fb_oauth)
}

fb_oauth <- fbOAuth(app_id="000000000000000", app_secret="11111111111111",extended_permissions = TRUE)

# To see user profile name

me <- getUsers("me",token=fb_oauth, private_info=TRUE)

me$name

# List of all the pages you have liked

likes = getLikes(user="me", token = fb_oauth)
sample(likes$names, 10)

# Update Facebook Status from R

updateStatus("this is just a test", token=fb_oauth)

# Search Pages that contain a particular keyword

pages <- searchPages( string="analytics", token=fb_oauth, n=100)

colnames(pages)
head(pages$name)

# Extract list of posts from a Facebook page
page <- getPage(page="bbcnews", token=fb_oauth, n=100)
View(page)

# Get all the posts from a particular date
page <- getPage("bbcnews", token=fb_oauth, n=100,
                since='2016/06/01', until='2017/04/30')

# Which of these posts got maximum likes?

summary = page[which.max(page$likes_count),]
summary$message
# Which of these posts got maximum comments?

summary1 = page[which.max(page$comments_count),]
summary1$message

# Which post was shared the most?
summary2 = page[which.max(page$shares_count),]
summary2$message

# Extract a list of users who liked the maximum liked posts
post <- getPost(summary$id[1], token=fb_oauth, comments = FALSE, n.likes=2000)
likes <- post$likes
head(likes)

# Extract FB comments on a specific post
post <- getPost(page$id[1], token=fb_oauth, n.comments=1000, likes=FALSE)
comments <- post$comments
fix(comments)

# What is the comment that got the most likes?
comments[which.max(comments$likes_count),]

#Extract Reactions for most recent post

post <- getReactions(post=page$id[1], token=fb_oauth)
post

#Get Posts of a particular group
# Extract posts from Machine Learning Facebook group
ids <- searchGroup(name="machinelearningforum", token=fb_oauth)
group <- getGroup(group_id=ids[1,]$id, token=fb_oauth, n=25)

group$message
