### install.packages("cognizer")
install.packages("devtools")
install.packages("curl")
#Install cognizer package from github in RStudio
devtools::install_github("ColumbusCollaboratory/cognizer")
#So we are done with the installation bit.

#Now, let us invoke the WATSON service from RStudio.
# Step 1: Install Data Packages
# 
# Install the following data packages which is used in the recommended tests by ColumbusCollaboratory:
install.packages(c("rmsfact", "testthat"))
#Create the Renvironment file
#Run the following commands to create the environment file:
r_env <- file.path(normalizePath("~"), ".Renviron")
if (!file.exists(r_env)) file.create(r_env)
library(cognizer)
#Visual Recognition
#Understand the contents of images. Create custom classifiers 
#to develop smart applications. Create custom collections to search for similar images.

# Intended Use
# Visual Recognition allows users to understand the 
# contents of an image or video frame, answering the question:
#   "What is in this image?" and "Are there similar images?"
# 
# This service accepts JPEG, or PNG photographic images as input.

# How it works
# 
# Example Usage
# You could use Visual Recognition to organize your large collection of 
# digital photographs into albums based on visual themes, such as cars, 
# sports, or photos of people. Similarly, you could organize groups of 
# photos of different people by building an app to associate them semantically. 
# For example, your app could associate person A and person X with their many 
# baseball and beach pictures, while associating persons J, K, and L with numerous
# indoor photos. You can create custom classifiers for your specific needs. For example, 
# if you attend many sporting events, your custom classifier could identify specific teams.
# You can input:
#   JPEG images
# 
# PNG images
# 
# Image or website URLs
# 
# Custom: Classifier name, JPEG images (positive examples for each class, and negative examples)
# 
# Custom: Collections - your own images that you want to search for similar images
# 
# And the service will output:
#   Class description
# 
# Class taxonomy
# 
# Face detection (Gender, age range, celebrity (very limited see docs)
#                 
#                 Similar images with confidence scores
#                 


image_face_path <- system.file("extdata/images_faces", package = "cognizer")
IMAGE_API_KEY <- Sys.getenv("IMAGE_API_KEY")

#Detect Faces in Image -top-

#The following Web Services use IBM Bluemix Watson Services IMAGE_API_KEY 
# specific to Image processsing.
images <- list.files(image_face_path, full.names = TRUE)
image_faces <- image_detectface(images, IMAGE_API_KEY)
#We run the str function on the output image_faces
str(image_faces)
# 
# List of 7
# $ :List of 2
# ..$ images          :'data.frame':	1 obs. of  2 variables:
#   .. ..$ faces:List of 1
# .. .. ..$ :'data.frame':	1 obs. of  3 variables:
#   .. .. .. ..$ age          :'data.frame':	1 obs. of  3 variables:
#   .. .. .. .. ..$ max  : int 64
# .. .. .. .. ..$ min  : int 55
# .. .. .. .. ..$ score: num 0.504
# .. .. .. ..$ face_location:'data.frame':	1 obs. of  4 variables:
#   .. .. .. .. ..$ height: int 241
# .. .. .. .. ..$ left  : int 124
# .. .. .. .. ..$ top   : int 103
# .. .. .. .. ..$ width : int 196
# .. .. .. ..$ gender       :'data.frame':	1 obs. of  2 variables:
#   .. .. .. .. ..$ gender: chr "MALE"
# .. .. .. .. ..$ score : num 0.998
# .. ..$ image: chr "bill.jpg"
# ..$ images_processed: int 1

#Classification of Image
image_classes <- image_classify(images, IMAGE_API_KEY)
str(image_classes)
# 
# List of 7
# $ :List of 3
# ..$ custom_classes  : int 0
# ..$ images          :'data.frame':	1 obs. of  2 variables:
#   .. ..$ classifiers:List of 1
# .. .. ..$ :'data.frame':	1 obs. of  3 variables:
#   .. .. .. ..$ classes      :List of 1
# .. .. .. .. ..$ :'data.frame':	5 obs. of  3 variables:
#   .. .. .. .. .. ..$ class         : chr [1:5] "person" "professor" "sociologist" "adult person" ...
# .. .. .. .. .. ..$ score         : num [1:5] 0.869 0.55 0.531 0.53 0.87
# .. .. .. .. .. ..$ type_hierarchy: chr [1:5] NA "/person/professor" "/person/sociologist" NA ...
# .. .. .. ..$ classifier_id: chr "default"
# .. .. .. ..$ name         : chr "default"
# .. ..$ image      : chr "bill.jpg"
# ..$ images_processed: int 1

#Detect Text in Image
image_text<- image_detecttext(images, IMAGE_API_KEY)
str(image_text)

# List of 7
# $ :List of 2
# ..$ images          :'data.frame':	1 obs. of  3 variables:
#   .. ..$ image: chr "bill.jpg"
# .. ..$ text : chr ""
# .. ..$ words:List of 1
# .. .. ..$ : list()
# ..$ images_processed: int 1

#Language Translate 
# You can input:
#   Plain text in one of the supported input languages and domains.
# 
# And the service will output:
#   Plain text in the target language selected.

# Identify the language text is written in. 
# Translate text from one language to another for specific domains.
# 
# Intended Use
# The Watson Language Translator service provides domain-specific translation between languages.
# Currently, three domains are available that provide translation between a total of seven applications. 
# For best results, a domain that matches the content to be translated should be chosen.

# How it works
# 
# Example Usage
# Our intention is to provide domain-relevant translations. 
# Examples of where Watson Language Translator could be used include: 
#   An English-speaking help desk representative assists a Spanish-speaking 
# customer through a chat session that is translated in real-time. 
# A soccer forum administrator can translate user-generated content between 
# English and Portuguese to enable communication between users in the US, UK, and Brazil. 
# A West African news website can curate English news from across the globe 
# and present it in French to its constituents. A patent attorney in the US can 
# effectively discover prior art (to invalidate a patent claims litigation from a competitor)
# based on invention disclosures made in Italian with the Italian Patent Office. 
# All the above examples can benefit from the ability to adapt the base 
# Language Translator models to specific use cases. The service includes 
# a new Glossary-Based Customization function (currently available at no additional charge).

LANGUAGE_USERNAME_PASSWORD = "39409e98-2b55-49bb-990c-ffeaca9a0009:KbNsDllXlHD0"
text <- c("Mirando hacia adelante a UseR2017 en Bruselas!")
result <- text_translate(text, LANGUAGE_USERNAME_PASSWORD)
str(result)

#####################################################
#Personality Insights:
# Personality Insights
# Enables deeper understanding of people's 
# personality characteristics, needs, and values to help engage users on their own terms

# Intended Use
# Certain Services in Watson Developer Cloud are contextually specific and knowledgeable depending on the domain model and content set they are connected to.
# 
# This service uses any content.

# How it works
# 
# Example Usage
# The service can analyze text based on a customer's twitter stream to help a 
# travel agency decide between leading with a budget or luxury trip offer.

# You can input:
#   JSON, or Text or HTML (such as social media, emails, blogs, or other communication) written by one individual
# 
# And the service will output:
#   A tree of cognitive and social characteristics in JSON or CSV format


PERSONALITY_USERNAME_PASSWORD = "bdd46d87-6b57-4fef-9cdb-b160c3034f2e:arHnNADcz5rC"
text <- paste(replicate(1000, rmsfact::rmsfact()), collapse = ' ') #Ten Richard Stallman Facts used for Personality Insights.
#text <- c("Chandni is working hard to do the 
          #best she can and she is very serious for her goals") 
text
result <- text_personality(text, PERSONALITY_USERNAME_PASSWORD)
str(result)

## List of 1
##  $ :List of 6
##   ..$ id            : chr "*UNKNOWN*"
##   ..$ source        : chr "*UNKNOWN*"
##   ..$ word_count    : int 13600
##   ..$ processed_lang: chr "en"
##   ..$ tree          :List of 3
##   .. ..$ id      : chr "r"
##   .. ..$ name    : chr "root"
##   .. ..$ children:'data.frame':  3 obs. of  3 variables:
##   .. .. ..$ id      : chr [1:3] "personality" "needs" "values"
##   .. .. ..$ name    : chr [1:3] "Big 5" "Needs" "Values"
##   .. .. ..$ children:List of 3
##   .. .. .. ..$ :'data.frame':    1 obs. of  5 variables:
##   .. .. .. .. ..$ id        : chr "Agreeableness_parent"
##   .. .. .. .. ..$ name      : chr "Agreeableness"
##   .. .. .. .. ..$ category  : chr "personality"
##   .. .. .. .. ..$ percentage: num 0.114
##   .. .. .. .. ..$ children  :List of 1
##   .. .. .. .. .. ..$ :'data.frame':  5 obs. of  6 variables:
##   .. .. .. .. .. .. ..$ id            : chr [1:5] "Openness" "Conscientiousness" "Extraversion" "Agreeableness" ...
##   .. .. .. .. .. .. ..$ name          : chr [1:5] "Openness" "Conscientiousness" "Extraversion" "Agreeableness" ...
##   .. .. .. .. .. .. ..$ category      : chr [1:5] "personality" "personality" "personality" "personality" ...
##   .. .. .. .. .. .. ..$ percentage    : num [1:5] 0.694 0.428 0.703 0.114 0.29
##   .. .. .. .. .. .. ..$ sampling_error: num [1:5] 0.0466 0.0594 0.0453 0.0812 0.0754
##   .. .. .. .. .. .. ..$ children      :List of 5
##   .. .. .. .. .. .. .. ..$ :'data.frame':    6 obs. of  5 variables:
##   .. .. .. .. .. .. .. .. ..$ id            : chr [1:6] "Adventurousness" "Artistic interests" "Emotionality" "Imagination" ...
##   .. .. .. .. .. .. .. .. ..$ name          : chr [1:6] "Adventurousness" "Artistic interests" "Emotionality" "Imagination" ...
##   .. .. .. .. .. .. .. .. ..$ category      : chr [1:6] "personality" "personality" "personality" "personality" ...
##   .. .. .. .. .. .. .. .. ..$ percentage    : num [1:6] 0.635 0.554 0.389 0.864 0.639 ...
##   .. .. .. .. .. .. .. .. ..$ sampling_error: num [1:6] 0.0417 0.0864 0.039 0.0535 0.0451 ...
##   .. .. .. .. .. .. .. ..$ :'data.frame':    6 obs. of  5 variables:
##   .. .. .. .. .. .. .. .. ..$ id            : chr [1:6] "Achievement striving" "Cautiousness" "Dutifulness" "Orderliness" ...
##   .. .. .. .. .. .. .. .. ..$ name          : chr [1:6] "Achievement striving" "Cautiousness" "Dutifulness" "Orderliness" ...

##############################################

#Tone Analyzer
# Helps users understand the tones that are present in text
# 
# Intended Use
# Wonder how your message might be perceived by the end user? 
# Perhaps a bit too angry in your emails? Tone Analyzer can help. 
# The IBM WatsonT Tone Analyzer Service uses linguistic analysis to detect 
# three types of tones from written text: emotion, social tendencies, and 
# language style. Emotions identified include things like anger, fear, joy, 
# sadness, and disgust. Identified social tendencies include things from the 
# Big Five personality traits used by some psychologists. These include openness, 
# conscientiousness, extroversion, agreeableness, and emotional range. Identified 
# writing styles include confident, analytical, and tentative.

# How it works
# 
# Example Usage
# Some common use cases for the Tone Analyzer service are to analyze email messages 
# before sending them, analyze the content of presentations and other customer communications
# before delivering them, examine how readers might perceive your blog posts, or build tone 
# detection into a digital virtual agent system to drive an improved customer experience.

# You can input:
#   Any Text
# 
# And the service will output:
#   JSON that provides a hierarchical representation of the analysis of the terms in the input message

TONE_USERNAME_PASSWORD="6d495037-b07e-4cb8-bc4b-ff9eb1d021de:gAM0BP1yGQsB"
text <- c("Boston, It is so cold and Awesome!")
result <- text_tone(text, TONE_USERNAME_PASSWORD)
str(result)

# 
# List of 1
# $ :List of 1
# ..$ document_tone:List of 1
# .. ..$ tone_categories:'data.frame':	3 obs. of  3 variables:
#   .. .. ..$ tones        :List of 3
# .. .. .. ..$ :'data.frame':	5 obs. of  3 variables:
#   .. .. .. .. ..$ score    : num [1:5] 0.03402 0.00658 0.03905 0.75744 0.05474
# .. .. .. .. ..$ tone_id  : chr [1:5] "anger" "disgust" "fear" "joy" ...
# .. .. .. .. ..$ tone_name: chr [1:5] "Anger" "Disgust" "Fear" "Joy" ...
# .. .. .. ..$ :'data.frame':	3 obs. of  3 variables:
#   .. .. .. .. ..$ score    : num [1:3] 0 0 0
# .. .. .. .. ..$ tone_id  : chr [1:3] "analytical" "confident" "tentative"
# .. .. .. .. ..$ tone_name: chr [1:3] "Analytical" "Confident" "Tentative"
# .. .. .. ..$ :'data.frame':	5 obs. of  3 variables:
#   .. .. .. .. ..$ score    : num [1:5] 0.805 0.441 0.686 0.729 0.393
# .. .. .. .. ..$ tone_id  : chr [1:5] "openness_big5" "conscientiousness_big5" "extraversion_big5" "agreeableness_big5" ...
# .. .. .. .. ..$ tone_name: chr [1:5] "Openness" "Conscientiousness" "Extraversion" "Agreeableness" ...
# .. .. ..$ category_id  : chr [1:3] "emotion_tone" "language_tone" "social_tone"
# .. .. ..$ category_name: chr [1:3] "Emotion Tone" "Language Tone" "Social Tone"


# Speech to Text
# The Speech to Text service converts the human voice into the written word. 
# It can be used anywhere there is a need to bridge the gap between the 
# spoken word and their written form, including voice control of embedded systems,
# transcription of meetings and conference calls, and dictation of email and notes. 
# This easy-to-use service uses machine intelligence to combine information about 
# grammar and language structure with knowledge of the composition of the 
# audio signal to generate an accurate transcription.
# The following languages and features are currently available:

# Keyword Spotting (BETA)
# Optional ability to search for one or more keywords in the audio stream. 
# The returned metadata includes the beginning time, end time and confidence score
# for each instance of the keyword found. Keyword Spotting is currently available 
# at no additional charge.
# 
# Metadata
# Receive a metadata object in the JSON response that includes confidence score 
# (per word), start/end time (per word), and alternate hypotheses / N-Best (per phrase).
# A new option for returning word alternatives per (sequential) 
# time intervals is now available.

# Intended Use
# Certain Services in Watson Developer Cloud are contextually specific 
# and knowledgeable depending on the domain model and content set they are connected to.
# 
# How it works
# 
# Example Usage
# A startup could create a mobile application that has voice-interactivity built in. 
# A financial institution may wish to transcribe their contact center calls for quality
# and training purposes. A corporation might make audio and video files in their 
# Enterprise Content Management system searchable based on the words spoken in those media files.
# 
# You can input:
#   Streamed audio with Intelligible Speech
# 
# Recorded audio with Intelligible Speech
# 
# And the service will output:
#   Text transcriptions of the audio with recognized words
SPEECH_TO_TEXT_USERNAME_PASSWORD="4b7dab15-ca73-43e7-80dc-5c72905ee135:QDGtQpX5epCi"
audio_path <- system.file("extdata/audio", package = "cognizer")
audios <- list.files(audio_path, full.names = TRUE)
audio_transcript <- audio_text(audios, SPEECH_TO_TEXT_USERNAME_PASSWORD)
str(audio_transcript)

## List of 1
##  $ :List of 2
##   ..$ results     :'data.frame': 1 obs. of  2 variables:
##   .. ..$ alternatives:List of 1
##   .. .. ..$ :'data.frame':   1 obs. of  2 variables:
##   .. .. .. ..$ confidence: num 0.954
##   .. .. .. ..$ transcript: chr "hello world "
##   .. ..$ final       : logi TRUE
##   ..$ result_index: int 0



#### CONVERSATION #############
# Add a natural language interface to your application to automate 
# interactions with your end users. Common applications include virtual 

###Language Detection



# agents and chat bots that can integrate and communicate on any channel or device.






######################################################

library(RCurl) # install.packages("RCurl") # if the package is not already installed
library(httr)
library(rjson)
#library(XML)

######### Housekeeping And Authentication

url_CONV="https://gateway.watsonplatform.net/conversation/api"
version="?version=2016-07-11"
username_CONV="0f74cc05-cd18-4bb3-8be4-b824a44ca4be" # check we got it from KEYs.R file in same directory - looks like this - "e63f524d-9999-9999-9999-e6b1d4e99b87"
password_CONV="X4vzahtoAXbz" # check we got it from KEYs.R file in same directory - looks like this - "ABCD0EggCXYZ"
workspace_CONV="e843d2e9-5523-45ea-84f6-97e41741b5d3" # check we got it from KEYs.R file - looks like this - "2ded4293-9999-9999-9999-4c8b1289be81" <- **** YOU NEED TO PULL THIS FROM BROWSER URL WHEN YOU TEST CONVO IN TOOLING - I THINK ONLY PLACE TO GET RIGHT NO

#getwd()


### CAREFUL - THIS CODE IS IN PROGRESS - JUST TO GET SYNTAX RIGHT - NO FUNCTIONS OR OPTIMIZATION YET


## LEVEL 0 - INITIATE - FIRST CONTACT - blank utterance works to start thigns off (But 'coffee' will also hit NLC and return confidence)
response <- POST(url=paste(url_CONV,"/workspaces/",workspace_CONV,"/message",version,sep=""),
                 authenticate(username_CONV,password_CONV),
                 add_headers("Content-Type"="application/json"),
                 body = '{ "input": { "text":""},
                 "system":{ "dialog_stack":["root"]},
                 "dialog_turn_counter":1,
                 "dialog_request_counter":1
                 }',
                 encode = "json"
)
response
response_text <- content(response, "text", encoding = "UTF-8")  # or encoding = "ISO-8859-1"
response_text


## RESPONSE [1] "{\"intents\":[],\"entities\":[],\"input\":{\"text\":\"\"},\"output\":{\"log_messages\":[],
# \"text\":[\"Welcome to the Tasty Helper!  \\nWhat are you looking for?\"],\"nodes_visited\":[\"node_6_1468893593485\"]},
# \"context\":{\"conversation_id\":\"1a598ffd-2b01-4330-923a-291a0cecc5f9\",
# \"system\":{\"dialog_stack\":[\"node_6_1468893593485\"],\"dialog_turn_counter\":1,\"dialog_request_counter\":1}}}"

## (!!) NOW WE NEED TO COPY/PASTE THE RETURNED NODE and ALSO THE CONVERSATION ID INTO SUBSEQUENT DIALOG - TO FOLLOW THREAD:

#### LEVEL 1 ENGAGE

response <- POST(url=paste(url_CONV,"/workspaces/",workspace_CONV,"/message",version,sep=""),
                 authenticate(username_CONV,password_CONV),
                 add_headers("Content-Type"="application/json"),
                 body = '{"input":{"text":"coffee"},
                 "context":{"conversation_id":"1a598ffd-2b01-4330-923a-291a0cecc5f9",
                 "system":{"dialog_stack":["node_6_1468893593485"],
                 "dialog_turn_counter":1,"dialog_request_counter":1}}}'
)
response
response_text <- content(response, "text", encoding = "UTF-8")  # or encoding = "ISO-8859-1"
response_text

# RESPONSE [1] "{\"intents\":[{\"intent\":\"coffee\",\"confidence\":1}],\"entities\":[],\"input\":{\"text\":\"coffee\"},
# \"output\":{\"log_messages\":[],\"text\":[\"Coffee makes me very productive (level 1)\"],
# \"nodes_visited\":[\"node_12_1468894103975\"]},\"context\":{\"conversation_id\":\"1a598ffd-2b01-4330-923a-291a0cecc5f9\",
# \"system\":{\"dialog_stack\":[\"node_12_1468894103975\"],\"dialog_turn_counter\":2,\"dialog_request_counter\":2}}}"


#### LEVEL 2 CONTINUE CONV - take note how we need to continue to propogate CONV_ID and ALSO UPDATE DIALOG_STACK NODE to new level (!)

response <- POST(url=paste(url_CONV,"/workspaces/",workspace_CONV,"/message",version,sep=""),
                 authenticate(username_CONV,password_CONV),
                 add_headers("Content-Type"="application/json"),
                 body = '{"input":{"text":"coffee"},
                 "context":{"conversation_id":"1a598ffd-2b01-4330-923a-291a0cecc5f9",
                 "system":{"dialog_stack":["node_12_1468894103975"],
                 "dialog_turn_counter":1,"dialog_request_counter":1}}}'
)
response
response_text <- content(response, "text", encoding = "UTF-8")  # or encoding = "ISO-8859-1"
response_text


# RESULTS _ [1] "{\"intents\":[{\"intent\":\"coffee\",\"confidence\":1}],\"entities\":[],\"input\":{\"text\":\"coffee\"},
#\"output\":{\"log_messages\":[],\"text\":[\"wow - two in a row - you must LOVE caffeine (level 2)\"],
#\"nodes_visited\":[\"node_6_1474949437344\"]},\"context\":{\"conversation_id\":\"1a598ffd-2b01-4330-923a-291a0cecc5f9\",
#\"system\":{\"dialog_stack\":[\"node_6_1474949437344\"],\"dialog_turn_counter\":2,\"dialog_request_counter\":2}}}"


##### LEVEL 3 DEEP INTO THE DIALOG TREE (and resets to beginning)

response <- POST(url=paste(url_CONV,"/workspaces/",workspace_CONV,"/message",version,sep=""),
                 authenticate(username_CONV,password_CONV),
                 add_headers("Content-Type"="application/json"),
                 body = '{"input":{"text":"coffee"},
                 "context":{"conversation_id":"d8c4f16c-1165-4486-9f71-085e0578a587",
                 "system":{"dialog_stack":["node_6_1474949437344"],
                 "dialog_turn_counter":1,"dialog_request_counter":1}}}'
)
response
response_text <- content(response, "text", encoding = "UTF-8")  # or encoding = "ISO-8859-1"
response_text