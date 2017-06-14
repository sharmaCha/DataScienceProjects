# Amazon Fine Foods Reviews
A series of NLP projects using the Amazon Fine Food Reviews dataset. This repository is a work in progress. I will add projects as inspiration comes. 

## Project One: Predicting Review Helpfulness Using Macro-Level Text Summary Statistics  

_Word Cloud of Helpful Reviews_  
![alt text](https://github.com/jcharit1/Amazon-Fine-Foods-Reviews/blob/master/plots/helpful_reviews_word_cloud.png "Word Cloud of Helpful Reviews")  

I marked a review as helpful if the number of times customers rated it as helpful divided by the total number of times they gave it any rating (positive or negative) was greater than or equal to 90%. Due to fairly dramatic changes in the data over time, I restricted the analysis to the most recent year of data, 2012. The data can be found here: [https://snap.stanford.edu/data/web-FineFoods.html](https://snap.stanford.edu/data/web-FineFoods.html). About 22% of the reviews were helpful. 

### Summary and Motivation  

The goal of this project is to determine if a review's helpfulness can be predicted without using word specific features (i.e. bag of words and its derivatives). Instead I used macro-level text summary statistics such as:  

1. Number of Sentences
2. Number of Words
3. Readability 
4. Sentiment Metrics

The motivations are multi-fold: I am curious about the predictive power of macro-level text summary statistics and using word-specific features can make training, storing, and deploying predictive models more computationally intensive. 

Stepping back, the whole exercise of predicting review helpfulness is worthwhile because it can improve customer satisfaction. A major annoyance from on-line shopping is the product will not be as satisfying as it is advertised. Many shoppers depend on helpful reviews to avoid buyer's remorse. Therefore placing helpful reviews front and center for as many products as possible can improve the shopping experience, customer retention, and eventually profits. 

Traditionally on-line businesses solve this problem by allowing customers to rate the product reviews of other customers, then rank the reviews by their ratings. This can be insufficient for very popular products where potentially highly helpful reviews are never rated by customers because they are drowned out by unhelpful or moderately helpful reviews. Also, for new or niche products, some potentially highly useful reviews can go unrated because customer volume to the product page is too low. Predictive models can step in to identify potentially highly helpful reviews and help the web design team ensure that customers can easily find them.  

Now given the potential business benefit of a review helpfulness model, why not spend as much resources as needed to train, store, and deploy one? Every resource spent on one project is a resource not spent on another project, therefore it is important to be aware of options for intelligently trading off the resources invested in a project with the terminal quality of that project.  

### Results 

The preliminary analysis shows that a reasonably predictive model can be built without word specific features:  

![alt text](https://github.com/jcharit1/Amazon-Fine-Foods-Reviews/blob/master/plots/ROC_Basic.png "AUC ROC on Test Data of Macro-Text Stats Models")  

The area under the curve of the ROC (AUC ROC) of the best models, k-nearest neighbors, bagged trees, and random forest, were 0.8, 0.8, and 0.81 respectively. While these models are not "highly predictive" (AUC ROC of 0.9+), this is a proof of concept.  

Using a word-specific features approach like bag of words only resulted in a moderate improvement in predictive performance:  


![alt text](https://github.com/jcharit1/Amazon-Fine-Foods-Reviews/blob/master/plots/ROC_Basic_BOW.png "AUC ROC on Test Data of BOW Models") 

The AUC ROC of best model for both approaches, the random forest, improved from 0.81 to 0.83. To be fair, since I was training on a personal computer, I was forced to make decisions that potentially significantly limited the performance of the bag of words approach, but kept the training time under 48 hours.  

For example, I limited the number of words that can be used as features to 200 (setting max_features of the TfidfVectorizer estimator to 200). It is possible that the words that are most predictive of helpfulness appear infrequently enough that they will drop out under the 200 word feature limit. The extent of this issue was partially attenuated by the text preprocessing I did when preparing the reviews for the model. I used the [lemma](https://en.wikipedia.org/wiki/Lemmatisation) of the words to collapse the many forms that they can take ([their inflected forms](https://en.wikipedia.org/wiki/Inflection)) into one base form ([their lexeme](https://en.wikipedia.org/wiki/Lexeme)). This essentially does a preliminary dimensionality reduction of the data, reducing the number of highly infrequent words and, as a result, the number of words that will be dropped out by the word feature limit.  

Also, I didn't apply any Latent Dirichlet Allocation (LDA) analysis to the word features (or rather I gave up as the training time prolonged beyond what was practical). LDA is a powerful technique for discovering topics included in documents. It is possible that LDA could have identified the key topics that make food reviews very helpful, like taste or appearance of the food for example, and therefore boosted the performance of the models.  

On the other hand, however, even with these steps to shorten the training time, the models built using macro-level text summary statistics were still trained 10-12x faster. At the very least, this shows that using macro-level text summary statistics as features can be a good option for quickly turning around a solid minimum viable product -- buying the data science team time to experiment with more time-consuming approaches. Moreover, while a data science team will have access to much more powerful computers, they may need to train over a much larger dataset to ensure that the model is reliable for a wide variety of product reviews. Therefore, even for highly resourced teams, options for intelligently trading off model quality and the time it takes to train and deploy them are still useful.

Finally, I tried combining both approaches. The plot below shows the bootstrapped AUC ROC of the random forest model for when the macro-text stats and/or bag of words features were used:  

![alt text](https://github.com/jcharit1/Amazon-Fine-Foods-Reviews/blob/master/plots/BoxPlot_ROC_MacText_BOX.png "Comparison of AUC ROC on Test Data of BOW + Macro-Text Stats Models") 

Interestingly, though not entirely surprising, combining both types of features improved the model across the board. Both the minimum, maximum, and mean AUC ROC improved.  

Another interesting observation is the models that performed the best where those that create highly non-linear decision boundaries: the tree-based models and the nearest neighbors model. This probably means that the true decision boundary is highly non-linear and the classes might be compact within the feature space. It is also possible that the logistic, QDA, and naive bayes models were compromised by uni- and multivariate outliers. The tree-based models and nearest neighbors models are also more robust to outliers. Correcting for them more aggressively would be a useful follow up.

### Strategy and Tools

For its combination of speed and parsimony, I used [spacy](https://spacy.io/) to process the text and count the number of words/sentences. The package [textstat](https://pypi.python.org/pypi/textstat/0.1.6) was used for the readability score (Automated Readability Index) and the [NLTK](http://www.nltk.org/install.html) package was used for sentiment analysis. All the machine learning was done with scikit-learn.

### Next Steps

Next I want to improve the model performance by experimenting with:

1. Different measures of text readability
2. Using sentence, as oppose to whole-review level, measures of sentiment
3. Use vector representations of words to capture semantic summaries of reviews
4. Formally addressing the moderate class imbalance problem using SMOTE, over/under sampling, and Tomek Link removal
5. Use topic modeling to identify topics that are highly predictive of review usefulness

