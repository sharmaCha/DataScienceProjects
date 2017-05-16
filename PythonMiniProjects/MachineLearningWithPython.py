import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn import svm

import seaborn as sbn
from datapad.plot import *
from pandas import Series, DataFrame
iris_bunch = load_iris()
iris = iris_bunch
print(iris_bunch['DESCR'])
iris
# ---
from datalore.meta_data import MetaData

_meta = MetaData()
_meta.set_dataframe('iris', 'iris_df', 'target')
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df
# ---
# ---
from datalore.plot import *

_meta.set_plot('scatter_plot', dataset_name='iris_df', x_data='sepal length (cm)', y_data='petal length (cm)', plot_type='scatter plot')
scatter_plot = ggplot(iris_df, aes(x='sepal length (cm)', y='petal length (cm)', color='target')) \
        + geom_point()
# ---
pd.scatter_matrix(iris_df);
# ---
sbn.pairplot(iris_df)
# ---
print(iris_df.shape)
# ---
print(iris_df.head(20))
# ---
print(iris_df.groupby('target').size())
# ---

# box and whisker plots
iris_df.boxplot(return_type='axes')

iris_df
# ---
_meta.set_plot('histogram', dataset_name='iris_df', x_data='sepal length (cm)', y_data='count', plot_type='histogram')
histogram = ggplot(iris_df, aes(x='sepal length (cm)', y='..count..')) \
        + geom_histogram()
# ---
iris_df
# ---
_meta.set_plot('histogram2', dataset_name='iris_df', x_data='sepal width (cm)', y_data='count', plot_type='histogram')
histogram2 = ggplot(iris_df, aes(x='sepal width (cm)', y='..count..')) \
        + geom_histogram()

iris_df
# ---
_meta.set_plot('histogram3', dataset_name='iris_df', x_data='petal length (cm)', y_data='count', plot_type='histogram')
histogram3 = ggplot(iris_df, aes(x='petal length (cm)', y='..count..')) \
        + geom_histogram()

iris_df
# ---
_meta.set_plot('histogram4', dataset_name='iris_df', x_data='petal width (cm)', y_data='count', plot_type='histogram')
histogram4 = ggplot(iris_df, aes(x='petal width (cm)', y='..count..')) \
        + geom_histogram()

#validation training
array = iris_df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#test harness
# ---
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# ---
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# ---
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# ---
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)
pred = logreg.predict(X_validation)
print("LOG REG Predictions")

print(accuracy_score(Y_validation, pred))
print(confusion_matrix(Y_validation, pred))
print(classification_report(Y_validation, pred))

# ---
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predlda = lda.predict(X_validation)
print("LDA Predictions")

print(accuracy_score(Y_validation, predlda))
print(confusion_matrix(Y_validation, predlda))
print(classification_report(Y_validation, predlda))
# ---
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("KNN Predictions")

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# ---
tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
predicttree = tree.predict(X_validation)
print("Decision Tree Predictions")

print(accuracy_score(Y_validation, predicttree))
print(confusion_matrix(Y_validation, predicttree))
print(classification_report(Y_validation, predicttree))

# ---
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictnb = nb.predict(X_validation)
print("Naive Bayes Predictions")

print(accuracy_score(Y_validation, predictnb))
print(confusion_matrix(Y_validation, predictnb))
print(classification_report(Y_validation, predictnb))
# ---
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, Y_train)
predictions1 = rf.predict(X_validation)
print("RF Predictions")
print(accuracy_score(Y_validation, predictions1))
print(confusion_matrix(Y_validation, predictions1))
print(classification_report(Y_validation, predictions1))
# ---
# Create SVM classification object 
#model_svm = svm.svc(kernel='linear', c=1, gamma=1) 
svm = SVC()
svm.fit(X_train, Y_train)
predictions2 = svm.predict(X_validation)
print("SVM Predictions")
print(accuracy_score(Y_validation, predictions2))
print(confusion_matrix(Y_validation, predictions2))
print(classification_report(Y_validation, predictions2))
# ---
h=.02
C = 1.0  # SVM regularization parameter
svc = SVC(kernel='linear', C=C).fit(X_train, Y_train)
rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, Y_train)
poly_svc = SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)
lin_svc = LinearSVC(C=C).fit(X_train, Y_train)

predictions3 = svc.predict(X_validation)
predictions4 = rbf_svc.predict(X_validation)
predictions5 = poly_svc.predict(X_validation)
predictions6 = lin_svc.predict(X_validation)

print("SVC Predictions")
print(accuracy_score(Y_validation, predictions3))
print(confusion_matrix(Y_validation, predictions3))
print(classification_report(Y_validation, predictions3))
print("RBF SVC Predictions")
print(accuracy_score(Y_validation, predictions4))
print(confusion_matrix(Y_validation, predictions4))
print(classification_report(Y_validation, predictions4))
print("POLY SVC Predictions")
print(accuracy_score(Y_validation, predictions5))
print(confusion_matrix(Y_validation, predictions5))
print(classification_report(Y_validation, predictions5))
print("LIN SVC Predictions")
print(accuracy_score(Y_validation, predictions6))
print(confusion_matrix(Y_validation, predictions6))
print(classification_report(Y_validation, predictions6))