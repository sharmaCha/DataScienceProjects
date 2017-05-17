import pandas as pd
import datetime
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.DataFrame()
train_cols = []
train_Y = pd.DataFrame()


def main():

    print "Preprocessing the data..."
    # Read data from CSV files
    df, test_len, train_Y, test_Y = readData()

    # Remove columns with maximum unique values
    df = removecols(df)

    # Convert Hour column in data to Day of week, Date and Hour of day
    df = changeHourCol(df)
    col_names_encode_list = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                             'device_id', 'device_model', 'device_ip']

    # Encodes non-float labels to new unique interger values
    df = labelEncode(col_names_encode_list, df)

    # Make traing and validation set of data
    train_data = df[1:(len(df) - test_len + 1)]
    test_data = df[(len(train_data)):]

    print "Length of train data: ", len(train_data)
    print "Length of validation data: ", len(test_data)

    # Scaling the values for the parameters
    sl = preprocessing.StandardScaler()
    sl.fit(train_data)
    train_data = sl.transform(train_data)
    sl.fit(test_data)
    test_data = sl.transform(test_data)
    print "Preprocessing done for the data."

    # Running different classifiers on data
    print(' ')
    naiveBayesModel(train_data, test_data, train_Y, test_Y)
    # print(' ')
    # KNNModel(train_data, test_data, train_Y, test_Y)
    print(' ')
    logisticModel(train_data, test_data, train_Y, test_Y)
    print(' ')
    logisticSGDModel(train_data, test_data, train_Y, test_Y)


def labelEncode(col_names, df):
    le = preprocessing.LabelEncoder()
    for col in col_names:
        le.fit(np.array(df[col]))
        df[col] = le.transform(np.array(df[col]))
    return df


def readData():
    # Read the file in panda dataframe
    # Column Names: 'id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id',
    # 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
    # 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    df = pd.read_csv('data/train_1million.csv')
    X, y = df.drop('click', 1), df['click']

    # Splitting data into training and validation set
    df, test_df, train_Y, test_Y = train_test_split(X, y, test_size=0.33, random_state=42)
    df = df.append(test_df)
    return df, len(test_df), train_Y, test_Y

# Calculate number of unique values in each columns and
# Dropping the columns with all or maximum unique values
def removecols(df):
    for col in list(df.columns.values):
        # print 'Number of unique values in', col, 'is :', len(df[col].unique())
        if (len(df) * 0.95) <= len(df[col].unique()):
            df = df.drop(col, 1)
    return df

# Change the hour column to Date, Day of Week and Day Time
def changeHourCol(df):
    df['date'] = df['hour'].apply(lambda x: x%10000/100)
    df['day_hour'] = df['hour'].apply(lambda x: x%100)
    df['dow'] = df['hour'].apply(lambda x: datetime.datetime.strptime(str(((x - x%100)/100) + 20000000), '%Y%m%d').strftime('%u'))

    df = df.drop('hour', 1)
    return df


def naiveBayesModel(train_data, test_data, train_Y, test_Y):

    # Build Naive Bayes Model
    model = GaussianNB()
    model.fit(train_data, train_Y)
    # print(model)

    # Make predictions
    predicted = model.predict_proba(test_data)
    # print predicted[0:,1]
    print "Naive Bayes :"
    print 'Log Loss :', metrics.log_loss(test_Y, predicted[0:,1])


def KNNModel(train_data, test_data, train_Y, test_Y):

    # Build Naive Bayes Model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_data, train_Y)
    # print(model)

    # Make predictions
    predicted = model.predict_proba(test_data)
    # print predicted[0:,1]
    print "KNN Model :"
    print 'Log Loss :', metrics.log_loss(test_Y, predicted[0:,1])


def logisticModel(train_data, test_data, train_Y, test_Y):

    # fit a logistic regression model to the data
    model = LogisticRegression();

    # Build LR Model with SGD and L2 Regularization
    # print(train_data)
    model.fit(train_data, train_Y)
    # print(model)

    # Make predictions
    predicted = model.predict_proba(test_data)
    # print predicted[0:,1]
    print "Logistic Regression (Vanilla): "
    print 'Log Loss :', metrics.log_loss(test_Y, predicted[0:,1])


def logisticSGDModel(train_data, test_data, train_Y, test_Y):

    # fit a logistic regression model to the data
    model = linear_model.SGDClassifier(alpha=0.00025, loss="log", penalty="l2")

    # Build LR Model with SGD and L2 Regularization
    # print(train_data)
    model.fit(train_data, train_Y)
    # print(model)

    # Make predictions
    predicted = model._predict_proba(test_data)
    # print predicted[0:,1]
    print "Logistic Model with SGD and L2 regularization :"
    print 'Log Loss :', metrics.log_loss(test_Y, predicted[0:,1])


if __name__=="__main__":
    main()