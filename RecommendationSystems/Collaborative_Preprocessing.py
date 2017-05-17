# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
import numpy as np
import pandas as pd
import simplejson as json
from datetime import datetime
from sklearn.cross_validation import train_test_split
from numpy.linalg import norm
from sklearn.pipeline import Pipeline, FeatureUnion
#import transformers
from datetime import datetime
from scipy.sparse import coo_matrix
import heapq

running_index = -1
utility = []

def submitUser(username):        
    fileheading = '.\Script Bundle/'
    
    cols = ('business_id', 'name')
    with open('.\Script Bundle/' + 'business.json') as f:
        df_business = pd.DataFrame(get_data(line, cols) for line in f)
    df_business = df_business.sort('business_id')
    df_business.index = range(len(df_business))

    # Load user data
    cols = ('user_id', 'name')
    with open(fileheading + 'user.json') as f:
        df_user = pd.DataFrame(get_data(line, cols) for line in f)

    df_user = df_user.sort('user_id')
    df_user.index = range(len(df_user))

    # Load review data
    cols = ('user_id', 'business_id', 'stars')
    with open(fileheading + 'review.json') as f:
        df_review = pd.DataFrame(get_data(line, cols) for line in f)
    
    data_load_time = datetime.now()
    print('Data was loaded at ' + data_load_time.time().isoformat())

    try:
        data_load_time = datetime.now()
    except NameError:
        execfile('src/load_data.py')
    else:
        print('Data was loaded at ' + data_load_time.time().isoformat())


    _name = username

    print('*** Using Collaborative Filtering for Recommendation ***')

    df_review['stars'] = df_review.groupby('business_id')['stars'].transform(lambda x : x - x.mean())
    df_review['user_idx'] = df_review.groupby('business_id')['user_id'].transform(get_idx)
    utility = df_review.groupby('business_id').apply(convert_to_sparse,df_user)
    
    fav_business = df_review.business_id[ df_review.stars[ df_review.user_id == _name ].argmax() ]

    rec = pd.DataFrame(get_recommended_businesses(2, fav_business,utility), columns=['similarity'])

    for business_id in rec.index:
        rec['name'] = df_business.name[ df_business.business_id == business_id ].values[0]
        
    #rec['name'] = [ df_business.name[ df_business.business_id == business_id ].values[1] for business_id in rec.index]

    print('Done')

    # Output recommendation
    print('Hi ' + df_user.name[df_user.user_id == _name].values[0] + '!\nCheck out these businesses!')
    _name1 = rec.name[0]
    print(_name1)
    #_name2 = rec.name[1]
     
    return _name1   
    #return render_template('submitUser.html',username=_name,name1=_name1,name2=_name2)

def get_data(line, cols):
    d = json.loads(line)
    return dict((key, d[key]) for key in cols)

def convert_to_sparse(group,df_user):
    #print(df_user)
    ratings = coo_matrix( (np.array(group['stars']), (np.array(group['user_idx']), np.zeros(len(group)))), shape = (len(df_user), 1) ).tocsc()
    return ratings / np.sqrt(float(ratings.T.dot(ratings).toarray()))

def cosine_similarity(v1, v2):
    return float(v1.T.dot(v2).toarray())

def get_recommended_businesses(n, business_id,utility):
    util_to_match = utility[utility.index == business_id]
    similarity = utility.apply(lambda x: cosine_similarity(util_to_match.values[0], x))
    similarity.sort(ascending=False)
    return similarity[1:(n+1)]

def get_idx(user_id):
    global running_index
    running_index = running_index + 1
    return pd.Series(np.zeros(len(user_id)) + running_index)

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    print('Input pandas.DataFrame #1:\r\n\r\n{0}'.format(dataframe1))
    userx = dataframe1
    #print(userx)
    print('username' + userx.userid[0])
    #print(type(userx.userid[0]))
    # If a zip file is connected to the third input port is connected,
    # it is unzipped under ".\Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    username = userx.userid[0]
    name = submitUser(username)
    result = pd.DataFrame({'Result': [name]})
    # Return value must be of a sequence of pandas.DataFrame
    return result,
