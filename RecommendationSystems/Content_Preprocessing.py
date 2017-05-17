# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import numpy as np
import pandas as pd
import simplejson as json
from datetime import datetime
from sklearn.cross_validation import train_test_split
from numpy.linalg import norm
from sklearn.pipeline import Pipeline, FeatureUnion
import transformers
from datetime import datetime
from scipy.sparse import coo_matrix
import heapq

running_index = -1
utility = []

def submitNewUser(username):
    print('**Loading data...')
    
    fileheading = '.\Script Bundle/'
    cols = ('business_id', 'name', 'categories', 'attributes', 'city', 'stars', 'full_address')
    with open(fileheading + 'business.json') as f:
        df_business = pd.DataFrame(get_data(line, cols) for line in f)
    df_business = df_business.sort('business_id')
    df_business.index = range(len(df_business))

    # Load user data
    cols = ('user_id', 'name', 'average_stars')
    with open(fileheading + 'user.json') as f:
        df_user = pd.DataFrame(get_data(line, cols) for line in f)
    df_user = df_user.sort('user_id')
    df_user.index = range(len(df_user))

    # Load review data
    cols = ('user_id', 'business_id', 'stars')
    with open(fileheading + 'review.json') as f:
        df_review = pd.DataFrame(get_data(line, cols) for line in f)
    #print(df_review)

    data_load_time = datetime.now()
    print('Data was loaded at ' + data_load_time.time().isoformat())

    # Load data
    try:
        data_load_time = datetime.now()
    except NameError:
        execfile('src/load_data.ipynb')
    else:
        print('Data was loaded at ' + data_load_time.time().isoformat())

    user = username
    print('*** Using Content-based Filtering for Recommendation ***')
    print('** Initializing feature extraction for user ' + user)

    categories = transformers.Column_Selector('categories')
    type(categories)

    #attributes = pd.DataFrame(list(df_business['attributes']))
    attributes = transformers.Column_Selector('attributes')
    type(attributes)

    #city = pd.DataFrame(list(df_business['city']))
    city = transformers.Column_Selector('city')
    type(city)

    OHE_cat = transformers.One_Hot_Encoder('categories', 'list', sparse=False)
    OHE_attr= transformers.One_Hot_Encoder('attributes', 'dict', sparse=False)
    OHE_city= transformers.One_Hot_Encoder('city', 'value', sparse=False)
    rating = transformers.Column_Selector('stars')
    OHE_union = FeatureUnion([ ('cat', OHE_cat), ('attr', OHE_attr), ('city', OHE_city), ('rating', rating) ])
    OHE_union.fit(df_business)
    print('Done')

    # Generate profile: weighted average of features for business she has reviewed
    print('**Getting businesses...')
    reviewed_businesses = df_review.ix[df_review.user_id == user]
    reviewed_businesses = reviewed_businesses[~ (reviewed_businesses.business_id.isin(['cM7eoSC_HhfS2D5VkFVY-Q']))]
    review = reviewed_businesses['stars'] - float(df_user.average_stars[df_user.user_id == user])
    #print(review)
    #idx_reviewed = [pd.Index(df_business.business_id).get_loc(b) for b in reviewed_businesses.business_id]
    #df_business.business_id.reset_index(drop=True)
    idx_reviewed = [pd.Index(df_business.business_id).get_loc(b) for b in reviewed_businesses.business_id if b not in 'cM7eoSC_HhfS2D5VkFVY-Q']


    print('**Creating profile...')
    features = OHE_union.transform(df_business.ix[idx_reviewed])
    profile = np.matrix(reviewed_businesses.stars) * features
    print('Done')

    # Given un-reviewed business, compute cosine similarity to user's profile
    print('**Computing similarity to all businesses...')
    idx_new = range(100) 
    #[pd.Index(df_business.business_id).get_loc(b) for b in df_business.business_id if b not in reviewed_businesses.business_id]
    features = OHE_union.transform(df_business.ix[idx_new])
    similarity = np.asarray(profile * features.T) * 1./(norm(profile) * norm(features, axis = 1))
    print('Done')

    # Output: recommend the most similar business
    idx_recommendation = similarity.argmax()
    print('\n**********')
    #print('Hi ' + df_user.name[df_user.user_id == user].iget_value(0) + '!')
    print('We recommend you to visit ' + df_business.name[idx_recommendation] + ' located at ')
    print(df_business.full_address[idx_recommendation])
    print('**********')

    _name = df_business.name[idx_recommendation]
    return _name
    #return render_template('submitUser.html', username=_name)



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
    print('username' + userx.userid[0])
    
    username = userx.userid[0]        
    name = submitNewUser(username)
    result = pd.DataFrame({'Result': [name]})
    # Return value must be of a sequence of pandas.DataFrame
    return result,
