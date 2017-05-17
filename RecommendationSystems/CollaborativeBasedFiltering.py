#import os
#pypath = os.environ.get("PYTHONPATH", "/anaconda3/bin")
#os.environ["PYTHONPATH"] = pypath
#print(pypath)

import numpy
print(numpy)
import pandas as pd
import simplejson as json
from datetime import datetime
from sklearn.cross_validation import train_test_split

print('**Loading data...')

# LOAD DATA FOR TYPE = dataset_type
fileheading = '/Desktop/Recommendation/Collabrative_Dataset/'

def get_data(line, cols):
    d = json.loads(line)
    return dict((key, d[key]) for key in cols)

# Load business data
cols = ('business_id', 'name')
with open(fileheading + 'business.json') as f:
    df_business = pd.DataFrame(get_data(line, cols) for line in f)
df_business = df_business.sort('business_id')
df_business.index = range(len(df_business))
print(df_business.index)

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

#print(df_review)

import sys
#sys.path.append("/Python Workspace/")

import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.pipeline import Pipeline, FeatureUnion
#from src import transformers
from datetime import datetime
from scipy.sparse import coo_matrix
import heapq

# Load data
try:
    data_load_time = datetime.now()
except NameError:
    execfile('src/load_data.py')
else:
    print('Data was loaded at ' + data_load_time.time().isoformat())

# Personalized recommendation for a specific user
user = 'zzmRKNph-pBHDL2qwGv9Fw'
#user = 'NZrLmHRyiHmyT1JrfzkCOA'

## -------------------
## COLLABORATIVE FILTERING
## -------------------
print('*** Using Collaborative Filtering for Recommendation ***')

df_review['stars'] = df_review.groupby('business_id')['stars'].transform(lambda x : x - x.mean())

def get_idx(user_id): 
    global running_index
    running_index = running_index + 1
    return pd.Series(np.zeros(len(user_id)) + running_index) 
# For speed, get_idx assumes df_review and df_user contain the same users, and is fed in sorted order.
running_index = -1 
df_review['user_idx'] = df_review.groupby('business_id')['user_id'].transform(get_idx)
#df_review['user_idx'] = df_review.groupby('user_id')['user_id'].transform(get_idx)

# Work in terms of sparse matrix
print('** Processing utility matrix...')

def convert_to_sparse(group):
    ratings = coo_matrix( (np.array(group['stars']), (np.array(group['user_idx']), np.zeros(len(group)))), 
                          shape = (len(df_user), 1) ).tocsc()
    return ratings / np.sqrt(float(ratings.T.dot(ratings).toarray()))

utility = df_review.groupby('business_id').apply(convert_to_sparse) 
print('UTILITY',utility)

# Get top recommendatiokns
print('** Generating recommendations...')

def cosine_similarity(v1, v2):
    return float(v1.T.dot(v2).toarray())

def get_recommended_businesses(n, business_id):
    util_to_match = utility[utility.index == business_id]
    similarity = utility.apply(lambda x: cosine_similarity(util_to_match.values[0], x))
    similarity.sort(ascending=False)
    return similarity[1:(n+1)]

fav_business = df_review.business_id[ df_review.stars[ df_review.user_id == user ].argmax() ]

rec = pd.DataFrame(get_recommended_businesses(2, fav_business), columns=['similarity'])
#print(rec.index)

for business_id in rec.index:
    rec['name'] = df_business.name[ df_business.business_id == business_id ].values[0]
    #print(rec['name'])

#rec['name'] = [ df_business.name[ df_business.business_id == business_id ].values[1] for business_id in rec.index]
print('Done')

# Output recommendation
print('Hi ' + df_user.name[df_user.user_id == user].values[0] + '!\nCheck out these businesses!')
for name in rec.name:
    print(name)
