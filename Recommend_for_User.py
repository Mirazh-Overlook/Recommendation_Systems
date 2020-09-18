# RECOMMENDATIONS FOR USERS BASED ON THEIR RATINGS 

import pandas as pd

ratings = pd.read_csv(r'C:\Users\BINDU\Desktop\DataSet\ml-latest-small\ratings.csv')
ratings = ratings[['userId', 'movieId','rating']]
movie_list=pd.read_csv(r'C:\Users\BINDU\Desktop\DataSet\ml-latest-small\movies.csv')

#get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
item_indices['movie_index']=item_indices.index

#get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
user_indices['user_index']=user_indices.index

#join the movie indices
df_with_index = pd.merge(ratings,item_indices,on='movieId')
df_with_index=pd.merge(df_with_index,user_indices,on='userId')


#import train_test_split module
from sklearn.model_selection import train_test_split

#take 80% as the training set and 20% as the test set
df_train, df_test= train_test_split(df_with_index,test_size=0.2)

n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
test_data_matrix = np.zeros((n_users, n_items))

#for every line in the data, set the value in the column and row to 
#line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index

for line in df_train.itertuples():
    train_data_matrix[line[5], line[4]] = line[3]

for line in df_test[:1].itertuples():
    test_data_matrix[line[5], line[4]] = line[3]
#train_data_matrix[line['movieId'], line['userId']] = line['rating'


from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    #select prediction values that are non-zero and flatten into 1 array
    prediction = prediction[ground_truth.nonzero()].flatten() 
    #select test values that are non-zero and flatten into 1 array
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    #return RMSE between values
    return sqrt(mean_squared_error(prediction, ground_truth))

#Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
for i in [1,2,5,20,40,60,100,200]:
    #apply svd to the test data
    u,s,vt = svds(train_data_matrix,k=i)
    #get diagonal matrix
    s_diag_matrix=np.diag(s)
    #predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
    #calculate rmse score of matrix factorisation predictions
    rmse_score = rmse(X_pred,test_data_matrix)
    rmse_list.append(rmse_score)



df_names = pd.merge(ratings,movie_list,on='movieId')
mf_pred = pd.DataFrame(X_pred)

#get movies rated by this user id

user_id = 1
users_movies = df_names.loc[df_names["userId"]==user_id]


user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
#get movie ratings predicted for this user and sort by highest rating prediction
sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
#rename the columns
sorted_user_predictions.columns=['ratings']
#save the index values as movie id
sorted_user_predictions['movieId']=sorted_user_predictions.index
print("Top 10 predictions for User " + str(user_id))
#display the top 10 predictions for this user
pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10]





