# Generating_Neighbourhood

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv(r'ml-latest-small\ratings.csv')
ratings = ratings[['userId', 'movieId','rating']]
movie_list=pd.read_csv(r'ml-latest-small\movies.csv')

# Creating a data frame that has user ratings accross all movies in form of matrix used in matrix factorisation
ratings_df = pd.pivot_table(ratings, index='userId', columns='movieId', aggfunc=np.max)
#merging ratings and movies dataframes
ratings_movies = pd.merge(ratings,movie_list, on = 'movieId')

#Gets the other top 10 movies which are watched by the people who saw this particular movie

def get_other_movies(movie_name):
    #get all users who watched a specific movie
    df_movie_users_series = ratings_movies.loc[ratings_movies['title']==movie_name]['userId']
    #convert to a data frame
    df_movie_users = pd.DataFrame(df_movie_users_series,columns=['userId'])
    #get a list of all other movies watched by these users
    other_movies = pd.merge(df_movie_users,ratings_movies,on='userId')
    #get a list of the most commonly watched movies by these other user
    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',ascending=False)
    other_users_watched['perc_who_watched'] = round(other_users_watched['userId']*100/other_users_watched['userId'][0],1)
    return other_users_watched[:10]

# Getting other top 10 movies which are watched by the people who saw 'Inception'
get_other_movies('Inception (2010)').drop(['userId','perc_who_watched'],axis=1)

#output
"""
title
Inception (2010)
Matrix, The (1999)
Fight Club (1999)
Dark Knight, The (2008)
Forrest Gump (1994)
Shawshank Redemption, The (1994)
Lord of the Rings: The Return of the King, The (2003)
Lord of the Rings: The Fellowship of the Ring, The (2001)
Lord of the Rings: The Two Towers, The (2002)
Pulp Fiction (1994)

"""
​
# Finding the average rating for movie and the number of ratings for each movie
avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean','count']))
avg_movie_rating['movieId']= avg_movie_rating.index
#Get the average movie rating across all movies 
avg_rating_all=ratings['rating'].mean()

#only include movies with more than 10 ratings
movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count']>=10]
movie_plus_10_ratings.index.name=None

filtered_ratings = pd.merge(movie_plus_10_ratings, ratings, on="movieId")

#create a matrix table with movieIds on the rows and userIds in the columns.
#replace NAN values with 0
movie_wide = filtered_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)

#specify model parameters and fit model to the data set
model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(movie_wide)

#Gets the top 10 nearest neighbours got the movie
def print_similar_movies(query_index) :
    #get the list of user ratings for a specific userId
    query_index_movie_ratings = movie_wide.loc[query_index,:].values.reshape(1,-1)
    #get the closest 10 movies and their distances from the movie specified
    distances,indices = model_knn.kneighbors(query_index_movie_ratings,n_neighbors = 11) 
    #write a lopp that prints the similar movies for a specified movie.
    for i in range(0,len(distances.flatten())):
        #get the title of the random movie that was chosen
        get_movie = movie_list.loc[movie_list['movieId']==query_index]['title']
        #for the first movie in the list i.e closest print the title
        if i==0:
            print('Recommendations for {0}:\n'.format(get_movie))
        else :
            #get the indiciees for the closest movies
            indices_flat = indices.flatten()[i]
            #get the title of the movie
            get_movie = movie_list.loc[movie_list['movieId']==movie_wide.iloc[indices_flat,:].name]['title']
            #print the movie
            print('{0}: {1}, with distance of {2}:'.format(i,get_movie,distances.flatten()[i]))

print_similar_movies(1)            

#output
"""
Recommendations for 0    Toy Story (1995)
Name: title, dtype: object:

1: 2355    Toy Story 2 (1999)
Name: title, dtype: object, with distance of 0.4273987396802845:
2: 418    Jurassic Park (1993)
Name: title, dtype: object, with distance of 0.4343631959138433:
3: 615    Independence Day (a.k.a. ID4) (1996)
Name: title, dtype: object, with distance of 0.43573830647233425:
4: 224    Star Wars: Episode IV - A New Hope (1977)
Name: title, dtype: object, with distance of 0.4426118294200633:
5: 314    Forrest Gump (1994)
Name: title, dtype: object, with distance of 0.4529040920598262:
6: 322    Lion King, The (1994)
Name: title, dtype: object, with distance of 0.4588546505397667:
7: 911    Star Wars: Episode VI - Return of the Jedi (1983)
Name: title, dtype: object, with distance of 0.4589106952274158:
8: 546    Mission: Impossible (1996)
Name: title, dtype: object, with distance of 0.46108722944164227:
9: 964    Groundhog Day (1993)
Name: title, dtype: object, with distance of 0.465831237415656:
10: 969    Back to the Future (1985)
Name: title, dtype: object, with distance of 0.46961865347827914:

"""



