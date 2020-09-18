#NON_PERSONALISED_RECOMMENDATIONS

import pandas as pd
import numpy as np

movie_list=pd.read_csv(r'C:\Users\BINDU\Desktop\DataSet\ml-latest-small\movies.csv')

genres=movie_list['genres']
genre_list=""
for index,row in movie_list.iterrows():
        genre_list += row.genres + "|"
        
genre_list_split = genre_list.split('|')
new_list = list(set(genre_list_split))
new_list.remove('')

#set of genres will be like
#['Thriller','Adventure','(no genres listed)','Animation','Action','Romance','Comedy','Children','Musical','War','Mystery','Documentary','IMAX','Fantasy','Drama','Crime','Film-Noir','Horror','Western','Sci-Fi']

#adding the various genres columns to movies dataset.
movies_with_genres = movie_list.copy()

for genre in new_list :
    movies_with_genres[genre] = movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)
movies_with_genres.index.name=None    

# Finding the average rating for movie and the number of ratings for each movie
avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean','count']))
avg_movie_rating['movieId']= avg_movie_rating.index
#Get the average movie rating across all movies 
avg_rating_all=ratings['rating'].mean()

#set a minimum threshold for number of reviews that the movie has to have
min_reviews=30

movie_score = avg_movie_rating.loc[avg_movie_rating['count']>min_reviews]

#create a function for weighted rating score based off count of reviews
def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    return (v/(v+m) * R) + (m/(m+v) * C)


movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)
movie_score.index.name=None
#join movie details to movie ratings
movie_score = pd.merge(movie_score,movies_with_genres,on='movieId')


# Gives the best movies according to genre based on weighted score
def best_movies_by_genre(genre,top_n):
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])

#EXAMPLE checking for top 10 movies genre 'ACTION' 
best_movies_by_genre('Action',10)
x=best_movies_by_genre('Action',10).drop(['mean','weighted_score','count'],axis=1)
x

"""
top 10 movies for genre 'ACTION' 
        title
474	Fight Club (1999)
66	Star Wars: Episode IV - A New Hope (1977)
227	Star Wars: Episode V - The Empire Strikes Back...
429	Matrix, The (1999)
229	Raiders of the Lost Ark (Indiana Jones and the...
776	Dark Knight, The (2008)
228	Princess Bride, The (1987)
237	Apocalypse Now (1979)
374	Saving Private Ryan (1998)
238	Star Wars: Episode VI - Return of the Jedi (1983)

"""
