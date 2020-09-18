# Recommendation_Systems

#Dataset - https://grouplens.org/datasets/movielens/
#Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
#ml-latest-small.zip (size: 1 MB)

# Various approaches used-

# Non personalised Recommendations
#used to calculate the best movies according to various genres and they can be recommended to any new user.

# Finding similar movies
#Without taking content into account (Just based on ratings)
#Based on the ratings of the users for different movies, we use K nearest neighbours algorithm to find the movies which are similar.

#With taking Content into account
#with information of genres to predict the most similar movies.

# Collaborative Filtering for particular user
#with matrix factorization
