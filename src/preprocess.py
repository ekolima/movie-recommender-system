import argparse
import os
from load_data import *
from metrics import *
from algorithms import *
from utils import *
import json
import time
import numpy as np


# set up argument parser
parser = argparse.ArgumentParser()

# data directory
parser.add_argument('-d', '--data_directory', help='directory of data')


# parse arguments
args = parser.parse_args()
data_directory = args.data_directory
# data_directory = 'ml-latest-small'

# validate arguments
# check if data directory exists
if not os.path.isdir(data_directory):
    raise ValueError('Data directory does not exist')


# load data
ratings = get_ratings(directory=data_directory)
tags = get_tags(directory=data_directory)
movies = get_movies(directory=data_directory)
keywords = generate_keywords(movies)


# set up similarity metric and algorithm dictionaries
similarity_dict = {'jaccard': jaccard, 'dice': dice,
                   'cosine': cosine, 'pearson': pearson}

algorithm_dict = {'user': user_user, 'item': item_item,
                  'tag': tag_based, 'title': content_based,
                  'hybrid': hybrid}

# create results directory if it doesn't exist
if not os.path.isdir('results'):
    os.mkdir('results')


def n_most_similar_items(user_similarity_matrix, user_item_matrix, k=128):
    # Get the top k most similar users for each user along with the similarity rate
    top_similar_users = user_similarity_matrix.unstack()
    # drop rows where level 0 and level 1 index are the same
    top_similar_users = top_similar_users[top_similar_users.index.get_level_values(0) != top_similar_users.index.get_level_values(1)]
    # keep only the top k values
    top_similar_users = top_similar_users.groupby(level=0).apply(lambda x: x.nlargest(k))
    # drop level 1 in index
    top_similar_users.index = top_similar_users.index.droplevel(1)
    # name multi index
    top_similar_users.index.names = ['user id', 'similar user id']
    top_similar_users = top_similar_users.reset_index()
    # rename columns
    top_similar_users.columns = ['target user id', 'user id', 'similarity']

    # get ratings by merging user_item_matrix
    top_similar_users_ratings = pd.merge(top_similar_users, user_item_matrix, on=['user id'])
    # get weighted ratings (similarity * columns 1, ...n)
    top_similar_users_ratings.set_index(['target user id', 'user id'], inplace=True)
    # multiply similarity by ratings
    top_similar_users_ratings2 = top_similar_users_ratings.drop(['similarity'], axis=1).multiply(top_similar_users_ratings['similarity'], axis="index")

    # remove rows with same id in index
    top_similar_users_ratings2 = top_similar_users_ratings2[top_similar_users_ratings2.index.get_level_values(0) != top_similar_users_ratings2.index.get_level_values(1)]

    # get weighted average
    denominator = top_similar_users_ratings2.groupby(level=0).sum()
    similarities = top_similar_users_ratings.groupby(level=0)['similarity'].sum()
    weighted_avg = denominator.div(similarities, axis=0)

    # Set the ratings of items already rated by users to NaN
    weighted_avg[~np.isnan(user_item_matrix.values)] = np.nan
    return weighted_avg

#------------------
# user-user
# create a matrix with users as rows and items as columns
user_item_matrix = ratings.pivot(index='user id', columns='item id', values='rating')

# Get the top N most similar items for each user and metric
start = time.time()
jaccard_user_similarity_matrix = jaccard_similarity_matrix(user_item_matrix.fillna(0))
user_user_jaccard_top_items = n_most_similar_items(jaccard_user_similarity_matrix, user_item_matrix)
user_user_jaccard_top_items.to_pickle('../results/user_jaccard_top_items.pkl')
print('Jaccard done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
dice_user_similarity_matrix = dice_similarity_matrix(user_item_matrix.fillna(0))
user_user_dice_top_items = n_most_similar_items(dice_user_similarity_matrix, user_item_matrix)
user_user_dice_top_items.to_pickle('../results/user_dice_top_items.pkl')
print('Dice done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
cosine_user_similarity_matrix = cosine_similarity_matrix(user_item_matrix.fillna(0))
user_user_cosine_top_items = n_most_similar_items(cosine_user_similarity_matrix, user_item_matrix)
user_user_cosine_top_items.to_pickle('../results/user_cosine_top_items.pkl')
print('Cosine done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
pearson_user_similarity_matrix = pearson_similarity_matrix(user_item_matrix)
user_user_pearson_top_items = n_most_similar_items(pearson_user_similarity_matrix, user_item_matrix)
user_user_pearson_top_items.to_pickle('../results/user_pearson_top_items.pkl')
print('Pearson done | Time elapsed: {}'.format(time.time() - start))


#------------------
# item-item
def item_item_all(ratings):
    for similarity in [jaccard, dice, cosine, pearson]:
        results = dict()
        for i in ratings['user id'].unique():
            results[i] = item_item(i, ratings, None, similarity, True)
        # expand to columns
        results = pd.DataFrame.from_dict(results, orient='index')
        # to pickle
        results.to_pickle('../results/item_{}_top_items.pkl'.format(similarity.__name__))

# create a matrix with users as rows and items as columns
item_item_all(ratings)
#------------------
# tag-based
tags_frequency = tags.groupby(['movie id', 'tag'], as_index=False).agg(count=('tag', 'count'))
tags_frequency_matrix = tags_frequency.pivot(index='movie id', columns='tag', values='count')

start = time.time()
jaccard_tag_similarity_matrix = jaccard_similarity_matrix(tags_frequency_matrix.fillna(0))
jaccard_tag_similarity_matrix.to_pickle('../results/tag_jaccard_top_items.pkl')
print('Jaccard done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
dice_tag_similarity_matrix = dice_similarity_matrix(tags_frequency_matrix.fillna(0))
dice_tag_similarity_matrix.to_pickle('../results/tag_dice_top_items.pkl')
print('Dice done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
cosine_tag_similarity_matrix = cosine_similarity_matrix(tags_frequency_matrix.fillna(0))
cosine_tag_similarity_matrix.to_pickle('../results/tag_cosine_top_items.pkl')
print('Cosine done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
pearson_tag_similarity_matrix = pearson_similarity_matrix(tags_frequency_matrix)
pearson_tag_similarity_matrix.to_pickle('../results/tag_pearson_top_items.pkl')
print('Pearson done | Time elapsed: {}'.format(time.time() - start))

#------------------
# content-based
tf_idf = calculate_tf_idf(keywords)
tf_idf_matrix = tf_idf.pivot(index='movie id', columns='text', values='tf-idf')

start = time.time()
jaccard_content_similarity_matrix = jaccard_similarity_matrix(tf_idf_matrix.fillna(0))
jaccard_content_similarity_matrix.to_pickle('../results/title_jaccard_top_items.pkl')
print('Jaccard done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
dice_content_similarity_matrix = dice_similarity_matrix(tf_idf_matrix.fillna(0))
dice_content_similarity_matrix.to_pickle('../results/title_dice_top_items.pkl')
print('Dice done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
cosine_content_similarity_matrix = cosine_similarity_matrix(tf_idf_matrix.fillna(0))
cosine_content_similarity_matrix.to_pickle('../results/title_cosine_top_items.pkl')
print('Cosine done | Time elapsed: {}'.format(time.time() - start))

start = time.time()
pearson_content_similarity_matrix = pearson_similarity_matrix(tf_idf_matrix)
pearson_content_similarity_matrix.to_pickle('../results/title_pearson_top_items.pkl')
print('Pearson done | Time elapsed: {}'.format(time.time() - start))

#------------------
# hybrid
hybrid_user_similarity_matrix = jaccard_user_similarity_matrix + jaccard_user_similarity_matrix
hybrid_user_similarity_matrix.to_pickle('../results/hybrid_user_top_items.pkl')

hybrid_user_similarity_matrix = dice_user_similarity_matrix + dice_user_similarity_matrix
hybrid_user_similarity_matrix.to_pickle('../results/hybrid_user_top_items.pkl')

hybrid_user_similarity_matrix = cosine_user_similarity_matrix + cosine_user_similarity_matrix
hybrid_user_similarity_matrix.to_pickle('../results/hybrid_user_top_items.pkl')

hybrid_user_similarity_matrix = pearson_user_similarity_matrix + pearson_user_similarity_matrix
hybrid_user_similarity_matrix.to_pickle('../results/hybrid_user_top_items.pkl')
