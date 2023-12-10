import argparse
import os
import load_data
import metrics
import algorithms
from utils import *
import json
import time




# set up argument parser
# parser = argparse.ArgumentParser()

# data directory
# parser.add_argument('-d', '--data_directory', help='directory of data', required=True)


# parse arguments
# args = parser.parse_args()
# data_directory = args.data_directory
data_directory = '../ml-latest-small'

# validate arguments
# check if data directory exists
if not os.path.isdir(data_directory):
    raise ValueError('Data directory does not exist')


# load data
ratings = load_data.get_ratings(directory=data_directory)
tags = load_data.get_tags(directory=data_directory)
movies = load_data.get_movies(directory=data_directory)
keywords = load_data.generate_keywords(movies)


# set up similarity metric and algorithm dictionaries
similarity_dict = {'jaccard': metrics.jaccard, 'dice': metrics.dice,
                   'cosine': metrics.cosine, 'pearson': metrics.pearson_matrix}

algorithm_dict = {'user': algorithms.user_user, 'item': algorithms.item_item,
                  'tag': algorithms.tag_based, 'title': algorithms.content_based,
                  'hybrid': algorithms.hybrid}

# create results directory if it doesn't exist
if not os.path.isdir('../results'):
    os.mkdir('../results')


def get_all_recommendations(algorithm_name, user_id, ratings, similarity_name):
    result = algorithm_dict[algorithm_name](user_id=int(user_id),
                                            data=ratings,
                                            n=None,
                                            similarity=similarity_dict[similarity_name],
                                            return_scores=True)

    # write to json
    final_result = {'algorithm': 'user',
                    'similarity_metric': similarity_name,
                    'user_id': user_id,
                    'recommendations': result}

    with open('../results/user/{}_{}.json'.format(similarity_name, user_id), 'w') as f:
        json.dump(final_result, f, default=np_encoder)


# get recommendations
all_results = []
for sim in similarity_dict.keys():
    print('Similarity metric: {}'.format(sim))
    time_start = time.time()
    for idx, user_id in enumerate(ratings.loc[:, 'user id'].unique()):
        get_all_recommendations(algorithm_name='user',
                                user_id=user_id,
                                ratings=ratings,
                                similarity_name=sim)

        if idx % 100 == 0:
            print('User {} done | Time elapsed: {}'.format(idx, time.time() - time_start))
