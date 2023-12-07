import argparse
import os
from load_data import get_data
import metrics
import algorithms
import warnings
from utils import *
warnings.simplefilter(action='ignore', category=FutureWarning)


# set up argument parser
parser = argparse.ArgumentParser()

# data directory
parser.add_argument('-d', '--data_directory', help='directory of data', required=True)
# number of recommendations
parser.add_argument('-n', '--number', help='number of recommendations', required=True)
# similarity metric: jaccard, dice, cosine, pearson
parser.add_argument('-s', '--similarity_metric', help='similarity metric', required=True)
# algorithm: user, item, tag, title, hybrid
parser.add_argument('-a', '--algorithm', help='algorithm', required=True)
# input: user id, movie id etc.
parser.add_argument('-i', '--input', help='input', required=True)

# parse arguments
args = parser.parse_args()
data_directory = args.data_directory
number = args.number
similarity_metric = args.similarity_metric
algorithm = args.algorithm
input = args.input

# validate arguments
# check if data directory exists
if not os.path.isdir(data_directory):
    raise ValueError('Data directory does not exist')


# check if similarity metric is valid
if similarity_metric not in ['jaccard', 'dice', 'cosine', 'pearson']:
    raise ValueError('Similarity metric not valid. Valid similarity metrics are: jaccard, dice, cosine, pearson')

# check if algorithm is valid
if algorithm not in ['user', 'item', 'tag', 'title', 'hybrid']:
    raise ValueError('Algorithm not valid. Valid algorithms are: user, item, tag, title, hybrid')

# check if number is a number
try:
    old_number = number
    number = int(number)
    if number != float(old_number):
        raise ValueError('Number of recommendations must be an integer')
    del old_number
except ValueError:
    raise ValueError('Number of recommendations must be an integer')

# check if number is positive
if number <= 0:
    raise ValueError('Number of recommendations must be positive')

# check if input is number
try:
    old_input = input
    input = int(input)
    if input != float(old_input):
        raise ValueError('Input id must be an integer')
    del old_input
except ValueError:
    raise ValueError('Input id must be an integer')

# TODO: check if input is in data


# load data
data = get_data(data_directory)
# TODO: check if input is in data


# set up similarity metric and algorithm dictionaries
similarity_dict = {'jaccard': metrics.jaccard, 'dice': metrics.dice,
                   'cosine': metrics.cosine, 'pearson': metrics.pearson_matrix}

algorithm_dict = {'user': algorithms.user_user, 'item': algorithms.item_item,
                  'tag': algorithms.tag_based, 'title': algorithms.content_based,
                  'hybrid': algorithms.hybrid}

data_dict = {'user': data['ratings'], 'item': data['ratings'], 'tag': data['tags'],
             'title': data['keywords'], 'hybrid': data}


# get recommendations
result = algorithm_dict[algorithm](id=int(input),
                                   data=data_dict[algorithm],
                                   n=number,
                                   similarity=similarity_dict[similarity_metric])

print('--------------------------------------------------------------------------------')
print('Recommendations for input {} using {} algorithm with {} similarity metric:'.format(input, algorithm, similarity_metric))
print('--------------------------------------------------------------------------------')
print(result)
print('--------------------------------------------------------------------------------')
