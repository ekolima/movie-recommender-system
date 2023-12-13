import argparse
import os
from load_data import get_data
import metrics
import algorithms
import warnings
from validators import *
warnings.simplefilter(action='ignore', category=FutureWarning)


def recommend(similarity_metric, algorithm, input, precalculated, number=100, data_directory='../ml-latest-small', movie_names=False):
    # validate arguments
    # check if data directory exists
    validate_directory(data_directory)

    # check if similarity metric is valid
    validate_metrics(similarity_metric)

    # check if algorithm is valid
    validate_algorithms(algorithm)

    # check if number is a number
    validate_is_number(number, 'Number of recommendations')
    number = int(number)
    validate_positive_number(number, 'Number of recommendations')

    # check if input is a number
    validate_is_number(input, 'Input id')
    input = int(input)

    # load data
    data = get_data(data_directory)

    # check if input is in data
    validate_input(input, data, algorithm)

    # set up similarity metric and algorithm dictionaries
    similarity_dict = {'jaccard': metrics.jaccard, 'dice': metrics.dice,
                       'cosine': metrics.cosine, 'pearson': metrics.pearson}

    algorithm_dict = {'user': algorithms.user_user, 'item': algorithms.item_item,
                      'tag': algorithms.tag_based, 'title': algorithms.content_based,
                      'hybrid': algorithms.hybrid}

    data_dict = {'user': data['ratings'], 'item': data['ratings'], 'tag': data['tags'],
                 'title': data['keywords'], 'hybrid': data}

    # get recommendations
    if precalculated:
        all_results = pd.read_pickle('results/{}_{}_top_items.pkl'.format(algorithm, similarity_metric))
        result = all_results[input].head(number).index.tolist()
    else:
        result = algorithm_dict[algorithm](id=int(input),
                                           data=data_dict[algorithm],
                                           n=number,
                                           similarity=similarity_dict[similarity_metric])

    # if movie names are requested, replace movie ids with movie names
    if movie_names:
        result = [data['movies'].set_index('movie id').loc[m].values[0] for m in result]

    print('--------------------------------------------------------------------------------')
    print('Recommendations for input {} using {} algorithm with {} similarity metric:'.format(input, algorithm, similarity_metric))
    print('--------------------------------------------------------------------------------')
    print(result)
    print('--------------------------------------------------------------------------------')
    return result


if __name__ == '__main__':
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
    # precalculated similarities
    parser.add_argument('-p', '--precalculated', help='precalculated similarities', required=False, default=False)

    # parse arguments
    args = parser.parse_args()
    data_directory = args.data_directory
    number = args.number
    similarity_metric = args.similarity_metric
    algorithm = args.algorithm
    input = args.input
    precalculated = args.precalculated


    recommend(similarity_metric, algorithm, input, precalculated, number, data_directory, movie_names=True)