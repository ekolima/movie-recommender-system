import pandas as pd
import os


def validate_directory(directory):
    # check if data directory exists
    if not os.path.isdir(directory):
        raise ValueError('Data directory does not exist')


def validate_metrics(metric):
    # check if metric is valid
    if metric not in ['jaccard', 'dice', 'cosine', 'pearson']:
        raise ValueError('Similarity metric not valid. Valid similarity metrics are: jaccard, dice, cosine, pearson')


def validate_algorithms(algorithm):
    # check if algorithm is valid
    if algorithm not in ['user', 'item', 'tag', 'title', 'hybrid']:
        raise ValueError('Algorithm not valid. Valid algorithms are: user, item, tag, title, hybrid')


def validate_is_number(number, name):
    # check if number is a number
    try:
        old_number = number
        number = int(number)
        if number != float(old_number):
            raise ValueError(name + ' must be an integer')
        del old_number
    except ValueError:
        raise ValueError(name + ' must be an integer')


def validate_positive_number(number, name):
    # check if number is positive
    if number <= 0:
        raise ValueError(name + ' must be positive')


def validate_input(input, data, algorithm):
    # check if input id is in data
    # for user and item algorithms, input is user id
    if algorithm in ['user', 'item']:
        if not input in data['ratings']['user id']:
            raise ValueError('The provided user id is not in data')
    # for the tag based algorithm, input is movie id and data is tags
    elif algorithm == 'tag':
        if not input in data['tags']['movie id']:
            raise ValueError('Movie id provided is not in data')
    # for the content based algorithm, input is movie id and data is keywords
    elif algorithm == 'title':
        if not input in data['keywords']['movie id']:
            raise ValueError('Input id is not in data')
    # for the hybrid algorithm, input is user id and data is ratings and tags
    else:
        if not input in data['ratings']['user id']:
            raise ValueError('The provided user id is not in data')
        if not input in data['tags']['movie id']:
            raise ValueError('Movie id provided is not in data')


def validate_preprocessed(algorithm, metric):
    # check if preprocessed data file exists
    if not os.path.isfile(os.path.join('results', '{}_{}_top_items.pkl'.format(algorithm, metric))):
        raise ValueError('Preprocessed data does not exist. Please run preprocess.py first')
