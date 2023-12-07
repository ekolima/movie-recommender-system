import pandas as pd
import os


def validate_directory(directory):
    if not os.path.isdir(directory):
        raise ValueError('Data directory does not exist')


def validate_metrics(metric):
    if metric not in ['jaccard', 'dice', 'cosine', 'pearson']:
        raise ValueError('Similarity metric not valid. Valid similarity metrics are: jaccard, dice, cosine, pearson')


def validate_algorithms(algorithm):
    if algorithm not in ['user', 'item', 'tag', 'title', 'hybrid']:
        raise ValueError('Algorithm not valid. Valid algorithms are: user, item, tag, title, hybrid')


def validate_is_number(number, name):
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
    if algorithm in ['user', 'item']:
        if not input in data['ratings']['user id']:
            raise ValueError('The provided user id is not in data')
    elif algorithm == 'tag':
        if not input in data['tags']['movie id']:
            raise ValueError('Movie id provided is not in data')
    elif algorithm == 'title':
        if not input in data['keywords']['movie id']:
            raise ValueError('Input id is not in data')
    else:
        if not input in data['ratings']['user id']:
            raise ValueError('The provided user id is not in data')
        if not input in data['tags']['movie id']:
            raise ValueError('Movie id provided is not in data')
