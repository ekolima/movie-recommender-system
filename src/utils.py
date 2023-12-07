import numpy as np


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def extract_ratings(id, ratings, id_type='user id'):
    return ratings[ratings[id_type] == id]
