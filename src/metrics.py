# Description: This file contains the implementation of the metrics used for the recommender system.
import numpy as np
import pandas as pd


def jaccard(a, b, extract_column=None):

    try:
        if extract_column is not None:
            a = a.loc[:, extract_column]
            b = b.loc[:, extract_column]

        return a.isin(b).sum() / len(np.unique(np.append(a, b)))
    except KeyError:
        pass
    except ZeroDivisionError:
        pass


def dice(a, b, extract_column=None):
    # TODO: check implementation

    try:
        if extract_column is not None:
            a = a.loc[:, extract_column]
            b = b.loc[:, extract_column]

        return 2 * a.isin(b).sum() / (len(a) + len(b))
    except KeyError:
        pass
    except ZeroDivisionError:
        pass


def cosine(a, b, extract_column='rating', id_col='item id'):
    try:
        col_a, col_b = extract_column + '_a', extract_column + '_b'
        merged = a.merge(b, on=id_col, how='outer', suffixes=('_a', '_b')).fillna(0)

        return np.sum(merged[col_a] * merged[col_b]) / \
            (np.sqrt(np.sum(merged[col_a]**2)) * np.sqrt(np.sum(merged[col_b]**2)))
    except KeyError:
        pass
    except ZeroDivisionError:
        pass


def pearson(a, b, extract_column='user id', id_col='tag'):
    col_a, col_b = extract_column + '_a', extract_column + '_b'
    merged = a.merge(b, on=id_col, how='outer', suffixes=('_a', '_b'))
    merged[col_a] = merged[col_a] - np.nanmean(merged[col_a])
    merged[col_b] = merged[col_b] - np.nanmean(merged[col_b])

    return np.sum(np.nan_to_num(merged[col_a])*np.nan_to_num(merged[col_b]))/\
        (np.sqrt(np.sum(np.nan_to_num(merged[col_a])**2)) * np.sqrt(np.sum(np.nan_to_num(merged[col_b])**2)))


def pearson_matrix(matrix, x_target, k=128):
    target_row = matrix.loc[x_target]
    target_row = target_row - np.mean(target_row)
    target_row_sqrt = np.sqrt(sum(target_row ** 2))

    matrix = matrix.drop(x_target)
    matrix['mean'] = matrix.mean(axis=1)
    matrix = matrix.sub(matrix['mean'], axis=0)
    matrix = matrix.drop('mean', axis=1)

    # multiply each row in matrix times target_row
    matrix = matrix.apply(lambda row: sum(row * target_row)/(np.sqrt(sum(row ** 2)) * target_row_sqrt), axis=1)
    return matrix.sort_values(ascending=False).head(k)

