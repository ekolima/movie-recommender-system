# Description: This file contains the implementation of the metrics used for the recommender system.
import numpy as np
import pandas as pd


def jaccard(a, b, extract_column=None):
    try:
        if extract_column is not None:
            a = a[extract_column]
            b = b[extract_column]

        set_a = set(a)
        set_b = set(b)

        intersection_size = len(set_a.intersection(set_b))
        union_size = len(set_a.union(set_b))

        return intersection_size / union_size if union_size > 0 else 0

    except KeyError:
        pass
    except ZeroDivisionError:
        pass

def dice(a, b, extract_column=None):
    try:
        if extract_column is not None:
            a = a[extract_column]
            b = b[extract_column]

        a_set = set(a)
        b_set = set(b)

        return 2 * len(a_set.intersection(b_set)) / (len(a_set) + len(b_set))
    except KeyError:
        pass
    except ZeroDivisionError:
        pass


def cosine(a, b, extract_column='rating', id_col='item id'):
    try:
        # Create dictionaries to represent the ratings for each item in a and b
        ratings_a = dict(zip(a[id_col], a[extract_column]))
        ratings_b = dict(zip(b[id_col], b[extract_column]))

        # Find common items
        common_items = set(ratings_a.keys()) & set(ratings_b.keys())

        # Calculate the numerator (dot product of the two vectors)
        numerator = sum(ratings_a[item] * ratings_b[item] for item in common_items)

        # Calculate the denominator (magnitude of each vector)
        magnitude_a = np.sqrt(sum(ratings_a[item] ** 2 for item in ratings_a))
        magnitude_b = np.sqrt(sum(ratings_b[item] ** 2 for item in ratings_b))

        # Calculate cosine similarity
        similarity = numerator / (magnitude_a * magnitude_b) if magnitude_a * magnitude_b != 0 else 0

        return similarity if not np.isnan(similarity) else 0  # Handle NaN case

    except KeyError:
        pass
    except ZeroDivisionError:
        pass


def pearson(a, b, extract_column='rating', id_col='item id'):
    try:
        # Create Series to represent the ratings for each item in a and b
        ratings_a = pd.Series(a[extract_column].values, index=a[id_col])
        ratings_b = pd.Series(b[extract_column].values, index=b[id_col])

        # Find all items by taking the union of indices
        all_items = ratings_a.index.union(ratings_b.index)

        # Fill missing values with 0
        ratings_a = ratings_a.reindex(all_items)
        ratings_b = ratings_b.reindex(all_items)

        # Convert to NumPy arrays for more efficient calculations
        array_a = ratings_a.values
        array_b = ratings_b.values

        # Calculate the mean of each vector using NumPy
        mean_a = np.nanmean(array_a)
        mean_b = np.nanmean(array_b)

        # Calculate the numerator and denominators using vectorized operations
        numerator = np.nansum((array_a - mean_a) * (array_b - mean_b))
        denominator_a = np.sqrt(np.nansum((array_a - mean_a) ** 2))
        denominator_b = np.sqrt(np.nansum((array_b - mean_b) ** 2))

        # Calculate Pearson similarity
        similarity = numerator / (denominator_a * denominator_b) if denominator_a * denominator_b != 0 else 0

        return similarity if not np.isnan(similarity) else 0  # Handle NaN case

    except KeyError:
        pass
    except ZeroDivisionError:
        pass


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

