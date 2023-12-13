# Description: This file contains the implementation of the metrics used for the recommender system.
import numpy as np
import pandas as pd


def jaccard(a, b, extract_column=None):
    try:
        # If extract_column is not None, extract the column from the dataframe
        if extract_column is not None:
            a = a[extract_column]
            b = b[extract_column]

        # Convert to sets
        set_a = set(a)
        set_b = set(b)

        # Calculate Jaccard similarity
        intersection_size = len(set_a.intersection(set_b))
        union_size = len(set_a.union(set_b))

        return intersection_size / union_size if union_size > 0 else 0

    except KeyError:
        pass
    except ZeroDivisionError:
        pass


def dice(a, b, extract_column=None):
    try:
        # If extract_column is not None, extract the column from the dataframe
        if extract_column is not None:
            a = a[extract_column]
            b = b[extract_column]

        # Convert to sets
        a_set = set(a)
        b_set = set(b)

        # Calculate Dice similarity
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

        # Convert Series to NumPy arrays
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


def jaccard_similarity_matrix(user_item_matrix):
    # Convert the user-item matrix to a binary matrix (1 if rated, 0 otherwise)
    zero_one_matrix = np.where(user_item_matrix > 0, 1, 0)
    binary_matrix = user_item_matrix > 0

    # Calculate the Jaccard similarity matrix
    numerator = zero_one_matrix @ zero_one_matrix.T
    denominator = np.clip(binary_matrix.sum(axis=1).values.reshape(-1, 1) + binary_matrix.sum(axis=1).values - numerator, 1, None)
    jaccard_matrix = numerator / denominator

    # Set the diagonal elements to 1.0
    jaccard_matrix = np.asmatrix(jaccard_matrix)
    np.fill_diagonal(jaccard_matrix, 1.0)

    # Convert the NumPy array to a DataFrame with IDs as index and columns
    user_similarity_matrix = pd.DataFrame(jaccard_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    return user_similarity_matrix


def dice_similarity_matrix(user_item_matrix):
    # Convert the user-item matrix to a binary matrix (1 if rated, 0 otherwise)
    zero_one_matrix = np.where(user_item_matrix > 0, 1, 0)
    binary_matrix = user_item_matrix > 0

    # Calculate the Dice similarity matrix
    numerator = zero_one_matrix @ zero_one_matrix.T
    denominator = np.clip(binary_matrix.sum(axis=1).values.reshape(-1, 1) + binary_matrix.sum(axis=1).values, 1, None)
    jaccard_matrix = 2 * numerator / denominator

    # Set the diagonal elements to 1.0
    jaccard_matrix = np.asmatrix(jaccard_matrix)
    np.fill_diagonal(jaccard_matrix, 1.0)

    # Convert the NumPy array to a DataFrame with IDs as index and columns
    user_similarity_matrix = pd.DataFrame(jaccard_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    return user_similarity_matrix


def cosine_similarity_matrix(user_item_matrix):
    # Calculate the cosine similarity matrix using NumPy's dot product
    cosine_matrix = user_item_matrix.dot(user_item_matrix.T) / (
        np.linalg.norm(user_item_matrix, axis=1).reshape(-1, 1) *
        np.linalg.norm(user_item_matrix, axis=1)
    )

    cosine_matrix = np.asmatrix(cosine_matrix)
    # Set the diagonal elements to 1.0
    np.fill_diagonal(cosine_matrix, 1.0)

    # Convert the NumPy array to a DataFrame with IDs as index and columns
    user_similarity_matrix = pd.DataFrame(cosine_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    return user_similarity_matrix


def pearson_similarity_matrix(user_item_matrix):
    # Convert the DataFrame to a NumPy matrix
    ratings_matrix = user_item_matrix.values

    # Calculate the mean of each user's ratings
    mean_ratings = np.nanmean(ratings_matrix, axis=1, keepdims=True)

    # Center the ratings matrix by subtracting the mean
    centered_ratings = ratings_matrix - mean_ratings

    # Calculate the square root of the sum of squared ratings
    norm_ratings = np.sqrt(np.nansum(centered_ratings**2, axis=1, keepdims=True))
    norm_ratings = np.where(norm_ratings == 0, np.nan, norm_ratings)

    # Calculate the normalized ratings matrix
    normalized_ratings = np.true_divide(centered_ratings, norm_ratings)
    normalized_ratings = np.nan_to_num(normalized_ratings)

    # Calculate the Pearson similarity matrix using matrix multiplication
    pearson_matrix = np.dot(normalized_ratings, normalized_ratings.T)

    # Set the diagonal elements to 1.0
    pearson_matrix = np.asmatrix(pearson_matrix)
    np.fill_diagonal(pearson_matrix, 1.0)

    # Convert the NumPy array to a DataFrame with IDs as index and columns
    user_similarity_matrix = pd.DataFrame(pearson_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    return user_similarity_matrix
