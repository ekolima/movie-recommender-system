import pandas as pd

from metrics import *
from utils import extract_ratings


def n_most_similar(all_users, ratings, user_ratings, k=128):
    all_users = all_users.sort_values('similarity', ascending=False).head(k)
    eligible_ratings = ratings[ratings['user id'].isin(all_users['user id'])]
    eligible_ratings = eligible_ratings[~eligible_ratings['item id'].isin(user_ratings['item id'])]
    eligible_ratings = eligible_ratings.merge(all_users, on='user id')
    eligible_ratings['weighted_rating'] = eligible_ratings['rating'] * eligible_ratings['similarity']
    total_similarity = sum(all_users['similarity'])
    try:
        result = eligible_ratings.groupby('item id', as_index=False)['weighted_rating'].sum()
        result['weighted_rating'] = result['weighted_rating'] / total_similarity
        return result
    except ZeroDivisionError:
        return np.nan


# ratings
def user_user(id, data, n, similarity=jaccard, return_scores=False):
    # extract ratings for user
    user_ratings = extract_ratings(id, data)
    if user_ratings.empty:
        return None

    users = pd.DataFrame(data[['user id']].drop_duplicates())
    if similarity.__name__ in ['jaccard', 'dice']:
        users['similarity'] = users['user id'].apply(lambda x: similarity(user_ratings, extract_ratings(x, data),
                                                                          extract_column='item id'))
    elif similarity.__name__ in ['cosine', 'pearson']:
        users['similarity'] = users['user id'].apply(lambda x: similarity(user_ratings, extract_ratings(x, data),
                                                                          extract_column='rating', id_col='item id'))
    result = n_most_similar(users.loc[users['user id'] != id], data, user_ratings)

    if return_scores:
        return result.sort_values('weighted_rating', ascending=False).head(n).set_index('item id')['weighted_rating'].to_dict()
    else:
        return result.sort_values('weighted_rating', ascending=False).head(n)['item id'].tolist()


# ratings
def item_item(id, data, n, similarity=pearson_matrix, return_scores=False):
    user_ratings = extract_ratings(id, data)
    if user_ratings.empty:
        return None

    # todo: fix for all methodologies
    user_rated_items = user_ratings['item id'].tolist()
    user_not_rated_items = pd.DataFrame(data.loc[~data['item id'].isin(user_rated_items)]['item id'].drop_duplicates())
    if similarity.__name__ in ['jaccard', 'dice']:
        user_not_rated_items['similarity'] = user_not_rated_items['item id'].apply(lambda x: similarity(user_ratings, extract_ratings(x, data),
                                                                          extract_column='item id'))
    elif similarity.__name__ in ['cosine', 'pearson_matrix']:
        user_rated_items = user_ratings['item id'].tolist()
        user_not_rated_items = data.loc[~data['item id'].isin(user_rated_items)]['item id'].unique()
        unknown_ratings = []
        for item in user_not_rated_items:
            matrix = data[data['item id'].isin([item]+user_rated_items)]
            matrix = matrix.pivot(columns='user id', index='item id', values='rating').fillna(0)
            similarities = similarity(matrix, item)
            final = pd.DataFrame(similarities, columns=['similarity']).merge(user_ratings.loc[:,['item id', 'rating']], on='item id')
            unknown_ratings.append(sum(final['similarity'] * final['rating'])/sum(final['similarity']))
        final_ratings = pd.DataFrame({'item id': user_not_rated_items, 'weighted_rating': unknown_ratings})

    if return_scores:
        return final_ratings.sort_values('weighted_rating', ascending=False).head(n).set_index('item id')['weighted_rating'].to_dict()
    else:
        return final_ratings.sort_values('weighted_rating', ascending=False).head(n)['item id'].tolist()


# tags
def tag_based(id, data, n, similarity=jaccard, return_scores=False):
    # calculate tag frequencies per movie
    tags_frequency = data.groupby(['movie id', 'tag'], as_index=False).count()
    # extract tag vector for target movie
    tag_target = tags_frequency[tags_frequency['movie id'] == id].drop('movie id', axis=1)
    if tag_target.empty:
        return None

    # apply similarity per movie group
    if similarity.__name__ in ['jaccard', 'dice']:
        tag_similarities = tags_frequency.groupby('movie id', as_index=False). \
            apply(lambda x: similarity(tag_target, x, extract_column='tag'))
    else:
        tag_similarities = tags_frequency.groupby('movie id', as_index=False).\
            apply(lambda x: similarity(tag_target, x.drop('movie id', axis=1), extract_column='user id', id_col='tag'))

    tag_similarities.columns = ['movie id', 'similarity']

    # TODO: expection handling

    # find n most similar movies
    if return_scores:
        return tag_similarities.loc[tag_similarities['movie id'] != id].sort_values('similarity', ascending=False).head(n).set_index('movie id')['similarity'].to_dict()
    else:
        return tag_similarities.loc[tag_similarities['movie id'] != id].sort_values('similarity', ascending=False).head(n)['movie id'].to_list()


def calculate_tf_idf(data):
    # calculate tf for each movie/keyword
    keywords_frequency_per_movie = data.groupby(['movie id', 'text'], as_index=False).agg(count=('text', 'size'))
    keywords_frequency_per_movie = keywords_frequency_per_movie.merge(
        data.groupby(['movie id'], as_index=False).agg(count2=('text', 'size')), on='movie id')
    keywords_frequency_per_movie['tf'] = keywords_frequency_per_movie['count'] / keywords_frequency_per_movie['count2']
    keywords_frequency_per_movie = keywords_frequency_per_movie.drop(['count', 'count2'], axis=1)

    # calculate idf for each keyword
    all_movies = data['movie id'].nunique()
    keywords_frequency = data.groupby(['text'], as_index=False).agg(count=('movie id', 'nunique'))
    keywords_frequency['idf'] = np.log(all_movies / keywords_frequency['count'])
    keywords_frequency = keywords_frequency.drop(['count'], axis=1)

    # calculate tf-idf for each movie/keyword
    keywords_frequency_per_movie = keywords_frequency_per_movie.merge(keywords_frequency, on='text')
    keywords_frequency_per_movie['tf-idf'] = keywords_frequency_per_movie['tf'] * keywords_frequency_per_movie['idf']
    keywords_frequency_per_movie = keywords_frequency_per_movie.drop(['tf', 'idf'], axis=1)

    return keywords_frequency_per_movie


# keywords
def content_based(id, data, n, similarity=jaccard, return_scores=False):

    # get tf-idf for each movie/keyword
    keywords_frequency_per_movie = calculate_tf_idf(data)

    # extract tf-idf vector for target movie
    keyword_target = keywords_frequency_per_movie.loc[keywords_frequency_per_movie['movie id'] == id, ['tf-idf', 'text']]
    if keyword_target.empty:
        return None

    # apply similarity per movie group
    if similarity.__name__ in ['jaccard', 'dice']:
        keyword_similarities = keywords_frequency_per_movie.groupby('movie id', as_index=False). \
            apply(lambda x: similarity(keyword_target, x, extract_column='text'))
    else:
        keyword_similarities = keywords_frequency_per_movie.groupby('movie id', as_index=False). \
            apply(lambda x: similarity(keyword_target, x.loc[:, ['tf-idf', 'text']], extract_column='tf-idf', id_col='text'))

    keyword_similarities.columns = ['movie id', 'similarity']

    # find n most similar movies
    if return_scores:
        return keyword_similarities.loc[keyword_similarities['movie id'] != id].sort_values('similarity', ascending=False).head(n).set_index('movie id')['similarity'].to_dict()
    else:
        return keyword_similarities.loc[keyword_similarities['movie id'] != id].sort_values('similarity', ascending=False).head(n)['movie id'].to_list()


# combine all methods
def hybrid(id, data, n, similarity=jaccard, return_scores=False):
    # data
    ratings, tags = data['ratings'], data['tags']

    # get top recommendation score using user-user and item-item
    user_user_recommendations = user_user(id, ratings, None, similarity, return_scores=True)
    item_item_recommendations = item_item(id, ratings, None, pearson_matrix, return_scores=True)

    # merge user user and item item scores
    user_user_recommendations = pd.DataFrame({'movie id': list(user_user_recommendations.keys()),
                                              'score': list(user_user_recommendations.values())})
    item_item_recommendations = pd.DataFrame({'movie id': list(item_item_recommendations.keys()),
                                              'score': list(item_item_recommendations.values())})
    all_recommendations = user_user_recommendations.merge(item_item_recommendations, on='movie id', how='outer', suffixes=['_user', '_item']).fillna(0)

    # combine recommendations
    all_recommendations['score'] = all_recommendations['score_user'] + all_recommendations['score_item']
    all_recommendations = all_recommendations.drop(['score_user', 'score_item'], axis=1)

    # get top N*5 recommendations to apply tag-based and content-based algorithms
    all_recommendations = all_recommendations.sort_values('score', ascending=False).head(n*5)

    # get tag based recommendations for top N*5 recommendations
    tag_based_results = dict()
    for m in all_recommendations['movie id'].to_list():
        print(m)
        tag_based_results[m] = tag_based(m, tags, None, similarity, return_scores=True)

    tag_based_results_long = pd.DataFrame(tag_based_results).unstack().reset_index()
    tag_based_results_long.columns = ['movie id', 'movie suggested', 'score']
    tag_based_results_long = tag_based_results_long.loc[tag_based_results_long['movie id'] != tag_based_results_long['movie suggested']]
    tag_based_results_long = tag_based_results_long.loc[tag_based_results_long['score'] > 0]

    tags_score = tag_based_results_long.groupby(['movie suggested'], as_index=False).agg({'score': 'sum'})
    tags_score.columns = ['movie id', 'score']

    # merge tag based recommendations with user-user and item-item recommendations
    all_recommendations = all_recommendations.merge(tags_score, on='movie id', how='outer').fillna(0)
    all_recommendations['score'] = all_recommendations['score_x'] + all_recommendations['score_y']
    all_recommendations = all_recommendations.drop(['score_x', 'score_y'], axis=1)

    # get top n recommendations
    all_recommendations = all_recommendations.sort_values('score', ascending=False).head(n)
    if return_scores:
        return all_recommendations.set_index('movie id')['score'].to_dict()
    else:
        return all_recommendations['movie id'].to_list()
