import pandas as pd
import string
import os


def get_ratings(dataset='ratings.csv', directory='ml-latest-small'):
    df = pd.read_csv(os.path.join(directory, dataset), sep=',', header=0, encoding='utf-8')
    df.columns = ['user id', 'item id', 'rating', 'timestamp']
    print(f'Dataset: {dataset} | Rows: {df.shape[0]} | Columns: {df.shape[1]}')
    return df.drop('timestamp', axis=1)


def get_tags(dataset='tags.csv', directory='ml-latest-small'):
    df = pd.read_csv(os.path.join(directory, dataset), sep=',', header=0, encoding='utf-8')
    df.columns = ['user id', 'movie id', 'tag', 'timestamp']
    df['tag'] = df['tag'].astype('str')
    print(f'Dataset: {dataset} | Rows: {df.shape[0]} | Columns: {df.shape[1]}')
    return df.drop('timestamp', axis=1)


def get_links(dataset='links.csv', directory='ml-latest-small'):
    df = pd.read_table(os.path.join(directory, dataset), sep=',', header=0, encoding='utf-8')
    df.columns = ['movie id', 'imdb id', 'tmdb id']
    print(f'Dataset: {dataset} | Rows: {df.shape[0]} | Columns: {df.shape[1]}')
    return df


def get_movies(dataset='movies.csv', directory='ml-latest-small'):
    df = pd.read_table(os.path.join(directory, dataset), sep=',', header=0, encoding='utf-8')
    df.columns = ['movie id', 'movie title', 'genres']
    print(f'Dataset: {dataset} | Rows: {df.shape[0]} | Columns: {df.shape[1]}')
    return df.loc[:, ['movie id', 'movie title']]


def generate_keywords(df):
    # remove year in the end of movie titles
    df['movie title'] = df['movie title'].str.replace('\(\d{4}\)','').str.strip().str.lower()

    # punctuation removal
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'  # `|` is not present here
    transtab = str.maketrans(dict.fromkeys(punct, ''))
    df = df.assign(text=df['movie title'].str.translate(transtab))

    # split into list of words
    df['text'] = df['text'].str.split()

    # explore list of words to a long dataframe
    df = df[['movie id', 'text']].explode('text')

    # remove stopwords
    stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
                 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
    df = df.loc[~df['text'].isin(stopwords)]

    return df


def get_data(data_directory='ml-latest-small'):
    print('--------------------------------------------------------------------------------')
    print(f'Loading data from directory {data_directory}')

    try:
        ratings = get_ratings(directory=data_directory)
        tags = get_tags(directory=data_directory)
        movies = get_movies(directory=data_directory)
        keywords = generate_keywords(movies)
    except Exception as e:
        print(f'Error: {e}')
        return None

    print('--------------------------------------------------------------------------------')
    return {'ratings': ratings, 'tags': tags, 'movies': movies, 'keywords': keywords}
