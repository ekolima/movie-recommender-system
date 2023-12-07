import pandas as pd
import numpy as np
import time
import string
import os


def get_ratings(dataset='u.data', directory='ml-100k'):
    df = pd.read_csv(os.path.join(directory, dataset), sep='\t', header=None, encoding='utf-8')
    df.columns = ['user id', 'item id', 'rating', 'timestamp']
    return df


def get_tags(dataset='tags.csv', directory='data'):
    df = pd.read_csv(os.path.join(directory, dataset), sep=',', encoding='utf-8')
    df.columns = ['user id', 'movie id', 'tag', 'timestamp']
    df['tag'] = df['tag'].astype('str')
    return df.drop('timestamp', axis=1)


def get_movies(dataset='u.item', directory='ml-100k'):
    df = pd.read_table(os.path.join(directory, dataset), sep='|', header=None, encoding='iso-8859-1')
    df.columns = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
                  'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                  'Western']
    df = df.loc[:, ['movie id', 'movie title']]
    return df


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




def extract_ratings(id, ratings, id_type='user id'):
    return ratings[ratings[id_type] == id]
