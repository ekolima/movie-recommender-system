# Movie Recommender System

## Implementation Details
This project is a recommender system implementing the following algorithms:
- User-user
- Item-item
- Tag based (tag frequency)
- Content based (TF-IDF vectorization)
- Hybrid

There are 4 similarity measures available to use:
- Jaccard
- Dice
- Cosine
- Pearson

All algorithms and metrics are not using existing implemetations (eg metrics in sklearn, numpy etc), but have been developed from scratch.

## Using the recommender
### Recommender in terminal
The recommender can be called through a terminal, as follows:
`python src/recommender.py -d path/to/data -a algorithm -s similarity -n N -i id -p True`

The arguments that can be used with the recommender are the following:
-d or --data_directory: Used to provide a directory from which the Movielens data will be loaded. It is required to be provided even in the cases where the recommender is called with the precalculated values. The provided directory is validated before loading the data, by checking if it exists.
-n or --number: The number of recommendations to be returned. It is required to be provided. The number is validated to be a non zero, positive number. In case that a float is provided instead of an integer, the recommender will raise an exception.
-s or --similarity_metric: The similarity metric to be used. The value provided should be one of the following: jaccard, dice, cosine, pearson. It is required to be provided. The validity of the provided similarity metric is checked.
-a or --algorithm: The algorithm to be used for the recommendations generation. The value provided should be one of the following: user, item, tag, title, hybrid. It is required to be provided. The validity of the provided algorithm is checked.
-i or --input: The input id to generate recommendations for. For user, item and hybrid algorithms a user id should be provided. For tag and title algorithms, a movie id should be provided. It is required to be provided. The recommender validates that the input exists in the data.
-p or --precalculated: This is an optional argument indicating whether the user wants to load all data and calculate the recommendation scores from scratch or the precalculated scores should be used. The default value is False, meaning that scores are calculated from scratch. The existence of the precalculated scores is validated.
In cases that any of the aforementioned validation tests fail, the recommender will halt its execution and raise an exception instead.

### Recommender App
In order to run the Flask app, in terminal the user should execute: `python flask_app/app.py`
The app can be accessed on http://127.0.0.1:5000/ using any browser.