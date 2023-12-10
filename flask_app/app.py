from flask import Flask, render_template, request
import sys
import os

sys.path.append(os.path.abspath('../src'))
from recommender import recommend

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None

    if request.method == 'POST':
        # Get user input from the form
        user_id = int(request.form['user_id'])
        similarity_function = request.form['similarity_function']
        algorithm = request.form['algorithm']

        # Call your recommendation function with the user input and selected options
        recommendations = recommend(similarity_function, algorithm, user_id, movie_names=True)
        # recommendations = [{'Rank': idx + 1, 'Movie': f'Movie {movie}'} for idx, movie in enumerate(recommendations)]

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
