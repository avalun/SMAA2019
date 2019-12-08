import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)
dataset.head()

keywords = pd.read_csv("the-movies-dataset/keywords.csv", low_memory=False)
keywords.head()

ratings = pd.read_csv("the-movies-dataset/ratings.csv", low_memory=False)
ratings.head()

dataset['budget'] = dataset['budget'].replace(0, dataset['budget'].mean())

X = dataset.iloc[:, :].values
y_revenue = dataset.iloc[:, 12].values
y_rating = dataset.iloc[:, 18].values

# picking independent variables
X = X[:, [0, 1, 4, 9, 11, 13, 14, 15, 22, 23]]

# Removing zero REVENUES from the data
y_revenue_removed = []
y_rating_removed = []
X_removed = []
for l in range(0, len(y_revenue)):
    if y_revenue[l] != 0:
        y_revenue_removed.append(y_revenue[l])
        y_rating_removed.append(y_rating[l])
        X_removed.append(X[l])
y_revenue = np.array(y_revenue_removed)
y_rating = np.array(y_rating_removed)
X = np.array(X_removed)

# Adjusting inflation to 2019 at average inflation - 3.22% do this only if using revenue (12 y index)
avg_inflation = 1.01322
year_now = 2019
for l in range(0, len(y_revenue)):
    try:
        film_year = int(X[l, 4][0:4])
        y_revenue[l] = y_revenue[l] * (avg_inflation ** (year_now - film_year))
        X[l, 7] = int(film_year)
    except:
        X[l, 4] = 0

dataset = pd.DataFrame(X)
