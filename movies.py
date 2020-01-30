import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def prepare_dataset():
    dataset = pd.read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)
    # print(dataset.head())
    # print(dataset.info())
    # print(dataset.describe())

    ratings = pd.read_csv("the-movies-dataset/ratings.csv", low_memory=False)
    # print(ratings.head())
    # print(ratings.info())
    plt.figure(figsize=(18, 8))
    plt.barplot(x='original_title', y='vote_average', data=dataset.head(10))
    plt.show()

    # Drop extraneous columns
    col = ['adult', 'belongs_to_collection', 'homepage', 'original_language', 'overview', 'popularity', 'poster_path',
           'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'video', 'production_countries', 'vote_count',
           'revenue', 'budget', 'id', 'imdb_id']
    dataset.drop(col, axis=1, inplace=True)
    print(dataset.info())

    # Drop the duplicates
    dataset.drop_duplicates(inplace=True)
    ratings.drop_duplicates(inplace=True)

    # Drop null values from datasets
    col2 = ['genres', 'original_title', 'production_companies', 'release_date',
            'vote_average']
    dataset.dropna(subset=col2, how='any', inplace=True)

    # print(dataset.isnull().sum())

    # print(dataset.head(2))

    groupedby_movieName = dataset.groupby('original_title')
    # print movie names
    movies = dataset.groupby('original_title').size().sort_values(ascending=True)[:100]
    # print(movies)

    solace_data = groupedby_movieName.get_group('Solace')
    # print(solace_data.shape)

    # Find and visualize the user votes of the movie “Solace”
    plt.figure(figsize=(10, 10))
    plt.scatter(solace_data['original_title'], solace_data['vote_average'])
    plt.title('Plot showing the user rating of the movie “Solace_data”')
    plt.show()

    le = preprocessing.LabelEncoder()
    num_features = 4
    for i in range(num_features):
        dataset.iloc[:, i] = le.fit_transform(dataset.iloc[:, i])
    return dataset


def dump(data, filename):
    pickle.dump(data, open(filename, 'wb'))


def load(filename):
    return pickle.load(open(filename, 'rb'))


def load_dataset():
    return load('dataset.p')


def load_svr():
    return load('grid.p')


def load_random_forest():
    return load('random_forest.p')


def grid_search_svr(train, train_labels):
    parameters = {
        "kernel": ["rbf"],
        "C": [10, 100, 1000],
        "gamma": [1e-3, 1e-2, 1e-1]
    }

    grid = GridSearchCV(SVR(), parameters, verbose=2, n_jobs=-1, cv=2)
    grid.fit(train, train_labels)
    dump(grid, "grid.p")
    pprint(grid.cv_results_)
    return grid


def train_svr(train, train_labels, C=100, gamma=0.01):
    svr = SVR(C=C, gamma=gamma)
    svr.fit(train, train_labels)
    return svr


def test_svr(model: SVR, test, test_labels):
    y_pred = model.predict(test)
    print('\nC: ' + str(model.C) + ', Gamma: ' + str(model.gamma))
    print('SVR - Mean Absolute Error:', metrics.mean_absolute_error(test_labels, y_pred))
    print('SVR - Mean Squared Error:', metrics.mean_squared_error(test_labels, y_pred))
    print('SVR - Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))
    print('SVR - R2 metric:', r2_score(test_labels, y_pred))


def grid_search_random_forest(train, train_labels):
    parameters = {
        "n_estimators": [2000, 4000, 8000]
    }

    grid = GridSearchCV(RandomForestRegressor(), parameters, verbose=2, n_jobs=-1, cv=2)
    grid.fit(train, train_labels)
    dump(grid, "regressor.p")
    pprint(grid.cv_results_)
    return grid


def test_random_forest(regressor: RandomForestRegressor, test, test_labels):
    y_pred = regressor.predict(test)
    print("Random Forest", "Estimators - Mean Absolute Error:",
          metrics.mean_absolute_error(test_labels, y_pred))
    print("Random Forest", "Estimators - Mean Squared Error:",
          metrics.mean_squared_error(test_labels, y_pred))
    print("Random Forest", "Estimators - Root Mean Squared Error:",
          np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))
    print('Random Forest - R2 metric:', r2_score(test_labels, y_pred))


def main():
    train, test, train_labels, test_labels = load_dataset()

    # svr = grid_search_svr(train, train_labels)
    # svr: GridSearchCV = load_svr()
    # test_svr(svr.best_estimator_, test, test_labels)

    # random_forest = grid_search_random_forest(train, train_labels)
    # random_forest: GridSearchCV = load_random_forest()
    # test_random_forest(random_forest.best_estimator_, test, test_labels)

    test_svr(train_svr(train, train_labels, gamma=0.01), test, test_labels)
    test_svr(train_svr(train, train_labels, gamma=0.1), test, test_labels)
    test_svr(train_svr(train, train_labels, gamma=1), test, test_labels)
    test_svr(train_svr(train, train_labels, gamma=10), test, test_labels)
    test_svr(train_svr(train, train_labels, C=1000, gamma=0.01), test, test_labels)
    test_svr(train_svr(train, train_labels, C=1000, gamma=0.1), test, test_labels)
    test_svr(train_svr(train, train_labels, C=1000, gamma=1), test, test_labels)
    test_svr(train_svr(train, train_labels, C=1000, gamma=10), test, test_labels)


if __name__ == '__main__':
    main()
