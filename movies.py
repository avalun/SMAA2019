import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def prepare_dataset():
    dataset = pd.read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)
    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())

    ratings = pd.read_csv("the-movies-dataset/ratings.csv", low_memory=False)
    print(ratings.head())
    print(ratings.info())

    # After discussing the structure of the data and any problems that need to be
    # Cleaned, perform those cleaning steps in the second part of this section.
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

    print(dataset.isnull().sum())

    print(dataset.head(2))

    groupedby_movieName = dataset.groupby('original_title')
    # print movie names
    movies = dataset.groupby('original_title').size().sort_values(ascending=True)[:100]
    print(movies)

    Solace_data = groupedby_movieName.get_group('Solace')
    print(Solace_data.shape)

    # Find and visualize the user votes of the movie “Solace”
    plt.figure(figsize=(10, 10))
    plt.scatter(Solace_data['original_title'], Solace_data['vote_average'])
    plt.title('Plot showing  the user rating of the movie “Solace_data”')
    plt.show()

    le = preprocessing.LabelEncoder()
    num_features = 4
    for i in range(num_features):
        dataset.iloc[:, i] = le.fit_transform(dataset.iloc[:, i])
    return dataset


def dump(data, filename='dump.p'):
    pickle.dump(data, open(filename, 'wb'))


def load(filename='dump.p'):
    return pickle.load(open(filename, 'rb'))


def svm(train, test, train_labels, test_labels):
    # Svm training
    svc = SVR(kernel="rbf", C=1e3, gamma=1e-8, epsilon=0.1)
    # svc = SVR(kernel="rbf", C=10000, gamma=1e-8, epsilon=0.1)

    svc.fit(train, train_labels)
    Y_pred = svc.predict(test)
    print(Y_pred)
    print(test_labels)


# Random forest training
def random_forest(train, test, train_labels, test_labels, n_estimators=1000):
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    regressor.fit(train, train_labels)
    y_pred = regressor.predict(test)
    regressor.score(train, train_labels)
    acc_random_forest = round(regressor.score(train, train_labels) * 100, 2)
    print("Random Forest ", n_estimators, " Estimators - Accuracy", acc_random_forest)
    print("Random Forest ", n_estimators, " Estimators- Mean Absolute Error:",
          metrics.mean_absolute_error(test_labels, y_pred))
    print("Random Forest ", n_estimators, " Estimators - Mean Squared Error:",
          metrics.mean_squared_error(test_labels, y_pred))
    print("Random Forest ", n_estimators, " Estimators - Root Mean Squared Error:",
          np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))


def main():
    # Create train and test data set
    # dataset = prepare_dataset()
    # dump(dataset)
    dataset = load()
    x = dataset.iloc[0:500, :-1].values
    y = dataset.iloc[0:500, -1].values
    train, test, train_labels, test_labels = train_test_split(x, y, test_size=0.33, random_state=25)
    svm(train, test, train_labels, test_labels)

    # Random Forests with different numbers of estimators
    # Larger number of n_estimators become useful for bigger datasets
    # so enable the lower lines for the final testing!
    random_forest(train, test, train_labels, test_labels, 500)
    # random_forest(train, test, train_labels, test_labels, 1000)
    # andom_forest(train, test, train_labels, test_labels, 2000)
    # random_forest(train, test, train_labels, test_labels, 4000)


if __name__ == '__main__':
    main()
    exit()
