import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)
print(dataset.head())
print(dataset.info())
# print(dataset.describe())

ratings = pd.read_csv("the-movies-dataset/ratings.csv", low_memory=False)
print(ratings.head())
print(ratings.info())

# After discussing the structure of the data and any problems that need to be
# Cleaned, perform those cleaning steps in the second part of this section.
# Drop extraneous columns
col = ['adult', 'belongs_to_collection', 'homepage', 'original_language', 'overview', 'popularity', 'poster_path',
       'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'video', 'production_countries','vote_count']
dataset.drop(col, axis=1, inplace=True)
print(dataset.info())

# Drop the duplicates
dataset.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

# Drop null values from datasets
col2 = ['budget', 'genres', 'id', 'original_title', 'imdb_id', 'production_companies', 'release_date', 'revenue',
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
num_features = 9
for i in range(num_features):
    dataset.iloc[:, i] = le.fit_transform(dataset.iloc[:, i])

# Create train and test data set

x = dataset.iloc[0:500, :-1].values
y = dataset.iloc[0:500, -1].values
train, test, train_labels, test_labels = train_test_split(x, y, test_size=0.33, random_state=30)

# Svm training
svc = SVC()
svc.fit(train, train_labels)
Y_pred = svc.predict(test)
print(dataset.production_companies)
acc_svc = round(svc.score(train, train_labels) * 100, 2)
print(acc_svc)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train, train_labels)
Y_pred = random_forest.predict(test)
random_forest.score(train, train_labels)
acc_random_forest = round(random_forest.score(train, train_labels) * 100, 2)
print(acc_random_forest)
