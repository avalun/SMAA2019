import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import preprocessing

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
num_features = 5
for i in range(num_features):
    dataset.iloc[:, i] = le.fit_transform(dataset.iloc[:, i])

# Create train and test data set

x = dataset.iloc[0:500, :-1].values
y = dataset.iloc[0:500, -1].values
train, test, train_labels, test_labels = train_test_split(x, y, test_size=0.33, random_state=25)

# Svm training
svc = SVR(kernel="rbf", C=1e3, gamma=1e-8, epsilon=0.1)
svc.fit(train, train_labels)
Y_pred = svc.predict(test)
acc_svc = round(svc.score(test, test_labels) * 100, 2)
print(acc_svc)
print(accuracy_score)
print(Y_pred)
# Random Forest
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(train, train_labels)
# Y_pred = random_forest.predict(test)
# random_forest.score(train, train_labels)
# acc_random_forest = round(random_forest.score(train, train_labels) * 100, 2)
# print(acc_random_forest)


# another
# regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# regressor.fit(train, train_labels)
# test_labels = regressor.predict(test)

# print(confusion_matrix(test,test_labels))
# print(classification_report(test,test_labels))
# print(accuracy_score(test, test_labels))
#
# rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
# rf.fit(train, train_labels)
#
# pred = rf.predict(test)
#
# print('Random Forest - Mean Absolute Error:', metrics.mean_absolute_error(test, test_labels))
# print('Random Forest - Mean Squared Error:', metrics.mean_squared_error(test, test_labels))
# print('Random Forest - Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test, test_labels)))
