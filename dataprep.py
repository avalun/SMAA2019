import pandas as pd
from sklearn import preprocessing


def prepare_dataset(plt=None):
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

    groupedby_moviename = dataset.groupby('original_title')
    # print movie names
    movies = dataset.groupby('original_title').size().sort_values(ascending=True)[:100]
    print(movies)

    solace_data = groupedby_moviename.get_group('Solace')
    print(solace_data.shape)

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
