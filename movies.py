import pandas as pd


metadata = pd.read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)
metadata.head()

keywords = pd.read_csv("the-movies-dataset/keywords.csv", low_memory=False)
keywords.head()

ratings = pd.read_csv("the-movies-dataset/ratings.csv", low_memory=False)
ratings.head()
