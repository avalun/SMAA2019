import pandas as pd


movies = pd. read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)
movies.head()
