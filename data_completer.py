import pandas as pd
from time import time
import numpy as np

def rework_metadata(A):

    df = pd.DataFrame(A, columns=['Movie', 'Genres'])
    genre_dummies = df['Genres'].str.get_dummies('|')
    genre_dummies = genre_dummies.drop('(no genres listed)', axis = 1)

    return genre_dummies.transpose()

def get_years(A):

    years = [int(movie.split('(')[-1].split(')')[0]) if '(' in movie else None for movie in A[:, 0]]
    years = [0 if year is None else year for year in years]
    mean = np.mean(years)
    years = [mean if year == 0 else year for year in years]
    years = np.array(years)

    return years

def distance_between_movies(A, years):

    intersection = A.transpose().dot(A)
    # Because it's binarized, the multiplication is the intersections.

    S = np.sum(A, axis = 0).values
    # Vector that counts the number of genres for each film

    U = np.ones(len(S))
    union = (S[:, np.newaxis]).dot(U[np.newaxis, :])
    # each line i is 4980 times the cardinal of genres for movies i
    union = union + union.transpose() - intersection

    years = (years[:, np.newaxis]).dot(U[np.newaxis, :])
    years = np.abs(years - years.transpose())/300
  
    # distances_matrix = 1 - intersection/(union + 1e-10) + years
    distances_matrix = 1 - intersection/(union + 1e-10)
    distances_matrix = distances_matrix.values

    return distances_matrix

def get_neighbors(distances_matrix, distance_max = 1e-10):

    neighbors = []
    for i, row in enumerate(distances_matrix):
        # List to store coordinates and values for the current row
        neighbors_of_row = []
        # Loop over elements in the row
        for j, value in enumerate(row):
            # Check if the value is under the threshold
            if value <= distance_max:
                # Append coordinates and value to the row result list
                neighbors_of_row.append(j)

        # Append the row result list to the main result list
        neighbors.append(neighbors_of_row)

    return neighbors

def add_ratings(R, neighbors):
    print('beginning of added values')

    added_values = np.full(R.shape, False)
    for i in range(R.shape[0]):
        begin = time()
        for j in range(R.shape[1]):
            if not np.isnan(R[i,j]):
                continue
            mean_ratings = 0
            seen_movies_in_neighborhood = 0
            for neighbor in neighbors[j]:
                if not np.isnan(R[i, neighbor]):
                    mean_ratings += R[i, neighbor]
                    seen_movies_in_neighborhood += 1
            if seen_movies_in_neighborhood != 0:
                mean_ratings /= seen_movies_in_neighborhood
                R[i,j] = mean_ratings
                added_values[i, j] = True

        end = time()
        t = end - begin
        print(i, t)
    return R, added_values