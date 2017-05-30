#!/usr/bin/python2.7

import pandas as pd
import numpy as np
import scipy
import pickle
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd


def matrix_factorization(R, testingIndex, K, steps=10000, alpha=0.0002, beta=0.02):
  numUser = len(R)
  numMovie = len(R[0])
  Q = np.random.rand(numMovie, K)
  P = np.random.rand(numUser, K)

  Q = Q.T
  indices = np.array(R.nonzero()).T
  trainError = []
  testingError = []
  for step in range(steps):
    for i, j in indices:
      eij = R[i][j] - np.dot(P[i, :], Q[:, j])
      P[i, :] += alpha * (2 * eij * Q[:, j] - beta * P[i, :])
      Q[:, j] += alpha * (2 * eij * P[i, :] - beta * Q[:, j])
    e = 0
    for i, j in indices:
      e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
    trainError.append(e)  # append the training error
    e += R.sum() + Q.sum()  # normalisation
    print(step, e)
    if e < 0.001:
      break




  with open("testingError_" + K, 'wb') as f:
    pickle.dump(testingError, f)
  with open("trainError_" + K, 'wb') as f:
    pickle.dump(trainError, f)
  with open("MovieMatrix_" + K, 'wb') as f:
    pickle.dump(P, f)
  with open("UserMatrix_" + K, 'wb') as f:
    pickle.dump(Q, f)
  return Q.T, P


def rmse (indexs, train, truth):
  for index in indexs:
    row, col, real = index[0], index[1], index[2]
    print(train[row].astype(int))
    print(truth[row].astype(int))
    print( train[row][col], truth[row][col] )
    input()


# mask out the first none zero value
# @return: (rows, cols, reals), [trainingMatrix]
def generate_test_train(ratingMatrix):
  ratingCopy = ratingMatrix.copy()
  rows, cols, reals = [],[],[]
  numUser = len(ratingMatrix)
  for row in range(0, int(numUser * 0.2)):  # split to test : 0->numUser/5, train: numUser/5->..
    nonezeros = ratingCopy[row].nonzero()
    if len(nonezeros[0]) > 1:
      col = nonezeros[0][0]
      rows.append(row)
      cols.append(col)
      reals.append(ratingCopy[row][col])
      ratingCopy[row][col] = 0
  return (np.array(rows), np.array(cols), np.array(reals)), ratingCopy


def main():
  movies = pd.read_csv("data/movies.csv")
  genres = pd.unique(movies['genres'])
  ratings = pd.read_csv("data/ratings.csv")
  ratings.rating = ratings.rating.astype(float)

  m1 = pd.merge(ratings, movies, on='movieId')
  m = m1[['userId', 'movieId', 'rating']]
  m = m.pivot(index='userId', columns='movieId', values='rating').fillna(0.00)

  ratingMatrix = np.array(m)
  numFeature = 600  # len(genres)

  testingIndex, trainingMatrix = generate_test_train(ratingMatrix)
  print(testingIndex)
  # matrix_factorization(trainingMatrix, testingIndex, numFeature)
  # print( np.mean(result, axis=1) )
  # rmse(testingIndex, result, ratingCopy)


if __name__ == '__main__':
  main()
