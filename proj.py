#!/usr/bin/python2.7

import pandas as pd
import numpy
import pickle
import os.path
from collections import Counter

class RecSys:

  def __init__(self):
    self.training_error = []
    self.testing_error = []

  def user_base():
    pass

  def matrix_factorization(self, R, K, steps=10000, alpha=0.0002, beta=0.02):
    numMovie = len(R)
    numUser = len(R[0])
    P = numpy.random.rand(numMovie,K)
    Q = numpy.random.rand(numUser,K)

    Q = Q.T
    indexs = numpy.array(R.nonzero()).T
    for step in xrange(steps):
        for i,j in indexs:
          eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
          P[i,:] += alpha * (2 * eij * Q[:,j] - beta * P[i,:])
          Q[:,j] += alpha * (2 * eij * P[i,:] - beta * Q[:,j])
        e = 0
        for i,j in indexs:
          e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
        e += R.sum() + Q.sum()
        print "new", step, e
        if e < 0.001:
            break
    return P, Q.T


  # def matrix_factorization(self, R, K, steps=5000, alpha=0.0002, beta=0.02):
  #   numMovie = len(R)
  #   numUser = len(R[0])

  #   P = numpy.random.rand(numMovie,K)
  #   Q = numpy.random.rand(numUser,K)

  #   Q = Q.T
  #   for step in xrange(steps):
  #       for i in xrange(len(R)):
  #           for j in xrange(len(R[i])):
  #               if R[i][j] > 0:
  #                   eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
  #                   for k in xrange(K):
  #                       P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
  #                       Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
  #       # eR = numpy.dot(P,Q)
  #       e = 0
  #       for i in xrange(len(R)):
  #           for j in xrange(len(R[i])):
  #               if R[i][j] > 0:
  #                   e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
  #                   for k in xrange(K):
  #                       e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
  #       print step, e
  #       if e < 0.001:
  #           break
  #   return P, Q.T


if __name__ == '__main__':
  # links = pd.read_csv("data/links.csv")
  # tags = pd.read_csv("data/tags.csv")
  movies = pd.read_csv( "data/movies.csv" )
  genres = pd.unique( movies['genres'] )
  ratings = pd.read_csv( "data/ratings.csv" )
  ratings.rating = ratings.rating.astype(float)

  m1 = pd.merge( ratings, movies, on='movieId' )
  m = m1[ ['userId', 'movieId', 'rating', 'title'] ]
  m = m.pivot(index='movieId', columns='userId', values='rating').fillna(0.00)

  RatingMatrix = numpy.array(m)
  # numpy.random.shuffle(RatingMatrix)


  numFeature = 25 # len(genres)

  with open("MovieMatrix_10", 'rb') as f:
    MovieMatrix = pickle.load(f)
  with open("UserMatrix_10", 'rb') as f:
    UserMatrix = pickle.load(f)

  recsys = RecSys()

  MovieMatrix, UserMatrix = recsys.matrix_factorization(RatingMatrix, numFeature)
  # nR = numpy.dot(MovieMatrix, UserMatrix.T)



