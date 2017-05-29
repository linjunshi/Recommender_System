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
  
  def matrix_factorization(self, rating, numFeature, steps=10000, alpha=0.0006, beta=0.02):
    numMovie = len(rating)
    numUser = len(rating[0])

    movie = numpy.random.rand(numMovie,numFeature)
    user = numpy.random.rand(numUser,numFeature)

    user = user.T
    preve = 0
    for step in xrange(steps):
      for mid in xrange(len(rating)): # update
        for uid in xrange(len(rating[mid])):
          if rating[mid][uid] > 0:
            eij = rating[mid][uid] - numpy.dot(movie[mid,:],user[:,uid])
            for f in xrange(numFeature):
              movie[mid][f] = movie[mid][f] + alpha * (2 * eij * user[f][uid] - beta * movie[mid][f])
              user[f][uid] = user[f][uid] + alpha * (2 * eij * movie[mid][f] - beta * user[f][uid])
      e = 0
      for i in xrange(len(rating)): # error
        for j in xrange(len(rating[i])):
          if rating[i][j] > 0:
            e = e + pow(rating[i][j] - numpy.dot(movie[i,:],user[:,j]), 2)
            for f in xrange(numFeature): # normalisation
              e = e + (beta/2) * (pow(movie[i][f],2) + pow(user[f][j],2))
    
      # debug
      print "%s: e = %.10f, preve = %.10f, diff = %.10f\n" % (step, e, preve, (preve - e))
      if abs(e - preve) < 0.0001:
        break
      preve = e

    with open("MovieMatrix_"+numFeature, 'wb') as f:
      pickle.dump(movie, f)

    with open("UserMatrix_"+numFeature, 'wb') as f:
      pickle.dump(user.T, f)

    return movie, user.T


"""  
  def matrix_factorization(self, R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        # eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        print step, e
        if e < 0.001:
            break
    return P, Q.T
"""


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

  numFeature = 2 # len(genres)

  # with open("MovieMatrix", 'rb') as f:
  #   MovieMatrix = pickle.load(f)
  # with open("UserMatrix", 'rb') as f:
  #   UserMatrix = pickle.load(f)

  recsys = RecSys()

  # MovieMatrix, UserMatrix = recsys.matrix_factorization(RatingMatrix, numFeature)
  # nR = numpy.dot(MovieMatrix, UserMatrix.T)

