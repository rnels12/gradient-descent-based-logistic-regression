#!/usr/bin/env python3

"""
File name     : regression.py
Author        : Ryky Nelson
Created Date  : October 2021
Python Version: Python 3.6.9

Preceptron class:
trains the model, i.e. obtains the optimal weight vector (w) of the hyperplane 
that separates the data into two classes, i.e. the binary classification.
The minimization of the cost function is done through gradient desecent (GD).
"""

import numpy as np

__author__    = "Ryky Nelson"
__copyright__ = "Copyright 2021"


def sigmoid(z):
    return 1. / (1. + np.exp(-z))

class logisticRegression:
    def __init__(self, rate = 0.1, epsilon = 2.1e-8, maxiter = 300000):
        self.w       = np.array([])
        self.rate    = rate
        self.epsilon = epsilon
        self.maxiter = maxiter
        # parameter below used to stabilze the cost calculation because it bumps sometimes into log of zeros
        self.__eta = 1.e-10 
        
    def cost(self, h):
        return (\
            np.dot( self.ytrain, np.log( h + self.__eta ) ) + \
            np.dot( ( 1 - self.ytrain ), np.log( 1 - h + self.__eta ) ) \
        ) / self.ytrain.shape[0]

    def __grad_cost(self, h):
        return np.dot( (h - self.ytrain), self.xtrain ) \
            / self.ytrain.shape[0]
        
    def fit(self, x, y):        
        nrow, col = x.shape
        self.xtrain = np.concatenate( (  np.array( x ), np.ones((nrow,1)) ), axis=1 )
        self.ytrain = np.array( y )

        self.w = np.random.rand( col + 1 )
        # self.w = np.zeros( col + 1 ) # an alternative initialization

        it = 0
        h = sigmoid( np.dot( self.w, self.xtrain.T ) )
        cost_old = self.cost( h )
        while it < self.maxiter:
            self.w    -= self.rate * self.__grad_cost( h )
            h          = sigmoid( np.dot( self.w, self.xtrain.T ) )
            cost_new   = self.cost( h )            
            dcost = abs( cost_old - cost_new )
            if dcost < self.epsilon: break
            cost_old   = cost_new
            it += 1
            if (it == self.maxiter): print("not converged", dcost)
            
        return self.w

    def predict(self, x):
        nrow, col = x.shape
        self.xtest = np.concatenate( (  np.array( x ),\
                     np.ones((nrow,1)) ), axis=1 )
        h = sigmoid( np.dot( self.w, self.xtest.T ) )
        return [ 1 if ix >= 0.5 else 0 for ix in h ]
