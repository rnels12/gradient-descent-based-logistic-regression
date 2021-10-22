#!/usr/bin/env python3

"""
File name     : main.py
Author        : Ryky Nelson
Created Date  : October 2021
Python Version: Python 3.6.9

Main function: 
- gets & processes the data
- separates the data into the training and test sets
- calls and feeds training data to the logistic regression (LR)
- measures the training LR against (sparred) test data
"""

import pandas as pd

from regression import logisticRegression as lr

if __name__ == "__main__":
    with  open("iris.data", 'r') as tr:
        data = pd.read_csv( tr, header=None )

    data  = data.sample(frac=1, random_state=20).reset_index(drop=True)

    ndata  = len(data)
    ntrain = int(0.80 * ndata)
    nfeat  = len( data.columns ) - 1

    training = data.loc[ [*range(ntrain)] ]
    test     = data.loc[ [*range(ntrain,ndata)] ]

    Xtrain = training[[0,1]]
    Ytrain = training[2].values.ravel()

    clf = lr(epsilon=5e-6, maxiter=10000)
    w = clf.fit(Xtrain, Ytrain)

    print("w = ", w)
    print("equation: ", str(-w[0]/w[1]) + ' * x + ' + str( -w[2] /w[1] ))

    Xtest = test[[0,1]]           
    Ytest = test[2].values.ravel()

    Ypred = clf.predict(Xtest)
    print( 'Accuracy = %3.1f%%' %\
           ( (Ytest == Ypred).sum() * 100 / len(Ytest) ) )
    
    
    
