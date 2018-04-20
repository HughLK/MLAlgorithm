import numpy as np
import random
import pandas as pd

def getDataSet():
	dataSet = open('testSet.txt', 'r').readlines()
	lines = len(dataSet)
	sample = np.zeros((lines, 2))
	label = np.zeros(lines)
	for i in range(lines):
		data = dataSet[i].strip().split()
		sample[i] = data[:2]
		label[i] = data[-1]
	# add 1 in every row
	return np.hstack((np.ones((lines, 1)), sample)), label

def sigmod(w, x):
	multi_w_x = np.dot(w, x.T)
	return np.exp(multi_w_x) / (1 + np.exp(multi_w_x))

def logisticRegression(X, Y, threhold, MaxTimes, step):
	# num of sample
	num = X.shape[0]
	# initialize w
	w = np.zeros(X.shape[1])
	# index of X
	index = range(num)
	random.seed(10)

	for t in range(MaxTimes):
		# choose one sample randomly
		i = random.choice(index)
		# only update one w everytime
		w = w - step * (sigmod(w, X[i]) - Y[i]) * X[i]

	# probability distribution
	P = sigmod(w, X)
	# determing classes based on threhold
	P[P >= threhold] = 1
	P[P < threhold] = 0
	return P

if __name__ == '__main__':
	X, Y = getDataSet()
	p = logisticRegression(X, Y, 0.5, 2000, 0.1)
	print(p)
