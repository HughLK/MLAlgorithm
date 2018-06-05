import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt

def getDataSet1():
	dataSet = open('testSet.txt', 'r').readlines()
	lines = len(dataSet)
	sample = np.zeros((lines, 2))
	label = np.zeros(lines)

	#split x and y from sample
	for i in range(lines):
		data = dataSet[i].strip().split()
		sample[i] = data[:2]
		label[i] = data[-1]
	# add x0 = 1
	return np.hstack((np.ones((lines, 1)), sample)), label

def getDataSet2():
	dataSet = pd.read_csv('./melon.csv')
   # add x0 = 1
	data = dataSet.drop('ripe', axis = 1).as_matrix()
	label = dataSet['ripe'].as_matrix()
    
	return np.hstack((np.ones((len(data), 1)), data)), label

def sigmod(w, x):
	multi_w_x = np.dot(w, x.T)
	return np.exp(multi_w_x) / (1 + np.exp(multi_w_x))

class LogisticRegression(object):
	# alg: choose which algrithm to run
	def __init__(self, alg = 'StochasticGradientDescent', threshold = 0.5, MaxTimes = 2000, Step = 0.1):
		self.alg = alg
		self.model = None
		self.paras = {'alg':'StochasticGradientDescent', 
				'threshold':threshold, 
				'MaxTimes':MaxTimes, 
				'Step':Step}

	def fit(self, X, Y):
		# get relating function by str alg
		alg_func = self.__getattribute__(self.alg)
		self.model = alg_func(X, Y)

	def predict(self, X_predict):
		w = self.model
		# probability distribution
		P = sigmod(w, X_predict)

		# get label by probability distribution and threshold
		P[P >= self.paras['threshold']] = 1
		P[P < self.paras['threshold']] = 0
		return P

	def predict_proba(self, X_predict):
		w, threshold = self.model
		# probability distribution
		P = sigmod(w, X_predict)
		return P
		
	def StochasticGradientDescent(self, X, Y):
		# num of sample
		num = X.shape[0]
		# initialize w
		w = np.zeros(X.shape[1])
		# index of X
		index = range(num)
		random.seed(10)

		for t in range(self.paras['MaxTimes']):
			# choose one sample randomly
			i = random.choice(index)
			# only update one w everytime
			w = w - self.paras['Step'] * (sigmod(w, X[i]) - Y[i]) * X[i]

		return w
    
	def get_params(self):
		return self.model

def draw(X, Y, model):
	# split smaple in two classes
	is_ripe = (Y == 1)
	X_1 = X[is_ripe]
	X_0 = X[~is_ripe]
	# plot samples
	plt.scatter(X_1[:,1], X_1[:,2], marker = 'o')
	plt.scatter(X_0[:,1], X_0[:,2], marker = 'x')
	
	weights = model.get_params()
	# identify value range of first dimension
	x = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.1)
	# calculate values of x on another dimension
	y = (-weights [0]-weights[1]*x)/weights[2]
	# plot split line
	plt.plot(x, y, color = 'r')
	plt.show()

if __name__ == '__main__':
	X, Y = getDataSet1()
	model = LogisticRegression(alg = 'StochasticGradientDescent')
	model.fit(X, Y)
	p = model.predict(X)
	print(p)
	if X.shape[1] == 3:
		draw(X, Y, model)