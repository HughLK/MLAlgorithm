"""
完全随机选择样本点(xi, yi)
pocketPLA在迭代过程中不断更改存储的w
存储一个表现最好的w
"""

from randomPLA import getDataSet

import random
import math

def pocketPLA(sample, lable, updates, step):
	# times of update
	update = 0
	# w0 = 0 including b
	w = np.zeros((1, sample.shape[1]))
	w_pocket = np.zeros((1, sample.shape[1]))
	# num of sample
	num = sample.shape[0]
	# error rate of best w
	errorRate_wp = 1.0
	# index of sample
	index = range(num)
	# after update w a fixed times, finish the loop and return w
	while update < updates:
		i = random.choice(index)
		
		if label[i] * np.dot(w, sample[i]) <= 0:
			w = w + step * label[i] * sample[i]
			errorRate_w = getErrorRate(sample, label, w)
			# if current w perfomances better, updates
			if errorRate_w < errorRate_wp:
				w_pocket = w
				errorRate_wp = errorRate_w
				
			update += 1
	return w, errorRate_wp

def getErrorRate(sample, label, w):
	# num of sample
	num = sample.shape[0]
	# flatten w*sample to 1d
	dot_array = np.dot(w, sample.T).flatten()
	result_array = error_array * label
	
	# if result in array <= 0 return 1, which means there is an error
	error_array = np.where(result_array <= 0, 1, 0)
	return float(np.sum(error_array)) / float(num)

def avgErrorRate(sample, label, times, updates, step):
	errors = 0.0
	for i in range(times):
		w, error = pocketPLA(sample, label, updates, step)
		errors += error
		print(i)
	return w, errors / times

def verify(sample, label, w, times):
	errors = 0.0
	for i in range(times):
		errors += getErrorRate(sample, label, w)
		print(i)
	return errors / times

if __name__ == '__main__':
	sample, label = getDataSet('hw1_18_train.dat')
	w, errorRate = avgErrorRate(sample, label, 2000, 100, 0.5)
	print(errorRate)
