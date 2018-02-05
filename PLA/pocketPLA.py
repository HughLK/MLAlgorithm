from naivePLA import *

import random
import math

def pocketPLA(sample, lable, updates):
	# times of wrong to correlect
	update = 0
	# w0 = 0
	w = np.zeros((1, sample.shape[1]))
	w_pocket = np.zeros((1, sample.shape[1]))
	# num of sample
	num = sample.shape[0]
	# initial error rate
	errorRate_wp = 1.0
	# index of sample
	index = range(num)

	while update < updates:
		i = random.choice(index)
		if label[i] * np.dot(w, sample[i]) <= 0:
		# if label[i] != sign(np.dot(w, sample[i])):
			w = w + label[i] * sample[i]
			errorRate_w = getErrorRate(sample, label, w)
			if errorRate_w < errorRate_wp:
				w_pocket = w
				errorRate_wp = errorRate_w
			update += 1
	return w, errorRate_wp

def getErrorRate(sample, label, w):
	# errors = 0.0
	# num = sample.shape[0]
	# for i in range(num):
	# 	if label[i] != sign(np.dot(w, sample[i])):
	# 		errors += 1
	# print(errors)
	# return errors / float(num)

	# num = sample.shape[0]
	# # after multiply, flatten the result to 1d and then turn to list
	# error_array_list = np.dot(w, sample.T).flatten().tolist()
	# # apply sign() to every element in list
	# error_array_list = list(map(lambda x: sign(x), error_array_list))
	# label_list = label.flatten().tolist()
	# error_array = list(map(lambda x: math.fabs(x[0] - x[1]) / 2, zip(error_array_list, label_list)))
	# # print(float(sum(error_array)))
	# # print(error_array)
	# return float(sum(error_array)) / float(num)
	
	# num = sample.shape[0]
	# error_array = np.dot(w, sample.T).flatten()
	# # apply sign() to every element in list
	# error_array = np.fromiter((sign(x) for x in error_array), error_array.dtype)
	# error_array = np.subtract(error_array, label.flatten())
	# error_array = np.fromiter((math.fabs(x) for x in error_array), error_array.dtype)
	# # print(float(sum(error_array)))
	# # print(error_array)
	# return float(np.sum(error_array) / 2) / float(num)
	
	num = sample.shape[0]
	error_array = np.dot(w, sample.T).flatten()
	# apply sign() to every element in array
	error_array = np.where(error_array > 0, 1, -1)
	error_array = np.subtract(error_array, label.flatten())
	# apply fabs() to every element in array
	error_array = np.fabs(error_array)
	return float(np.sum(error_array) / 2) / float(num)

def avgErrorRate(sample, label, times, updates):
	errors = 0.0
	for i in range(times):
		w, error = pocketPLA(sample, label, updates)
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
	w, errorRate = avgErrorRate(sample, label, 2000, 100)
	print(errorRate)

	# test, test_label = getDataSet('hw1_18_test.dat')
	# verifyErrorRate = verify(test, test_label, w, 2000)
	# print(errorRate)	
	# print(verifyErrorRate)
	
	# w = np.ones((1, sample.shape[0]))
	# s = np.dot(w, sample).flatten()
	# print(s)
	# x = np.array([[0], [2], [3], [4]])
	# f = lambda x: 1 if x > 0 else -1
	# # squares = np.fromiter((sign(xi) for xi in s), x.dtype)
	# squares = np.where(s > 0, 1, -1)
	# print(squares)
