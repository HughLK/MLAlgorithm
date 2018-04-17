"""
每次随机检测样本是否有误
循环指定次数
"""

import random

def getDataSet(fileName):
	dataSet = open(fileName, 'r').readlines()
	lines = len(dataSet)
	sample = np.zeros((lines, 5))
	label = np.zeros(lines)
	
	for i in range(lines):
		data = dataSet[i].strip().split()
		
		sample_data = data[:4]
		# add x0 = 1
		sample_data.insert(0, 1)
		sample[i] = sample_data
		
		label[i] = data[-1]
	return sample, label

def randomPLA(sample, lable, times,step):
	# w0 = 0 including b
	w = np.zeros((1, sample.shape[1]))
	# num of sample
	num = sample.shape[0]
	# indexs of sample
	index = range(num)
	random.seed(10)

	for t in range(times):
		# choose index randomly
		i = random.choice(index)
		if label[i] * np.dot(w, sample[i]) <= 0:
			w = w + step * label[i] * sample[i]
	return w

if __name__ == '__main__':
	sample, label = getDataSet('hw1_15_train.dat')
	w = randomPLA(sample, label, 2000, 0.5)
	print(w)
