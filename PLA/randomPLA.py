"""
固定随机的PLA
更新开始前打乱样本
对打乱后的样本顺序检测是否有误
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

def randomPLA(sample, lable):
	# times allowed update
	updates = 0
	# error flag
	error = True
	# w0 = 0 including b
	w = np.zeros((1, sample.shape[1]))
	# num of sample
	num = sample.shape[0]
	# indexs of sample
	index = range(num)
	# shuffle indexs
	random.shuffle(index)

	# if there's no mistake in sample, finish the loop
	while error:
		error = False
		for i in index:
			if label[i] * np.dot(w, sample[i]) <= 0:
				w = w + label[i] * sample[i]
				error = True
				updates += 1
	return w, updates

def avgUpdates(sample, label, times):
	sum = 0
	for i in range(times):
		sum += randomPLA(sample, label)[1]
		print(i)
	return sum / times

if __name__ == '__main__':
	sample, label = getDataSet('hw1_15_train.dat')
	updates = avgUpdates(sample, label, 2000)
	print(updates)
