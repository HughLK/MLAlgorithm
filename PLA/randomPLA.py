from naivePLA import *
import random

def randomPLA(sample, lable):
	# times of wrong to correlect
	updates = 0
	error = True
	# w0 = 0
	w = np.zeros((1, sample.shape[1]))
	# num of samplt
	num = sample.shape[0]
	# index of sample
	index = range(num)
	# shuffle the index
	random.shuffle(index)

	# if there's no mistake in sample, halts
	while error:
		error = False
		for i in index:
			if label[i] * np.dot(w, sample[i]) <= 0:
			# if label[i] != sign(np.dot(w, sample[i])):
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