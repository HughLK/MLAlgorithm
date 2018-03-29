import numpy as np
import math
import treePlotter

# class TreeNode(object):
# 	# Class records class of leaf node,feature records feature of non-leaf node
# 	def __init__(self, value = None, feature_index = None, left = None, right = None):
# 		self.left = left
# 		self.right = right
# 		self.value = value
# 		self.feature_index = feature_index

# 	def __repr__(self):
# 		if self.feature_index != None:
# 			return 'split at index:{}, value: {}'.format(self.feature_index, self.value)
# 		elif self.feature_index == None:
# 			return 'leaf node, value:{}'.format(self.value)

def createDataSet():
	data = np.array([
			[1, 4.5],
			[2, 4.75],
			[3, 4.91],
			[4, 5.34],
			[5, 5.8],
			[6, 7.05],
			[7, 7.9],
			[8, 8.23],
			[9, 8.7],
			[10, 9]
			])
	return data

def least_square(dataset):
	# create an array filled with c as same shape as y
	c_Mat = np.tile(getC(dataset), dataset.shape[0])
	return np.sum((dataset[:,-1] - c_Mat) ** 2)

# mean of values of all samples
def getC(dataset):
	# return round(np.mean(dataset[:,-1]), 3)
	return np.mean(dataset[:,-1], dtype = np.float32)

def dataSetSplit(dataset, col, val):
	return dataset[dataset[:,col] <= val], dataset[dataset[:,col] > val]

def getBestSplitPair(dataset):
	# if there's only one sample or several samples with same value in dataset, return mean of samples in this field
	if len(set(dataset[:,-1])) == 1:
		return None, getC(dataset)

	# index of best split feature
	bestJ = -1
	num_feature = dataset.shape[1] - 1

	for j in range(num_feature):
		minC = float('inf')
		# value of best split feature
		bestS = None

		for s in set(dataset[:,j]):
			subDataSet1, subDataSet2 = dataSetSplit(dataset, j, s)
			if len(subDataSet2) == 0:
				continue

			C = least_square(subDataSet1) + least_square(subDataSet2)
			if C < minC:
				minC = C
				bestS = s
				bestJ = j

	return bestJ, bestS

# def getRegressionTree(dataSet):
# 	feature_index, val = getBestSplitPair(dataSet)
# 	# if there's a leaf node, return its value
# 	if feature_index == None:
# 		return TreeNode(val)
	
# 	tree = TreeNode(val, feature_index)

# 	subDataSet1, subDataSet2 = dataSetSplit(dataSet, feature_index, val)
# 	tree.left = getRegressionTree(subDataSet1)
# 	tree.right = getRegressionTree(subDataSet2)
# 	return tree

def getRegressionTree(dataSet):
	feature_index, val = getBestSplitPair(dataSet)
	# if there's a leaf node, return its value
	if feature_index == None:
		return val
	
	# create root node
	tree = {(feature_index, val):{}}

	subDataSet1, subDataSet2 = dataSetSplit(dataSet, feature_index, val)
	tree[(feature_index, val)]['left'] = getRegressionTree(subDataSet1)
	tree[(feature_index, val)]['right'] = getRegressionTree(subDataSet2)
	return tree

def predict(tree, dataset):
	def searchTree(tree, data):
		# if tree is leaf node, return its value in this field
		if not isinstance(tree, dict):
			return tree

		feature_index, val = tree.keys()[0]
		# direciton showing that go to left node or right node
		direction = 'left' if data[feature_index] <= val else 'right'
		return searchTree(tree[ tree.keys()[0] ][direction], data)

	score = []
	for data in dataset:
		score.append(searchTree(tree, data))

	return np.sum(score)

if __name__ == '__main__':
	dataset = createDataSet()
	# half to train, half to test
	train, test = np.array_split(dataset, 2)
	tree = getRegressionTree(train)
	score = predict(tree, test)
	print(score)
	treePlotter.createPlot(tree)