import numpy as np
import math
import treePlotter

def createDataSet():
	data = np.array([
			['youth', 'no', 'no', 'normal', 'no'],
			['youth', 'no', 'no', 'good', 'no'],
			['youth', 'yes', 'no', 'good', 'yes'],
			['youth', 'yes', 'yes', 'normal', 'yes'],
			['youth', 'no', 'no', 'normal', 'no'],
			['mid_age', 'no', 'no', 'normal', 'no'],
			['mid_age', 'no', 'no', 'good', 'no'],
			['mid_age', 'yes', 'yes', 'good', 'yes'],
			['mid_age', 'no', 'yes', 'excellent', 'yes'],
			['mid_age', 'no', 'yes', 'excellent', 'yes'],
			['old', 'no', 'yes', 'excellent', 'yes'],
			['old', 'no', 'yes', 'good', 'yes'],
			['old', 'yes', 'no', 'good', 'yes'],
			['old', 'yes', 'no', 'excellent', 'yes'],
			['old', 'no', 'no', 'normal', 'no']
			])
	features = np.array(['Age', 'HasJobs', 'HasOwnHouse', 'Credit'])
	return data, features

# col: index of selected feature
def getFeaturesVal(dataset, col):
	return list(set(dataset[:,col].flatten().tolist()))

def getEntropyD(dataset):
	# entropy of D
	H = 0
	# number of samples
	n = dataset.shape[0]
	# ys
	labels = dataset[:,-1].flatten()
	# value of label of D
	values = getFeaturesVal(dataset, -1)

	for val in values:
		p = np.sum(labels == val) / float(n)
		H += -(p * math.log(p, 2))

	return H

def getEntropyDOnA(dataset, feature_index):
	# entropy of D On A
	H = 0
	# number of samples
	n = dataset.shape[0]
	# values of one feature of Di
	values = getFeaturesVal(dataset, feature_index)

	for val in values:
		Di = dataset[dataset[:,feature_index] == val]
		H +=  Di.shape[0] / float(n) * getEntropyD(Di)

	return H

def getEntropyDofA(dataset, feature_index):
	# entropy of D of A
	H = 0
	# number of samples
	n = dataset.shape[0]
	# values of label of Di
	values = getFeaturesVal(dataset, feature_index)

	for val in values:
		p = dataset[dataset[:,feature_index] == val].shape[0] / float(n)
		H += -(p * math.log(p, 2))

	return H

def informationGainRatio(dataset, features, selected_feature):
	feature_index = features.tolist().index(selected_feature)
	return (getEntropyD(dataset) - getEntropyDOnA(dataset, feature_index)) / getEntropyDofA(dataset, feature_index)

def getClassOfTheMostInD(dataset):
	# values of lable of dataset
	values = getFeaturesVal(dataset, -1)
	# number of samples in all classes
	num_classes = np.array([len(dataset[dataset[:,-1] == val]) for val in values])
	# np.where returns a tuple including all coordinates which show as a np.array
	index = np.where(num_classes == np.max(num_classes))[0][0]

	return values[index]

def DecisionTree(dataset, features, threshold):
	# values of lable of dataset
	values = getFeaturesVal(dataset, -1)
	# number of features
	num_features = len(features)

	# if samples in dataset belong to one class
	if len(values) == 1:
		return values[0]

	# if all features are uesd
	if num_features == 0:
		return getClassOfTheMostInD(dataset)

	# information gain ratios of all features
	information_gain_ratios = np.array([informationGainRatio(dataset, features, features[i]) for i in range(num_features)])
	# index of max information gain ratios
	max_index = np.where(information_gain_ratios == np.max(information_gain_ratios))[0][0]

	if information_gain_ratios[max_index] < threshold:
		return getClassOfTheMostInD(dataset)

	max_igr_values = getFeaturesVal(dataset, max_index)
	# split dataset by feature with max igr
	subDataSet = [np.delete(dataset[dataset[:,max_index] == val], max_index, axis = 1) for val in max_igr_values]
	# features removing feature with max igr
	subFeatures = np.delete(features, max_index)
	# use dict to store a decision tree
	tree = {features[max_index] : {}}

	for i in range(len(max_igr_values)):
		tree[ features[max_index] ][max_igr_values[i]] = DecisionTree(subDataSet[i], subFeatures, threshold)

	return tree

def predict(tree, features, dataset):
	def searchTree(tree, features, data):
		# if tree is leaf node, return its class
		if not isinstance(tree, dict):
			return tree

		node_feature = tree.keys()[0]
		# index of feature stored in the node
		index = features.tolist().index(node_feature)
		return searchTree(tree[node_feature][data[index]], features, data)

	class_array = []
	for data in dataset:
		class_array.append(searchTree(tree, features, data))

	return class_array

def getErrorRate(predict_class, dataset):
	true_class = dataset[:, -1]
	diff = (true_class != predict_class)
	return np.mean(diff) if len(diff) != 0 else 0

if __name__ == '__main__':
	dataset, features = createDataSet()
	# half to train, half to test
	train, test = np.array_split(dataset, 2)
	tree = DecisionTree(train, features, 0.1)
	treePlotter.createPlot(tree)
	predict_class = predict(tree, features, test)
	print('predict:', predict_class)
	score = getErrorRate(predict_class, test)
	print('error rate:', score)
