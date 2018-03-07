import numpy as np
import math
import operator

def createDataSet():
	group = np.array([
			[2, 3],
			[5, 4],
			[9, 6],
			[4, 7],
			[8, 1],
			[7, 2]
			])
	labels = ['A', 'B', 'B', 'B', 'B', 'A']

	return group, labels

def classify0(inX, dataSet, labels, k):
	# tile inXs in rows(row number of dataSet) and one inX for each row
	inMat = np.tile(inX, (dataSet.shape[0], 1))
	differ = inMat - dataSet

	distance = np.array(map(lambda x:math.sqrt(x[0]**2 + x[1]**2), differ))
	# index of sorted distance array
	sortedDistanceIndex = distance.argsort()
	# record votes
	Count = {}

	for i in range(k):
		vote = labels[sortedDistanceIndex[i]]
		Count[vote] = Count.get(vote, 0) + 1
	# sort Count accroding to vote number
	sortedCount = sorted(Count.items(), key = operator.itemgetter(1), reverse = True)
	return sortedCount[0][0]

if __name__ == '__main__':
	dataSet, labels = createDataSet()
	result = classify0(np.array([1.0, 1.0]), dataSet, labels, 3)
	print(result)