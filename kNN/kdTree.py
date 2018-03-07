import numpy as np
import math

class Node:  
	def __init__(self, data, lchild = None, rchild = None):  
		self.data = data  
		self.lchild = lchild  
		self.rchild = rchild
	
def createKdTree(dataSet, depth):
	if len(dataSet) > 0:
		m, n = dataSet.shape
		midIndex = m / 2
		# index start by 0
		axis = depth % n
		# sort dataset by axis
		sortedDataSet = dataSet[dataSet[:,axis].argsort()]

		node = Node(sortedDataSet[midIndex])
		leftDataSet = sortedDataSet[:midIndex]
		rightDataSet = sortedDataSet[midIndex + 1:]

		node.lchild = createKdTree(leftDataSet, depth + 1)
		node.rchild = createKdTree(rightDataSet, depth + 1)
		return node
	else:
		return

def searchKdTree(Node, point):
	global nearstNode
	global nearstDist
	nearstNode = None
	# distance between nearstNode and point
	nearstDist = float('inf')

	def searchLeafNode(Node, depth = 0):
		global nearstNode
		global nearstDist
		
		if Node == None:
			return
		else:
			# number of features
			n = len(point)
			axis = depth % n

			# search for closest leaf node
			if point[axis] < Node.data[axis]:
				searchLeafNode(Node.lchild, depth + 1)
			elif point[axis] >= Node.data[axis]:
				searchLeafNode(Node.rchild, depth + 1)

			# calculate distance between nearstNode and point
			distance = getDist(Node.data, point)
			# if current node is closer than nearstNode, update nearstNode and nearstDist
			if distance < nearstDist:
				nearstNode = Node
				nearstDist = distance

			# check if round with radius of nearstDist and axis intersects
			if abs(point[axis] - Node.data[axis]) <= nearstDist:
				if point[axis] < Node.data[axis]:
					searchLeafNode(Node.rchild, depth + 1)
				elif point[axis] >= Node.data[axis]:
					searchLeafNode(Node.lchild, depth + 1)

	searchLeafNode(Node)
	return nearstNode

def getDist(x1, x2):
	return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5

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

if __name__ == '__main__':
	dataSet, labels = createDataSet()
	node = createKdTree(dataSet, 2)
	nearst = searchKdTree(node, [2.1, 3.1])
	print(nearst.data)
