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
		# index starting at 0
		axis = depth % n
		# sorting dataset by axis
		sortedDataSet = dataSet[dataSet[:,axis].argsort()]

		node = Node(sortedDataSet[midIndex])
		leftDataSet = sortedDataSet[:midIndex]
		rightDataSet = sortedDataSet[midIndex + 1:]

		node.lchild = createKdTree(leftDataSet, depth + 1)
		node.rchild = createKdTree(rightDataSet, depth + 1)
		return node
	else:
		return

def searchKdTree(Node, point, k):
	global nearstNode
	global furthestNode_in_kNodes
	# dict that records k nearst nodes and its distance to point
	nearstNode = {}
	# furthest node's data in k nearst nodes 
	furthestNode_in_kNodes = -1

	def searchLeafNode(Node, depth = 0):
		global nearstNode
		global furthestNode_in_kNodes

		if Node == None:
			return
		else:
			# number of features
			n = len(point)
			axis = depth % n
			
			# searching for closest leaf node
			if point[axis] < Node.data[axis]:
				searchLeafNode(Node.lchild, depth + 1)
			elif point[axis] >= Node.data[axis]:
				searchLeafNode(Node.rchild, depth + 1)

			# if length of nearstNode is larger than 0, updates furthestNode_in_kNodes	
			if len(nearstNode) > 0:
				furthestNode_in_kNodes = sorted(nearstNode, key = lambda x: nearstNode[x])[-1]
			# distance between point and current Node
			distance = getDist(Node.data, point)
			# get the distance of furthest node in k nearst nodes returning infinity if there's no node in nearstNode
			furthestDist_in_kNodes = nearstNode.get(furthestNode_in_kNodes, float('inf'))
			# if distance is less than furthestDist_in_kNodes or there's less than k nodes in nearstNode, append current node
			if distance < furthestDist_in_kNodes or len(nearstNode) < k:
				nearstNode[tuple(Node.data)] = distance
				# if there's more than k nodes in nearstNode, delete the furthest node and update furthestNode_in_kNodes
				if len(nearstNode) > k:
					del nearstNode[furthestNode_in_kNodes]
					furthestNode_in_kNodes = sorted(dict, key = lambda x: dict[x])[-1]
					furthestDist_in_kNodes = nearstNode[furthestNode_in_kNodes]

			# check if there's a need to search another child filed of father node
			if abs(point[axis] - Node.data[axis]) <= furthestDist_in_kNodes:
				if point[axis] < Node.data[axis]:
					# check another child node
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
	nearst = searchKdTree(node, [2, 4.5], 3)
	print(nearst)
