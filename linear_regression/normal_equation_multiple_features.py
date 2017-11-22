#NOTE: no need to feature scale

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


class Data(object):
  def __init__(self, x1, x2, y):
  	self.x1 = x1
  	self.x2 = x2
  	self.y = y


def get_data(file_name):
	x1 = []
	x2 = []
	y = []
	with open(file_name, "r") as lines:
		for line in lines:
			split_line = line.strip().split(',')
			x1.append(float(split_line[0]))
			x2.append(float(split_line[1]))
			y.append(float(split_line[2]))	
	
	return Data(x1, x2, y) 	

def hypothesis(thetas, x):
	return thetas.transpose().dot(x)	

data = get_data("data2.txt")	

n = len(data.x1)
x0 = np.ones((n,1))
x1 = np.column_stack([data.x1])
x2 = np.column_stack([data.x2])
x = np.column_stack((x0, x1, x2))
x_transpose = x.transpose()
y = np.column_stack([data.y])

thetas = inv(x_transpose.dot(x)).dot(x_transpose).dot(y)

sample_x = np.array([[1, 2300, 3]]).transpose()

print 'Thetas : {}'.format(thetas)
print 'For x1=2300 and x2=3, y={}'.format(hypothesis(thetas, sample_x))
