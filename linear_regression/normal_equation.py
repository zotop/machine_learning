import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


class Data(object):
  def __init__(self, x, y):
  	self.x = x
  	self.y = y	


def get_data(file_name):
  	x = []
	y = []
	with open(file_name, "r") as lines:
		for line in lines:
			split_line = line.strip().split(',')
			x.append(float(split_line[0]))
			y.append(float(split_line[1]))	
	return Data(x,y)		

def cost(o0,o1,x,y):
	return (float(1)/(2 * n)) * sum([(o0 + o1 * x[i] - y[i])**2 for i in range(n)])	

data = get_data("data.txt")
n = len(data.x)
x0 = np.ones((n,1))
x1 = np.column_stack([data.x])
x = np.column_stack((x0, x1))
x_transpose = x.transpose()
y = np.column_stack([data.y])

theta_matrix = inv(x_transpose.dot(x)).dot(x_transpose).dot(y)

theta_0 = theta_matrix[0][0]
theta_1 = theta_matrix[1][0]
cost = cost(theta_0, theta_1, data.x, data.y)

print 'Theta 0: {}'.format(theta_0)
print 'Theta 1: {}'.format(theta_1)
print 'Cost: {}'.format(cost)



 



