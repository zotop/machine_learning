import numpy as np
from scipy.io import loadmat

def sigmoid(z):
	return float(1)/( 1 + np.exp(-z))

def h_theta(theta, x):
	return sigmoid(x.dot(theta))

def cost(theta, x, y, lambda_value):
	m = len(x)
	regularization_term = ( float(lambda_value)/(2*m) ) * (theta[1:theta.shape[0]]**2).sum()
	cost_value = 0
	h_theta_value = h_theta(theta, x)
	term1 = y.dot(np.log(h_theta_value).transpose())
	term2 = (1 - y).dot(np.transpose(np.log( 1 - h_theta_value )) )
	cost_value += (term1 + term2).sum()
	cost_value *=  (-float(1)/m)	
	return 	cost_value
 

data = loadmat('data.mat')
x = data['X']
y = data['y']

initial_theta = np.zeros((x.shape[1], 1))
print cost(initial_theta, x, y, 1)
