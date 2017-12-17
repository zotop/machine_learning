import numpy as np
from scipy.io import loadmat

def sigmoid(z):
	return float(1)/( 1 + np.exp(-z))

def cost(theta, x, y, lambda_value):
	m = len(x)
	h_theta_value = sigmoid(np.dot(x,theta))
	term1 = y.transpose().dot(np.log(h_theta_value))
	term2 = (1 - y).transpose().dot( np.log( 1 - h_theta_value) )
	cost_value = -(1./m) * (term1 + term2)
	regularization_term = float(lambda_value)/(2*m) * (theta[1:len(theta)]**2).sum()
	gradient = (1./m) * x.transpose().dot(h_theta_value - y) + ( float(lambda_value) / m ) * theta
	gradient_no_regularization = (1./m) * x.transpose().dot(h_theta_value - y)
	gradient[0] = gradient_no_regularization[0]
	
	return 	(cost_value + regularization_term).item(0), gradient.flatten()
