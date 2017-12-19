from scipy.io import loadmat
import numpy as np

def sigmoid(z):
	return float(1)/( 1 + np.exp(-z))

data = loadmat('data.mat')
weights_data = loadmat('weights.mat')

x = data['X']
y = data['y'].flatten()
m,n = x.shape
theta_1 = weights_data['Theta1']
theta_2 = weights_data['Theta2']

x = np.column_stack((np.ones((m,1)), x))
hidden_layer = sigmoid(x.dot(theta_1.transpose()))

hidden_layer = np.column_stack((np.ones((m,1)), hidden_layer))
output_layer = sigmoid(hidden_layer.dot(theta_2.transpose()))

output_layer = np.argmax(output_layer, axis=1) + 1
print "Accuracy: {} %".format(np.mean(output_layer == y))


