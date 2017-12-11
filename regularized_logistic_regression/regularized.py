import matplotlib.pyplot as plt
import numpy as np
import mapfeature as mp
from scipy.optimize import minimize	 

def sigmoid(z):
	return float(1)/( 1 + np.exp(-z))

def cost(theta, x, y):
	n = len(x)
	cost_value = 0
	lambda_value = 1

	for i in range(n):
		sigmoid_value =  sigmoid(theta.transpose().dot(x[i, :]))
		term1 = y[i] * np.log(sigmoid_value)
		term2 = (1 - y[i]) * np.log(1 - sigmoid_value)
		cost_value += term1 + term2

	cost_value *=  (-float(1)/n)	
	regularization_term = ( float(lambda_value)/(2*n) ) * sum([theta[i]**2 for i in range(1, len(theta))] )
	return 	cost_value + regularization_term

#### MAIN ####

data = np.loadtxt('data.txt', delimiter=',') 
test_1 = data[:, 0:1]
test_2 = data[:, 1:2]
results = data[:, 2:3]

accepted = np.where(results == 1)[0]
accepted_test_1_result = test_1[accepted]
accepted_test_2_result = test_2[accepted]

rejected = np.where(results == 0)[0]
rejected_test_1_result = test_1[rejected]
rejected_test_2_result = test_2[rejected]

x = mp.map_feature(test_1, test_2)
y = results
initial_theta = np.zeros((x.shape[1], 1))

optimal_theta = minimize(cost, x0=initial_theta, args=(x,y), method='BFGS').x

print 'Cost with initial thetas: {}'.format(cost(initial_theta, x, y)[0])
print 'Cost with optimal thetas: {}'.format(cost(optimal_theta, x, y)[0])

plt.title('Microchip QA')
plt.xlabel('Test 1')
plt.ylabel('Test 2')
plt.scatter(accepted_test_1_result, accepted_test_2_result, marker='*', color='b', label='Accepted')
plt.scatter(rejected_test_1_result, rejected_test_2_result, marker='x', color='r', label='Rejected')
plt.legend(loc='upper right');
plt.show()    