from scipy.io import loadmat
import scipy.optimize as opt
import numpy as np
import cost_function as cf

def sigmoid(z):
	return float(1)/( 1 + np.exp(-z))

data = loadmat('data.mat')
x = data['X']
y = data['y']
m, n = x.shape
x = np.column_stack((np.ones((m,1)), x))
initial_theta = np.zeros((x.shape[1], 1))
lambda_value = 0.1
initial_theta = np.zeros((n + 1, 1))

number_of_digits = 10
thetas = np.zeros((number_of_digits, n + 1))

for i in range(number_of_digits):
	theta = opt.minimize(cf.cost, x0=initial_theta, args=(x, (y%10==i).astype(int).transpose()[0], lambda_value), options={'disp': False, 'maxiter':13}, method="Newton-CG", jac=True)
	thetas[i] = theta.x
	print "Done calculating thetas for digit {}. Cost: {}".format(i, theta.fun)

predictions =  sigmoid(x.dot(thetas.transpose()))
predictions = np.argmax(predictions, axis=1)
y = y.flatten()

print('Accuracy: {} %').format(np.mean(predictions == y%10) * 100)
print('Accuracy for digit 0: {} %'.format(np.mean(predictions[0:500] 	 == y[0:500]    %10) * 100))
print('Accuracy for digit 1: {} %'.format(np.mean(predictions[500:1000]  == y[500:1000] %10) * 100))
print('Accuracy for digit 2: {} %'.format(np.mean(predictions[1000:1500] == y[1000:1500]%10) * 100))
print('Accuracy for digit 3: {} %'.format(np.mean(predictions[1500:2000] == y[1500:2000]%10) * 100))
print('Accuracy for digit 4: {} %'.format(np.mean(predictions[2000:2500] == y[2000:2500]%10) * 100))
print('Accuracy for digit 5: {} %'.format(np.mean(predictions[2500:3000] == y[2500:3000]%10) * 100))
print('Accuracy for digit 6: {} %'.format(np.mean(predictions[3000:3500] == y[3000:3500]%10) * 100))
print('Accuracy for digit 7: {} %'.format(np.mean(predictions[3500:4000] == y[3500:4000]%10) * 100))
print('Accuracy for digit 8: {} %'.format(np.mean(predictions[4000:4500] == y[4000:4500]%10) * 100))
print('Accuracy for digit 9: {} %'.format(np.mean(predictions[4500:5000] == y[4500:5000]%10) * 100))