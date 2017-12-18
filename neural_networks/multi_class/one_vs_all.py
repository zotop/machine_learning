from scipy.io import loadmat
import scipy.optimize as opt
import numpy as np
import cost_function as cf

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

for i in range(1, number_of_digits + 1):
	theta = opt.minimize(cf.cost, x0=initial_theta, args=(x, (y%10==i).astype(int).transpose()[0], lambda_value), options={'disp': False, 'maxiter':13}, method="Newton-CG", jac=True)
	thetas[i%10] = theta.x
	print "Done calculating thetas for digit {}".format(i%10)


