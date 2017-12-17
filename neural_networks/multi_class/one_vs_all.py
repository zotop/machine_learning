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

min_result = opt.minimize(cf.cost, x0=initial_theta, args=(x, (y%10==0).astype(int).transpose()[0], lambda_value), options={'disp': True, 'maxiter':13}, method="Newton-CG", jac=True)