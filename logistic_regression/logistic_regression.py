import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

def get_admissions(admissions, exam_1_scores, exam_2_scores):
	admitted_exam_1_scores = []
	admitted_exam_2_scores = []

	for i in range(0, len(admissions)):
		if admissions[i] == 1:
			admitted_exam_1_scores.append(exam_1_scores[i])
			admitted_exam_2_scores.append(exam_2_scores[i]) 
	
	return admitted_exam_1_scores, admitted_exam_2_scores		

def get_rejections(admissions, exam_1_scores, exam_2_scores):
	rejected_exam_1_scores = []
	rejected_exam_2_scores = []

	for i in range(0, len(admissions)):
		if admissions[i] == 0:
			rejected_exam_1_scores.append(exam_1_scores[i])
			rejected_exam_2_scores.append(exam_2_scores[i])

	return rejected_exam_1_scores, rejected_exam_2_scores		 

def sigmoid(z):
	return float(1)/( 1 + np.exp(-z))

def cost(theta, x, y):
	n = len(x)
	cost_value = 0

	for i in range(n):
		sigmoid_value =  sigmoid(theta.transpose().dot(np.column_stack([x[i]])))
		term1 = y[i] * np.log(sigmoid_value)
		term2 = (1 - y[i]) * np.log(1 - sigmoid_value)
		cost_value = cost_value + term1 + term2

	cost_value = cost_value * (-float(1)/n)	
	return 	cost_value


def predict(theta, score_1, score_2):
	x = [1, score_1, score_2]
	z = theta.transpose().dot(x)
	return sigmoid(z)


#### MAIN ####

data = np.loadtxt('data.txt', delimiter=',') 
exam_1_scores = data[:, 0:1]
exam_2_scores = data[:, 1:2]
admissions = data[:, 2:3]

admitted_exam_1_scores, admitted_exam_2_scores = get_admissions(admissions, exam_1_scores, exam_2_scores)
rejected_exam_1_scores, rejected_exam_2_scores = get_rejections(admissions, exam_1_scores, exam_2_scores)

n = len(admissions)
x = np.concatenate((np.ones((n, 1)) , exam_1_scores, exam_2_scores), axis=1) 
y = np.column_stack([admissions])
number_of_features = 3

initial_theta = np.zeros((number_of_features, 1))
print 'Cost with initial thetas: {}'.format(cost(initial_theta, x, y)[0])

optimal_theta = fmin(cost, x0=initial_theta, args=(x,y))
print 'Optimal Thetas: {}'.format(optimal_theta)

print 'Cost with optimal thetas: {}'.format(cost(optimal_theta, x, y)[0])

print 'Prediction: {}'.format(predict(optimal_theta, 45, 85))

plt.title('Logistic Regression')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.scatter(admitted_exam_1_scores, admitted_exam_2_scores, marker='*', color='b', label='Admitted')
plt.scatter(rejected_exam_1_scores, rejected_exam_2_scores, marker='x', color='r', label='Rejected')
plt.legend(loc='upper right');
plt.show()    