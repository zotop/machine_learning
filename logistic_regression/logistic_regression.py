import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

def get_data(file_name):
  	exam_1_scores = []
	exam_2_scores = []
	admissions = []
	with open(file_name, "r") as lines:
		for line in lines:
			split_line = line.strip().split(',')
			exam_1_scores.append(float(split_line[0]))
			exam_2_scores.append(float(split_line[1]))
			admissions.append(float(split_line[2]))	
	return exam_1_scores, exam_2_scores, admissions		


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


def sigmoid(x, theta):
	z = -theta.transpose().dot(x)[0]
	return float(1)/( 1 + np.exp(-z))

def cost(n, x, y, theta):
	return (float(1)/n) * sum([-y[i] * np.log(sigmoid(x[i], theta)) - (1 - y[i]) * np.log(1 - sigmoid(x[i], theta)) for i in range(n)])

#### MAIN ####

exam_1_scores, exam_2_scores, admissions = get_data("data.txt")
admitted_exam_1_scores, admitted_exam_2_scores = get_admissions(admissions, exam_1_scores, exam_2_scores)
rejected_exam_1_scores, rejected_exam_2_scores = get_rejections(admissions, exam_1_scores, exam_2_scores)

n = len(admissions)
x0 = np.ones((n, 1))
x1 = np.column_stack([exam_1_scores])
x2 = np.column_stack([exam_2_scores])
x = np.column_stack((x0, x1, x2))
y = np.column_stack([admissions])
x_transpose = x.transpose()
number_of_features = 3


initial_theta = np.zeros((number_of_features, 1))
print 'Cost with initial thetas: {}'.format(cost(n, x, y, initial_theta)[0])

plt.title('Logistic Regression')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.scatter(admitted_exam_1_scores, admitted_exam_2_scores, marker='*', color='b', label='Admitted')
plt.scatter(rejected_exam_1_scores, rejected_exam_2_scores, marker='x', color='r', label='Rejected')
plt.legend(loc='upper right');
plt.show()    



