import matplotlib.pyplot as plt

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


exam_1_scores, exam_2_scores, admissions = get_data("data.txt")
admitted_exam_1_scores, admitted_exam_2_scores = get_admissions(admissions, exam_1_scores, exam_2_scores)
rejected_exam_1_scores, rejected_exam_2_scores = get_rejections(admissions, exam_1_scores, exam_2_scores)


plt.title('Logistic Regression')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.scatter(admitted_exam_1_scores, admitted_exam_2_scores, marker='*', color='b', label='Admitted')
plt.scatter(rejected_exam_1_scores, rejected_exam_2_scores, marker='x', color='r', label='Rejected')
plt.legend(loc='upper right');
plt.show()    



