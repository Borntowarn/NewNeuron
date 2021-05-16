import numpy as np
from numpy.random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

def classifier(data, xtrain, ytrain, n_neighbours = 1):
	ytrain = np.array(ytrain)
	answers_data =[]
	distance = cdist(data, xtrain, 'euclidean')
	indexes = np.array([i.argsort() for i in distance[:]])
	for i, x in enumerate(data):
		a = np.where(ytrain[indexes[i, 0:n_neighbours]] == 0, 1, 0).sum()
		b = np.where(ytrain[indexes[i, 0:n_neighbours]] == 1, 1, 0).sum()
		c = np.where(ytrain[indexes[i, 0:n_neighbours]] == 2, 1, 0).sum()
		answers_data.append([a,b,c])
	return ([i.index(max(i)) for i in answers_data])
	
	


x1 = []
y1 = []
with open("C:\\Users\\kozlo\\source\\repos\\VSCODE PYTHON\\Neuron\\NewNeuron\\data.txt", "r") as f:
	for a in f:
		a = a.strip().split()
		x1.append(a[0:4])
		if a[4] == "Iris-setosa":
			y1.append(0)
		elif a[4] == "Iris-versicolor":
			y1.append(1)
		else: y1.append(2)

x_train, x_test, y_train, y_test = train_test_split(np.array(x1),np.array(y1), test_size=0.3, random_state=0)

print(np.where(classifier(x_test, x_train, y_train, n_neighbours = 5) == y_test, 1, 0).sum())