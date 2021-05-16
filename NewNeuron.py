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
	weights_for_dist = [(n_neighbours - i)/n_neighbours for i in range(n_neighbours)]
	v =[]
	for i, x in enumerate(data):
		s = [np.where(ytrain[indexes[i, 0:n_neighbours]] == j, weights_for_dist[0:n_neighbours], 0).sum() for j in np.unique(ytrain)] # считает количество соседей разных классов вблизи объекта
		v.append(s.index(max(s))) # выбирает класс, количество объектов вокруг наибольшее
	return (v)
	


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