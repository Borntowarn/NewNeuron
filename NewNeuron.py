import numpy as np
from numpy.random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def classifier(data, xtrain, ytrain, h, n_neighbours = 100):
	ytrain = np.array(ytrain)
	answers_data =[]
	distance = cdist(data, xtrain, 'euclidean')
	indexes = np.array([i.argsort() for i in distance[:]])
	distance /= h # h - размер окна Парзена
	weights_ker = np.exp(-2*distance**2) # инфинитное гауссово ядро
	v = []
	for i, x in enumerate(data):
		s = [np.where(ytrain[indexes[i, 0:n_neighbours]] == j, weights_ker[i, indexes[i, 0:n_neighbours]], 0).sum() for j in np.unique(ytrain)] # считает количество соседей разных классов вблизи объекта
		v.append(s.index(max(s))) # выбирает класс, вес объектов которого больше воздействует на объект
		a = 0
		b = 0
		c = 0
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

print(np.where(classifier(x_test, x_train, y_train, h=1,  n_neighbours = 100) == y_test, 1, 0).sum())