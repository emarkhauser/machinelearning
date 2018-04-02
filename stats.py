# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Modelling:

	# Split into training and validation datasets
	def __init__(self, X, Y):
		self.X_train, self.X_val, self.Y_train, self.Y_val = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)

	# Run models
	def run(self):
		# Spot Check Algorithms
		models = []
		models.append(('LR', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		# evaluate each model in turn
		results = []
		names = []
		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=7)
			cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring='accuracy')
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			print(msg)

		# Compare Algorithms
		fig = plt.figure()
		fig.suptitle('Algorithm Comparison')
		ax = fig.add_subplot(111)
		plt.boxplot(results)
		ax.set_xticklabels(names)
		plt.show()
