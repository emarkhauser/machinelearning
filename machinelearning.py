import pandas
import numpy
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.pipeline import make_union, make_pipeline
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
import matplotlib.pyplot as plt

class MachineLearning:

	data = None
	Y = None
	X = None
	test_set_size = 0.2
	k_fold_splits = 10

	def __init__(self, dataFile):
		self.data = pandas.read_csv(dataFile)
		self.data.replace([numpy.inf, -numpy.inf], numpy.nan)

	def cleanIntervalData(self, vars):
		transformers = make_union(
			Imputer(missing_values='NaN', strategy='mean', axis=0),
		)
		self.data[vars] = transformers.fit_transform(self.data[[vars]])

	def crosstabsY(self, Y, X):
		for x in range(0, len(X)):
			ct = pandas.crosstab(self.data[Y], self.data[X[x]], normalize='columns')
			print(ct)
			print('')

	def showHistogram(self, var):
		plt.hist(self.data[var], bins='auto')
		plt.show()

	def prepareData(self, xVars, yVar):
		self.Y = self.data[yVar]
		# Make category variables binary
		self.X = pandas.get_dummies(self.data[xVars])

	def runAlgorithm(self, algorithm):
		# Remove features with low variance
		transformers = make_union(
			#VarianceThreshold(threshold=(.8 * (1 - .8))),
			PCA(),
		)
		# Make directory to store cache for transformers
		cachedir = mkdtemp()
		# Make pipeline
		pipe = make_pipeline(transformers, algorithm, memory=cachedir)
		# Split into training and validation datasets
		X_train, X_val, Y_train, Y_val = model_selection.train_test_split(self.X, self.Y, test_size=self.test_set_size, random_state=7)
		# Set up kfold
		kfold = model_selection.KFold(n_splits=self.k_fold_splits, random_state=7)
		# Run Algorithm
		scores = model_selection.cross_val_score(pipe, X_train, Y_train, cv=kfold, scoring='accuracy')
		print("%s: %f (%f)" % (pipe.steps[1][0], scores.mean(), scores.std()))
		# Clear Cache
		rmtree(cachedir)

	def runAlgorithms(self):
		self.runAlgorithm(LinearSVC())
		self.runAlgorithm(LogisticRegressionCV())
		self.runAlgorithm(KNeighborsClassifier())
		self.runAlgorithm(SVC())
		self.runAlgorithm(BernoulliNB())
		if len(self.X) > 100000:
			self.runAlgorithm(SGDClassifier())


		
