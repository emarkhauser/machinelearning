import pandas
import stats

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pandas.read_csv(url, names=names)

# Divide dataset into X and Y
X = data.values[:,0:4]
Y = data.values[:,4]

# Run modelling
modelling = stats.Modelling(X,Y)
modelling.run()



