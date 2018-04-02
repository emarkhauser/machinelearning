import machinelearning

ml = machinelearning.MachineLearning('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
ml.data.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# Cleaning and Splitting
ml.makeYX('class', ['sepal-length', 'sepal-width', 'petal-length', 'petal-width'])

# Run algorithms
ml.runAlgorithms()

