import machinelearning

ml = machinelearning.MachineLearning('train.csv')
ml.cleanIntervalData('Age')
ml.prepareData(['Pclass','Sex','SibSp','Parch','Embarked','Age'], 'Survived')
ml.runAlgorithms()

