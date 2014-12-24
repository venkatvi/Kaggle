import numpy as np
from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

workDir = r'C:\temp\Kaggle\\'

# Read data
train = np.genfromtxt(open(workDir + 'train.csv','rb'), delimiter=',')
target = np.genfromtxt(open(workDir + 'trainLabels.csv','rb'), delimiter=',')
test = np.genfromtxt(open(workDir + 'test.csv','rb'), delimiter=',')

# Scale data
train = pp.scale(train)
test = pp.scale(test)

# Select features
selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
train = selector.fit_transform(train, target)
test = selector.transform(test)

# Estimate score
classifier = SVC(C=8, gamma=0.17)
scores = cv.cross_val_score(classifier, train, target, cv=30)
print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

# Predict and save
result = classifier.fit(train, target).predict(test)
np.savetxt(workDir + 'result.csv', result, fmt='%d')