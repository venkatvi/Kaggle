import pandas as pd
import numpy as np
import math
import re
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
	
def mapTitle(titleStr):
	if titleStr == 'MASTER':
		return 1
	elif titleStr == 'MISS': 
		return 2
	elif titleStr == 'MR':
		return 3
	elif titleStr == 'MRS':
		return 4
	else:
		return 0
# Use  only age, sex, fare, cabin class
# dropna for age, cabin, and embarkment
def cleanUpData(fileName, dataType):
	ds = pd.read_csv(fileName)
	
	# clean ups
	# 1. map sex - to [0-female, 1-male]
	# Add a new col which is nominal for sex. 
	ds['Sex'] = ds['Sex'].map(lambda x: 0 if x is 'female' else 1)


	
	# 3. map title - [0 - Other, 1 - Master, 2 - Miss, 3 - Mr, 4 - Mrs]
	ds['Title'] = ds['Name'].map(lambda x: re.match(r'([^,]*)\,([^.]*).', x).group(2).upper().replace(".", "").strip())
	ds['Title'] = ds['Title'].map(lambda x: mapTitle(x) )

	# backup passenger id
	passengerIds = ds['PassengerId']

	# Drop PassengerId, Name, Sex, Embarked
	ds = ds.drop(['PassengerId', 'Name', 'Embarked', 'Cabin', 'Ticket', 'SibSp', 'Parch', 'Fare', 'Age'], axis=1)

	print ds.info(verbose=True)
	return [ds, passengerIds]

if __name__ == '__main__':
	[train, trainPassengerIds] = cleanUpData('Data/train.csv', 'train')
	[test, testPassengerIds] = cleanUpData('Data/test.csv', 'test')

	# Fit the training data to the Survived labels and create the decision trees
	target = train.filter(['Survived'])
	target = np.array(target.values).ravel()
	train = train.drop(['Survived'], axis=1)

	X,X_test,y,y_test = train_test_split(train,target,
                                     test_size=.20,
                                     random_state=1899)

	
	parameters = [{'n_estimators': np.arange(100,1000,100)}]
	clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, scoring='accuracy', verbose='3')
	clf.fit(X,y) # running the grid search

	prediction_accuracy = accuracy_score(y_test, clf.best_estimator_.predict(X_test))
	print prediction_accuracy
	
	#Take the same decision trees and run it on the test data
	output = clf.best_estimator_.predict(test)
	submissionData = {'PassengerId': testPassengerIds, 'Survived': output}
	submissionDF = pd.DataFrame(submissionData)
	submissionDF.to_csv('Data/Titanic_Gender_Class_Title_RF.csv', index=False)