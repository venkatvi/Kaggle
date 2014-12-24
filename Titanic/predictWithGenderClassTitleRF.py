import pandas as pd
import numpy as np
import math
import re
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

	
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


	# 2. map embarked - [1-C , 2-Q, 3-S]
	#ds['EmbarkedOrd'] = ds['Embarked'].map(lambda x: ord(x) if x and type(x) is str else 0).astype(int)
	
	# 3. map title - [0 - Other, 1 - Master, 2 - Miss, 3 - Mr, 4 - Mrs]
	ds['Title'] = ds['Name'].map(lambda x: re.match(r'([^,]*)\,([^.]*).', x).group(2).upper().replace(".", "").strip())
	ds['Title'] = ds['Title'].map(lambda x: mapTitle(x) )

	# 2. map ticket - strip of any alphanumerics and retain number
	#ds['TicketNumber'] = ds['Ticket'].map(lambda x: re.match(r'([^0-9]*)([0-9]*)', x).group(2).upper().replace(".", "").strip())
	#ds['TicketNumber'] = ds['TicketNumber'].map(lambda x: 0 if not x else int(x))


	# backup passenger id
	passengerIds = ds['PassengerId']

	# Drop PassengerId, Name, Sex, Embarked
	ds = ds.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Cabin', 'Ticket', 'Fare', 'Age', 'Embarked'], axis=1)

	# clean ups 
	# only age, cabin and embarked are underfilled
	# when age not available, fill it with mean age
	#meanAge = ds['Age'].mean()
	#ds['Age'] = ds['Age'].map(lambda x: x if np.isfinite(x) else meanAge).astype(float)
	
	# when cabin not available, fill it with cabin of the previous  next ticket number if available, else fill with -1
	# already taken care of in cabinclass and cabinnumber
	# when emabrked not available, randomly choose one of C, Q, S
	# already taken care of in CQS map earlier

	# fill up missing value in Fare
	#meanFare = ds['Fare'].mean()
	#ds['Fare'] = ds['Fare'].map(lambda x: x if np.isfinite(x) else meanFare).astype(float)
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
                                     test_size=.30,
                                     random_state=42)

	
	parameters = [{'n_estimators': np.arange(1000,3000,1000), 'n_jobs': [9]}]
	clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, scoring='accuracy', verbose='3')
	clf.fit(X,y) # running the grid search

	prediction_accuracy = accuracy_score(y_test, clf.best_estimator_.predict(X_test))
	print prediction_accuracy
	
	
	#Take the same decision trees and run it on the test data
	output = clf.best_estimator_.predict(test)
	submissionData = {'PassengerId': testPassengerIds, 'Survived': output}
	submissionDF = pd.DataFrame(submissionData)
	submissionDF.to_csv('Data/Titanic_Gender_Class_Ttle_RF.csv', index=False)