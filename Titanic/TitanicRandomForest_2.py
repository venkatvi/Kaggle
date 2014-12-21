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


	# 2. map embarked - [1-C , 2-Q, 3-S]
	ds['EmbarkedOrd'] = ds['Embarked'].map(lambda x: ord(x) if x and type(x) is str else 0).astype(int)
	
	# 3. map title - [0 - Other, 1 - Master, 2 - Miss, 3 - Mr, 4 - Mrs]
	ds['Title'] = ds['Name'].map(lambda x: re.match(r'([^,]*)\,([^.]*).', x).group(2).upper().replace(".", "").strip())
	ds['Title'] = ds['Title'].map(lambda x: mapTitle(x) )

	# backup passenger id
	passengerIds = ds['PassengerId']

	# Drop PassengerId, Name, Sex, Embarked
	ds = ds.drop(['PassengerId', 'Name', 'Embarked', 'Cabin', 'Ticket'], axis=1)

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
	#print ds.info(verbose=True)
	return [ds, passengerIds]

if __name__ == '__main__':
	[train, trainPassengerIds] = cleanUpData('Data/train.csv', 'train')
	[test, testPassengerIds] = cleanUpData('Data/test.csv', 'test')
	print test.info(verbose=True)
	
	ageTest = train[train['Age'].notnull()==False]
	ageTest = ageTest.drop(['Age'], axis=1)
	
	ageTrain = train[np.isfinite(train['Age'])]
	
	#Predict Missing Values of Age:
	ageTarget = ageTrain.filter(['Age'])
	ageTarget = np.array(ageTarget.values).ravel()
	ageTrain = ageTrain.drop(['Age'], axis=1)
	
	gnb = GaussianNB()
	y_pred = gnb.fit(ageTrain, ageTarget).predict(ageTrain)
	print "Number of mislabeled age out of a total %d points : %d" % (ageTrain.shape[0],(ageTarget != y_pred).sum())
	y_test = gnb.predict(ageTest)

	ageTest['Age'] = y_test
	ageTrain['Age'] = ageTarget

	train = ageTrain.append(ageTest)

	
	#prediction_accuracy = accuracy_score(y_test, clf.best_estimator_.predict(X_test))
	#print "AgePredictionAccuracy:" + str(prediction_accuracy) 
	

	# # Fit the training data to the Survived labels and create the decision trees
	# target = train.filter(['Survived'])
	# target = np.array(target.values).ravel()
	# train = train.drop(['Survived'], axis=1)

	# X,X_test,y,y_test = train_test_split(train,target,
 #                                     test_size=.20,
 #                                     random_state=1899)

	
	# parameters = [{'n_estimators': np.arange(100,500,100)}]
	# clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, scoring='accuracy', verbose='3')
	# clf.fit(X,y) # running the grid search

	# prediction_accuracy = accuracy_score(y_test, clf.best_estimator_.predict(X_test))
	# #forest = forest.fit(train,target)

	#print train.shape
	#print train.columns

	#GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',cv=10) 

	#predictions = np.array(forest.predict(train))
	#rmse = math.sqrt(np.mean((np.array(target) - predictions)**2))
	#imp = sorted(zip(train.columns, forest.feature_importances_), key=lambda tup: tup[1], reverse=True)

	#print "RMSE: " + str(rmse)
	#print "10 Most Important Variables:" + str(imp[:10])

	# Take the same decision trees and run it on the test data
	#print test.shape
	#print test.columns
	#print np.isfinite(test.sum())
	#print np.isfinite(test).all()
	#print np.unique(test['Fare'].values)
	#output = forest.predict(test)
	#output = clf.best_estimator_.predict(test)
	#submissionData = {'PassengerId': testPassengerIds, 'Survived': output}
	#submissionDF = pd.DataFrame(submissionData)
	#submissionDF.drop_duplicates()

	#submissionDF.to_csv('Data/kaggleOutput2.csv', index=False)