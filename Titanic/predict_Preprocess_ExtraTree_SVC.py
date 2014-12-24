import pandas as pd
import numpy as np
import math
import re
from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

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
def computeCabinScore(x):
	return x['CabinClassOrd'] * 1000 + x['CabinNumber']
def cleanUpData(fileName):
	ds = pd.read_csv(fileName)
	#print train.info(verbose=True)

	# clean ups
	# 1. map sex - to [0-female, 1-male]
	# Add a new col which is nominal for sex. 
	ds['SexOrd'] = pd.Series(data=ds['Sex'].map(lambda x: 0 if x is 'female' else 1))

# 2. map ticket - strip of any alphanumerics and retain number
	ds['TicketNumber'] = ds['Ticket'].map(lambda x: re.match(r'([^0-9]*)([0-9]*)', x).group(2).upper().replace(".", "").strip())
	ds['TicketNumber'] = ds['TicketNumber'].map(lambda x: 0 if not x else int(x))


	ds['TicketPrefix'] = ds['Ticket'].map(lambda x: re.match(r'([^0-9]*)([0-9]*)', x).group(1).upper().replace(".", "").strip())
	ticketPrefixes = np.unique(ds['TicketPrefix'].values)
	#np.where returns an array of lists containing all indices which match x. Hence [0][0] 
	ds['TicketPrefixOrd'] = ds['TicketPrefix'].map(lambda x: np.where(ticketPrefixes == x)[0][0])

	# 3. cabin - remove any prefixes of C
	# cabin name is of type [ABCDEFGT]{1}\d*
	# read cabin string - split by space, for each cabin name, add another row with same info
	ds['CabinClass'] = ds['Cabin']
	ds['CabinNumber'] = ds['Cabin']
	row_iterator = ds.iterrows()
	last = row_iterator.next()
	for i, row in row_iterator:
		cabinObj = str(row['Cabin'])
		
		cabins = cabinObj.split(' ')
		j=0
		for cabin in cabins:
			j = j+1
			if cabin == 'nan' or len(cabin) == 0:
				cabinClass = 'N'
				cabinNumber = 0
			else:
				cabinClass = cabin[0]
				cabinNumber = 0
				if len(cabin) > 1:
					cabinNumber = int(cabin[1:])
			if len(cabins) > 1 and j > 1:
				# add a new row,
				new_row = row
				new_row['CabinClass'] = cabinClass
				new_row['CabinNumber'] = cabinNumber
				ds.loc[(ds.shape)[0]+1] = new_row
			else:
				ds.loc[i,'CabinClass'] = cabinClass
				ds.loc[i,'CabinNumber'] = cabinNumber
		last = row

	
	# for each cabinclass  now assign a number
	ds['CabinClassOrd'] = ds['CabinClass'].map(lambda x: ord(x) if x and type(x) is str else ord('N') ).astype(int)
	ds['CabinNumber'] = ds['CabinNumber'].map(lambda x: x if np.isfinite(x) else 0).astype(int)
	ds['CabinAgg'] = ds.apply(computeCabinScore, axis=1)

	# 3. map title - [0 - Other, 1 - Master, 2 - Miss, 3 - Mr, 4 - Mrs]
	ds['Title'] = ds['Name'].map(lambda x: re.match(r'([^,]*)\,([^.]*).', x).group(2).upper().replace(".", "").strip())
	ds['Title'] = ds['Title'].map(lambda x: mapTitle(x) )


	# 4. map embarked - [1-C , 2-Q, 3-S]
	ds['EmbarkedOrd'] = ds['Embarked'].map(lambda x: ord(x) if x and type(x) is str else 0).astype(int)

	# backup passenger id
	passengerIds = ds['PassengerId']

	# Drop PassengerId, Name, Sex, EmbarkedOrd
	ds = ds.drop(['PassengerId', 'Sex', 'Name', 'Embarked', 'Cabin', 'CabinClass', 'CabinNumber', 'CabinClassOrd', 'TicketPrefix', 'Ticket'], axis=1)
	
	# clean ups 
	# when age not available, fill it with mean age
	meanAge = ds['Age'].mean()
	ds['Age'] = ds['Age'].map(lambda x: x if np.isfinite(x) else meanAge).astype(float)
	
	# fill up missing value in Fare
	meanFare = ds['Fare'].mean()
	ds['Fare'] = ds['Fare'].map(lambda x: x if np.isfinite(x) else meanFare).astype(float)
	print ds.info(verbose=True)
	return [ds, passengerIds]

if __name__ == '__main__':
	[train, trainPassengerIds] = cleanUpData('Data/train.csv')
	[test, testPassengerIds] = cleanUpData('Data/test.csv')

	# Fit the training data to the Survived labels and create the decision trees
	target = train.filter(['Survived'])
	target = np.array(target.values).ravel()
	train = train.drop(['Survived'], axis=1)
	
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
	submissionData = {'PassengerId': testPassengerIds, 'Survived': result}
	submissionDF = pd.DataFrame(submissionData)
	submissionDF.to_csv('Data/Titanic_Preprocess_XtraTrees_SVC.csv', index=False)