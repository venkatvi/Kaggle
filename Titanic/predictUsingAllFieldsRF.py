import pandas as pd
import numpy as np
import math
import re
from sklearn.ensemble import RandomForestClassifier 

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
				cabinClass = ''
				cabinNumber = -1
			else:
				cabinClass = cabin[0]
				cabinNumber = -1
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
	ds['CabinClassOrd'] = ds['CabinClass'].map(lambda x: ord(x) if x and type(x) is str else 0 ).astype(int)
	ds['CabinNumber'] = ds['CabinNumber'].map(lambda x: x if np.isfinite(x) else -1).astype(int)

	# 4. map embarked - [1-C , 2-Q, 3-S]
	ds['EmbarkedOrd'] = ds['Embarked'].map(lambda x: ord(x) if x and type(x) is str else 0).astype(int)

	# backup passenger id
	passengerIds = ds['PassengerId']

	# Drop PassengerId, Name, Sex, Embarked
	ds = ds.drop(['PassengerId', 'Sex', 'Name', 'Embarked', 'Cabin', 'CabinClass', 'Ticket', 'TicketPrefix'], axis=1)
	# clean ups 
	# only age, cabin and embarked are underfilled
	# when age not available, fill it with mean age
	meanAge = ds['Age'].median()
	ds['Age'] = ds['Age'].map(lambda x: x if np.isfinite(x) else meanAge).astype(float)
	
	# when cabin not available, fill it with cabin of the previous  next ticket number if available, else fill with -1
	# already taken care of in cabinclass and cabinnumber
	# when emabrked not available, randomly choose one of C, Q, S
	# already taken care of in CQS map earlier

	# fill up missing value in Fare
	meanFare = ds['Fare'].median()
	ds['Fare'] = ds['Fare'].map(lambda x: x if np.isfinite(x) else meanFare).astype(float)
	print ds.info(verbose=True)
	return [ds, passengerIds]

if __name__ == '__main__':
	[train, trainPassengerIds] = cleanUpData('Data/train.csv')
	[test, testPassengerIds] = cleanUpData('Data/test.csv')

	
	#print test.info

	# Create the random forest object which will include all the parameters
	# for the fit
	forest = RandomForestClassifier(n_estimators = 100)

	# Fit the training data to the Survived labels and create the decision trees
	target = train.filter(['Survived'])
	target = np.array(target.values).ravel()
	train = train.drop(['Survived'], axis=1)
	forest = forest.fit(train,target)

	#print train.shape
	#print train.columns
	predictions = np.array(forest.predict(train))
	rmse = math.sqrt(np.mean((np.array(target) - predictions)**2))
	imp = sorted(zip(train.columns, forest.feature_importances_), key=lambda tup: tup[1], reverse=True)

	print "RMSE: " + str(rmse)
	print "10 Most Important Variables:" + str(imp[:10])

	# Take the same decision trees and run it on the test data
	#print test.shape
	#print test.columns
	#print np.isfinite(test.sum())
	#print np.isfinite(test).all()
	#print np.unique(test['Fare'].values)
	output = forest.predict(test)

	submissionData = {'PassengerId': testPassengerIds, 'Survived': output}
	submissionDF = pd.DataFrame(submissionData)
	submissionDF.drop_duplicates()

	submissionDF.to_csv('Data/kaggleOutput.csv', index=False)