import pandas as pd
import re
import numpy as np

#load data
dsProgReports = pd.read_csv('Data/School_Progress_Reports_-_All_Schools_-_2009-10.csv')
dsDistrict = pd.read_csv('Data/School_District_Breakdowns.csv')
dsClassSize = pd.read_csv('Data/2009-10_Class_Size_-_School-level_Detail.csv')
dsAttendEnroll = pd.read_csv('Data/attendance.csv')
dsSATs = pd.read_csv('Data/SAT__College_Board__2010_School_Level_Results.csv')

pd.merge(dsSATs, dsClassSize, how='inner', left_on='DBN',  right_on='SCHOOL CODE')
pd.merge(dsProgReports, dsSATs, how='inner', on='DBN')
pd.merge(dsDistrict, dsProgReports, how='inner', right_on='DBN', left_on='JURISDICTION NAME')
pd.merge(dsAttendEnroll, dsProgReports, how='inner', right_on='DISTRICT', left_on='District')

# strip the first two characters of DBN from dsSATs and dsProgReports to match School Code of dsClassSize
dsProgReports.DBN = dsProgReports.DBN.map(lambda x: x[2:])
dsSATs.DBN = dsSATs.DBN.map(lambda x: x[2:])
pd.DataFrame(data=[dsProgReports['DBN'].take(range(5)), dsSATs['DBN'].take(range(5)), dsClassSize['SCHOOL CODE'].take(range(5))])

# format the district codes 
# get field value  of type "CSD 01 \n Mahattan", match it against regex [A-Za-z]*\s[0-9]*, get second group's value, convert it to int
dsDistrict['JURISDICTION NAME'] = dsDistrict['JURISDICTION NAME'].map(lambda x: re.match(r'([A-Za-z]*\s)([0-9]*)', x).group(2)).astype(int)

#get field of type "DISTRICT 01", get last two characters, convert to int

dsAttendEnroll.District = dsAttendEnroll.District.map(lambda x: x[-2:]).astype(int)

# match the keys using a DataFrame
pd.DataFrame(data=[dsProgReports['DISTRICT'][:3], dsDistrict['JURISDICTION NAME'][:3], dsAttendEnroll['District'][:3]])

# pandas can reindex data:
dsProgReports = dsProgReports.set_index('DBN')
dsDistrict = dsDistrict.set_index('JURISDICTION NAME')
dsClassSize = dsClassSize.set_index('SCHOOL CODE')
dsAttendEnroll = dsAttendEnroll.set_index('District')
dsSATs = dsSATs.set_index('DBN')

# create boolean vector mask ?
mask = dsSATs['Number of Test Takers'].map(lambda x: x != 's')
dsSATs = dsSATs[mask]

# cast field to integers
dsSATs['Number of Test Takers'] = dsSATs['Number of Test Takers'].astype(int)
dsSATs['Critical Reading Mean'] = dsSATs['Critical Reading Mean'].astype(int)
dsSATs['Mathematics Mean'] = dsSATs['Mathematics Mean'].astype(int)
dsSATs['Writing Mean'] = dsSATs['Writing Mean'].astype(int)

# get shape of the DataFrame
print dsClassSize.columns
print dsClassSize.take([0, 1, 10]).values

dsPupilTeacher = dsClassSize.filter(['SCHOOLWIDE PUPIL-TEACHER RATIO'])
mask = dsPupilTeacher['SCHOOLWIDE PUPIL-TEACHER RATIO'].map(lambda x: x>0)
dsPupilTeacher = dsPupilTeacher[mask]
dsClassSize = dsClassSize.drop('SCHOOLWIDE PUPIL-TEACHER RATIO', axix = 1)

dsClassSize = dsClassSize.drop(['BORO', 'CSD', 'SCHOOL NAME', 'GRADE ', 'PROGRAM TYPE',
	'CORE SUBJECT (MS CORE and 9-12 ONLY)',
	'CORE COURSE (MS CORE and 9-12 ONLY)', 
	'SERVICE CATEGORY(K-9* ONLY)',
	'NUMBER OF CLASSES',
	'DATA SOURCE'], axis=1)

grouped = dsClassSize.groupby(level=0)
dsClassSize = grouped.aggregate(np.max).\
join(grouped.aggregate(np.min), lsuffix=".max").\
join(grouped.aggregate(np.mean), lsuffix=".min", rsuffix=".mean").\
join(dsPupilTeacher)

mask = dsProgReports['SCHOOL LEVEL*'].map(lambda x: x == 'High School')
dsProgReports = dsProgReports[mask]

final = dsSATs.join(dsClassSize).\
join(dsProgReports).\
merge(dsDistrict, left_on='DISTRICT', right_index=True).\
merge(dsAttendEnroll, left_on='DISTRICT', right_index=True)


#Just drop string columns.
#In theory we could build features out of some of these, but it is impractical here
final = final.drop(['School Name','SCHOOL','PRINCIPAL','SCHOOL LEVEL*','PROGRESS REPORT TYPE'],axis=1)

#Remove % signs and convert to float
final['YTD % Attendance (Avg)'] = final['YTD % Attendance (Avg)'].map(lambda x: x.replace("%","")).astype(float)

#The last few columns we still have to deal with
final.dtypes[final.dtypes.map(lambda x: x=='object')]

gradeCols = ['2009-2010 OVERALL GRADE','2009-2010 ENVIRONMENT GRADE','2009-2010 PERFORMANCE GRADE','2009-2010 PROGRESS GRADE','2008-09 PROGRESS REPORT GRADE']

grades = np.unique(final[gradeCols].values) #[nan, A, B, C, D, F]

for c in gradeCols:
    for g in grades:
    	colName = c + "_is_" + str(g)
        final[colName] = pd.Series(data=final[c].map(lambda x: 1 if x is g else 0))


final = final.drop(gradeCols, axis=1) 

#Uncomment to generate csv files 
#final.drop(['Critical Reading Mean','Mathematics Mean','Writing Mean'],axis=1).to_csv('Data/train.csv') 
#final.filter(['Critical Reading Mean','Mathematics Mean','Writing Mean']).to_csv('Data/target.csv')