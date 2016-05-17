'''
The purpose of this class is to gather all the functions that are used for clearning and loading data into pandas frame
Written by Sarunas Girdenas, 2016, spring, sgirdenas@gmail.com
'''

class PandasDataFrameTypeError(Exception):
	'''
	Purpose: Raise Exception when given input is not pandas data frame
	'''
	def __init__(self,msg):
		self.msg = msg
		Exception.__init__(self,msg, '')

	def __str__(self):
		return self.msg

class DataLoader(object):

	def __init__(self):
		return None

	@classmethod
	def LoadCSV(self,fileName):
		'''
		Purpose: load given data to pandas data frame
		Input: fileName  - data file name
			   sheetName - CSV file sheetName
		Output: pandas data frame
		'''

		from os import listdir, getcwd

		if fileName not in listdir(getcwd()):
			raise Exception('File is not in the Current Directory!')

		from pandas import read_csv

		LoadedData = read_csv(fileName)

		return LoadedData

	@staticmethod
	def renameColumns(DataFrameName,relVar,queryVar,DocVar,FeatVar):
		'''
		Purpose: rename columns of data frame to something user friendly
		Input: DataFrameName: name of data frame
			   relVar:        name that user wants to give for document relevance
			   queryVar: 	  name that user wants to give for query ID
			   DocVar:   	  name that user wants to give for document ID
			   FeatVar:  	  generic name that user wants to give to features, number of feature is appended at the end

		Output: DataFrameName - data frame with renamed columns	   
		'''

		if (type(relVar) != str) or (type(queryVar) != str) or (type(DocVar) != str) or (type(FeatVar) != str):
			raise Exception('All Naming Variables must be Strings!')

		import pandas

		if type(DataFrameName) != pandas.core.frame.DataFrame:
			raise PandasDataFrameTypeError('Given File is not Pandas Data Frame!')

		for j in xrange(len(DataFrameName.T)):

			if DataFrameName[j].dtype == 'int64':
				DataFrameName = DataFrameName.rename(columns={j:relVar})
			elif (':' in DataFrameName[j][0]) and ('qid' == DataFrameName[j][0][0:DataFrameName[j][0].index(':')]):
				DataFrameName = DataFrameName.rename(columns={j:queryVar})
			elif j == len(DataFrameName.T)-1:
				DataFrameName = DataFrameName.rename(columns={j:DocVar})
			elif ':' in DataFrameName[j][0]:
				NewTitle = FeatVar + DataFrameName[j][0][0:DataFrameName[j][0].index(':')]
				DataFrameName = DataFrameName.rename(columns={j:NewTitle})
			else:
				DataFrameName = DataFrameName.rename(columns={j:j})

		return DataFrameName

	@staticmethod
	def convertToFloat(DataFrameName,FeatVar):
		'''
		Purpose: convert features from string to float
		Input:   DataFrameName - pandas DataFrame
				 FeatVar       - generic name of features
		Output:  DataFrameName - pandas DataFrame with converted features to strings
		'''

		if type(FeatVar) != str:
			raise Exception('Generic Feature Name must be String!')

		import pandas

		if type(DataFrameName) != pandas.core.frame.DataFrame:
			raise PandasDataFrameTypeError('Given File is not Pandas Data Frame!')

		for j in xrange(len(DataFrameName.T)):

			if (type(DataFrameName.columns[j]) == str) and (DataFrameName.columns[j][0:len(FeatVar)] == FeatVar) and (DataFrameName[DataFrameName.columns[j]].dtype != float):
				DataFrameName[DataFrameName.columns[j]] = DataFrameName[DataFrameName.columns[j]].map(lambda x: float(x[x.index(':')+1:len(x)]), DataFrameName[DataFrameName.columns[j]].all() )
		
		return DataFrameName

	@staticmethod
	def convertQueryID(DataFrameName,queryVar,symbolName):
		'''
		Purpose: remove 'qid:' before queryID value
		Input:   DataFrameName - pandas DataFrame
				 queryVar      - name of query ID
				 symbolName    - symbol before which we want to remove stuff
		Output:  DataFrameName - pandas DataFrame with removed 'qid:'
		'''

		if type(queryVar) != str:
			raise Exception('Query ID Variable must be String!')

		listOfDataFeatures = DataFrameName.columns.tolist()

		if queryVar not in listOfDataFeatures:
			raise Exception('Query ID Name not in Data Frame!')

		if type(symbolName) != str:
			raise Exception('Symbol must be String Type!')

		import pandas

		if type(DataFrameName) != pandas.core.frame.DataFrame:
			raise PandasDataFrameTypeError('Given File is not Pandas Data Frame!')

		DataFrameName[queryVar] = DataFrameName[queryVar].map(lambda x: str(x[x.index(symbolName)+1:len(x)]), DataFrameName[queryVar].all() )

		return DataFrameName

	@staticmethod
	def removeColumns(DataFrameName):
		'''
		Purpose: remove columns that are redundant
		Input:   DataFrameName - pandas dataframe
		Output:  the same data frame but with reduced number of cols
		'''

		import pandas

		if type(DataFrameName) != pandas.core.frame.DataFrame:
			raise PandasDataFrameTypeError('Given File is not Pandas Data Frame!')

		listNum = []

		for h in xrange(len(DataFrameName.columns)):
			if type(DataFrameName.columns[h]) == int:
				listNum.append(h)

		for k in xrange(len(listNum)):
			del DataFrameName[listNum[k]]

		return DataFrameName



