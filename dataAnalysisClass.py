'''
The purpose of this class is to gather all the functions that are used for analysing the data that they could be used conveniently
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

class dataAnalysisClass(object):

	def __init__(self,dataFrame):
		self.data = dataFrame
		return None

	def findNansColumns(self):
		'''
		Purpose: find missing values/Nans in the columns
		Input:   dataFrame
		Output:  list of column names that have Nans
		'''

		colNames = self.data.columns.tolist()

		nanCols = []

		for j in xrange(len(colNames)):
			if True in list(self.data[colNames[j]].isnull()):
				nanCols.append(colNames[j])

		if nanCols == []:
			return 'There are no Nans in the Columns!'

		return nanCols

	def histFeature(self,FeatureName):
		'''
		Purpose: do histogram of selected feature
		Input:   FeatureName - name of the feature
		Output:  histogram
		'''

		if type(FeatureName) != str:
			raise Exception('Feature Name must be String!')


		listOfDataFeatures = self.data.columns.tolist()

		if FeatureName not in listOfDataFeatures:
			raise Exception('Feature Not in Data Frame!')

		import matplotlib.pyplot as plt
		plt.style.use('ggplot')

		plt.figure()
		self.data[FeatureName].hist()
		plt.xlabel('Values')
		plt.ylabel('Occurence')
		plt.title(FeatureName)
		plt.show()

		return None

	def GetFeatureType(self):
		'''
		Purpose: determine feature type and if it is positive
		Input:   dataFrame
		Output:  list of FeatureNames and types
		'''

		listOfDataFeatures = self.data.columns.tolist()

		FeatureType = {}

		for j in xrange(len(listOfDataFeatures)):
			FeatureType[listOfDataFeatures[j]] = {}
			FeatureType[listOfDataFeatures[j]]['Type'] = str(self.data[listOfDataFeatures[j]].dtype)
			if FeatureType[listOfDataFeatures[j]]['Type'] != 'str':
				if self.data[listOfDataFeatures[j]].min() >= 0:
					FeatureType[listOfDataFeatures[j]]['Positive'] = True
				else:
					FeatureType[listOfDataFeatures[j]]['Positive'] = False

		return FeatureType

	def calculateCorrelationFeatures(self,corLevel):
		'''
		Purpose: calculate correlation (Pearson Coefficient)
		Input:   data file with features
				 corLevel - correlation level 
		Output:  dictionary with feature names as keys and lists as values where list
			     contains names of features where Pearson Coefficient > corLevel
		'''

		if type(corLevel) != float:
			raise Exception('Correlation Level must be Float!')

		if (corLevel > 1) or (corLevel < -1):
			raise Exception('Correlation Level must be between -1.0 and 1.0!')

		corrMatrix = self.data.corr()

		listOfDataFeatures = self.data.columns.tolist()

		dictOut = {}

		for j in xrange(len(listOfDataFeatures)):

			FeatureName = listOfDataFeatures[j]

			if (FeatureName in listOfDataFeatures) and (corLevel > 0):
				dictOut[FeatureName] = corrMatrix[FeatureName][corrMatrix[FeatureName] > corLevel].index.tolist()
				if FeatureName in dictOut[FeatureName]:
					dictOut[FeatureName].remove(FeatureName)
			elif (FeatureName in listOfDataFeatures) and (corLevel < 0):
				dictOut[FeatureName] = corrMatrix[FeatureName][corrMatrix[FeatureName] < corLevel].index.tolist()

		return dictOut

	def getCategoricalFeatures(self):
		'''
		Purpose: get the number of categorical features in the data set
		Input:   dataset
		Output:  list of feature names that are categorical (have only two values which are 0 and 1)
		NOTE:  this function will not work with variables such as years
		'''

		listOfDataFeatures = self.data.columns.tolist()

		catFeatures = []

		for k in xrange(len(listOfDataFeatures)):
			if (len(dict(self.data[listOfDataFeatures[k]].value_counts()).keys()) == 2) and ((0 in dict(self.data[listOfDataFeatures[k]].value_counts()).keys()) or (1 in dict(self.data[listOfDataFeatures[k]].value_counts()).keys())):
				catFeatures.append(listOfDataFeatures[k])

		return catFeatures

	def countUnique(self,ColumnName):
		'''
		Purpose: count unique occurences in the column
		Input:   ColumnName - name of column we want to count
		Output:  dictionary with values and occurences
		Note: should be used with str, but also works with floats
		'''

		if type(ColumnName) != str:
			raise Exception('Column Name must be String!')

		listOfDataFeatures = self.data.columns.tolist()

		if ColumnName not in listOfDataFeatures:
			raise Exception('Given Column Name is not in The Data Frame!')

		bigList = self.data[ColumnName].value_counts()

		keysOut = bigList.keys()
		valsOut = bigList.values

		dictOut = {}

		for j in xrange(len(keysOut)):
			dictOut[keysOut[j]] = valsOut[j]

		return dictOut

	def overSampling(self,ColumnName,listCatFeatures,daysFeatures):
		'''
		Purpose: do oversampling to balance the dataset
		Input:   dataFrame        - data frame that we have from class instance
		         ColumnName       - column according to which we want to resample
		         listCatFeatures  - list of Features that are categorical so we dont touch them!
		         daysFeatures     - list of Features that has integers in them, so we add random integer to them
		Output:  dataFrameOut - oversampled dataFrame
		NOTE:    1. We assume that there are only two values in the resampling column
		         2. We assume that user wants to distort oversampled features, we could slightly change the function to make this choice optional
		'''

		if type(listCatFeatures) != list:
			raise Exception('List of Categorical Features type must be List!')

		if type(ColumnName) != str:
			raise Exception('Column Name must be String!')

		listOfDataFeatures = self.data.columns.tolist()

		if ColumnName not in listOfDataFeatures:
			raise Exception('Given Column Name is not in The Data Frame!')

		countRelevance = self.countUnique(ColumnName)

		# choose how many instances to resample of the under represented class

		if countRelevance[0] > countRelevance[1]:
			numResample   = countRelevance[0] / countRelevance[1]
			classResample = countRelevance.keys()[1]

		elif countRelevance[0] < countRelevance[1]:
			numResample   = countRelevance[1] / countRelevance[0]
			classResample = countRelevance.keys()[0]

		dataFrameOut = self.data

		from numpy.random import random, randint
		from numpy.random import seed
		seed(0)

		for j in xrange(0,numResample):
			newFrame = self.data.loc[self.data[ColumnName] == classResample]

			# add small random number to each oversampled instance
			# THIS LOOP IS a BOTTLE NECK OF THE FUNCTION!
			newFrameIdx = newFrame.columns.tolist()
			for k in xrange(len(newFrameIdx)):
				if newFrameIdx[k] not in listCatFeatures:
					newFrame[newFrameIdx[k]] + 0.0001*random(1)
				elif newFrameIdx[k] in daysFeatures:
					newFrame[newFrameIdx[k]] + randint(-10,10)
			dataFrameOut = dataFrameOut.append(newFrame)

		# set new index to make sure that it is unique
		newIndex = range(len(dataFrameOut))
		dataFrameOut.index = newIndex

		return dataFrameOut

	@staticmethod
	def splitDataSet(dataFile,percentOne,ColumnName):
		'''
		Purpose: split dataset into 2 parts as defined by percentTwo and percentOne
		Input:   percentOne   - returns this percentage of dataset with the same proportion
				 ColumnName   - column by which we want to split the dataset
		Output:  dataFrameOut - dataFrame which has percentOne # of obs with the same proportion of colName as the original dataSet
		NOTE: we assume that ColumnName has just two values!
			  this function is applied after oversampling so we know that propotion is 50/50 of each class/type!
		'''

		import pandas

		if type(dataFile) != pandas.core.frame.DataFrame:
			raise PandasDataFrameTypeError('Given Variable is not Pandas Data Frame!')

		if type(percentOne) != float:
			raise Exception('Given Percentage must be Float!')

		if (percentOne > 1) or (percentOne < 0):
			raise Exception('Given Percentage must be between 0 and 1!')

		listOfDataFeatures = dataFile.columns.tolist()

		if ColumnName not in listOfDataFeatures:
			raise Exception('Given Column Name is not in The Data Frame!')

		# create new index

		dataSetIndex   = range(len(dataFile))
		dataFile.index = dataSetIndex

		#dataSetIndex = dataFile.index.tolist()

		# get the indices of each value of class (assuming there are two values only)

		value1 = dataFile[ColumnName].value_counts().keys()[0]
		value2 = dataFile[ColumnName].value_counts().keys()[1]

		idx1 = dataFile.loc[dataFile[ColumnName] == value1].index.tolist()
		idx2 = dataFile.loc[dataFile[ColumnName] == value2].index.tolist()

		lengthOut = int(len(dataFile)*percentOne)

		from random import sample
		from numpy.random import seed
		seed(0)

		idx1Out     = sample(idx1,int(lengthOut/2))
		idx2Out     = sample(idx2,int(lengthOut/2))
		commonIndex = idx1Out + idx2Out

		oneFrameOut = dataFile.iloc[commonIndex,:]

		secondIndex = []

		# get the other index
		# THIS LOOP SLOWS DOWN THE FUNCTION, CHECK IT!

		for j in xrange(len(dataSetIndex)):
			if dataSetIndex[j] not in commonIndex:
				secondIndex.append(dataSetIndex[j])

		twoFrameOut = dataFile.iloc[secondIndex,:]

		return oneFrameOut, twoFrameOut

	@staticmethod
	def normalizeFeatures(dataFile,keepFeatures):
		'''
		Purpose: normalize features by subtracting mean and dividing by std
		Input:   dataFile     - pandas dataFrame
				 keepFeatures - features that should not be normalized (such as categorical features)
		Output:  data file with normalized features
		'''

		import pandas

		if type(dataFile) != pandas.core.frame.DataFrame:
			raise PandasDataFrameTypeError('Given Variable is not Pandas Data Frame!')

		if type(keepFeatures) != list:
			raise Exception('Keep Features Type must be List!')

		listOfDataFeatures = dataFile.columns.tolist()

		for j in xrange(len(listOfDataFeatures)):
			if listOfDataFeatures[j] not in keepFeatures:
				dataFile.loc[:,listOfDataFeatures[j]] = (dataFile.loc[:,listOfDataFeatures[j]] - dataFile.loc[:,listOfDataFeatures[j]].mean()) / dataFile.loc[:,listOfDataFeatures[j]].std()

		return dataFile




