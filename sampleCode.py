'''
This is some sample code for some sample stuff
Written by Sarunas Girdenas, 2016
sgirdenas@gmail.com
'''

from __future__ import division

# first, load data files

from DataLoaderClass import DataLoader

loadData      = DataLoader()
loadedChurn   = loadData.LoadCSV('data.csv')
loadedDataSet = loadData.LoadCSV('pred.csv')

# index according to 'user_account_id'

loadedChurn.set_index('user_account_id',inplace=True)
loadedDataSet.set_index('user_account_id',inplace=True)

loadedDataSet['feature'] = loadedChurn.churn
loadedDataSet.reset_index(inplace=True)

print loadedDataSet.churn.value_counts()

# find duplicated rows

print 'Number of Duplicated Rows: ', sum(loadedDataSet.duplicated())

# get number of unique users

print 'Number of Unique Users: ', len(loadedDataSet.user_account_id.unique())

# make sure that features are integers (and positive integers too)

from dataAnalysisClass import dataAnalysisClass

dataAnalysis = dataAnalysisClass(loadedDataSet)

FeatureTypes = dataAnalysis.GetFeatureType()

for k in xrange(len(FeatureTypes.keys())):
	print FeatureTypes[FeatureTypes.keys()[k]]['Positive']
	print FeatureTypes[FeatureTypes.keys()[k]]['Type']

# now we know that all features are of reasonable data types and all are positive

# find nans and missing values (knowing that data types are compatible)

print dataAnalysis.findNansColumns()

# get histogram of the feature
dataAnalysis.histFeature('month')

##################### droped here

loadedDataSet = loadedDataSet.drop(loadedDataSet[loadedDataSet.user_no_outgoing_activity_in_days > 365].index)

########## case 1

month_8_data = loadedDataSet[loadedDataSet.month == 8]

month_8_data[month_8_data.user_lifetime > 365].feature.value_counts()

################### dropped data here!


### do clustering here

# now drop features that are not relevant for clustering


# list of most important features from Decision Tree                                       ################# drop Decision Tree Features
ForestFeatures = []
ForestFeatures.append('list of feature names')


month_8_CL_Raw = month_8_data[ForestFeatures]
dataAnalysis = dataAnalysisClass(month_8_CL_Raw)

# oversample the data set before clustering

categoricalFeatures = dataAnalysis.getCategoricalFeatures()
categoricalFeatures.append('year')
daysFeatures = []
daysFeatures.append('some feature')

# oversample the raw file
month_8_CL_Raw_Oversampled = dataAnalysis.overSampling('feature',categoricalFeatures,daysFeatures)

# split data into TRAINING and TESTING sets
month_8_CL_Raw_Oversampled_train, month_8_CL_Raw_Oversampled_test = dataAnalysis.splitDataSet(month_8_CL_Raw_Oversampled,0.8,'feautre')


# normalize all the features
dataAnalysis = dataAnalysisClass(month_8_CL_Raw_Oversampled)
month_8_CL_Raw_Normalized_Train = dataAnalysis.normalizeFeatures(month_8_CL_Raw_Oversampled_train,categoricalFeatures)
month_8_CL_Raw_Normalized_Test  = dataAnalysis.normalizeFeatures(month_8_CL_Raw_Oversampled_test,categoricalFeatures)



# now do the clustering using K-Means on TESTING DATA 											#### drop key features too!
from sklearn.cluster import KMeans
clusters = 5
import numpy as np
np.random.seed(1)
k_means = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
month_8_CL_Raw_Normalized_Train['cluster'] = k_means.fit_predict(month_8_CL_Raw_Normalized_Train.drop(['someFeature'],axis=1))
# do histogram of features
import matplotlib.pyplot as plt
plt.style.use('ggplot')
month_8_CL_Raw_Normalized_Train['cluster'].hist()
plt.xlabel('Clusters')
plt.ylabel('Number of Samples')
plt.show()

cluster_0_10_Feat = month_8_CL_Raw_Normalized_Train[month_8_CL_Raw_Normalized_Train.cluster == 0]
cluster_1_10_Feat = month_8_CL_Raw_Normalized_Train[month_8_CL_Raw_Normalized_Train.cluster == 1]
cluster_2_10_Feat = month_8_CL_Raw_Normalized_Train[month_8_CL_Raw_Normalized_Train.cluster == 2]
cluster_3_10_Feat = month_8_CL_Raw_Normalized_Train[month_8_CL_Raw_Normalized_Train.cluster == 3]
cluster_4_10_Feat = month_8_CL_Raw_Normalized_Train[month_8_CL_Raw_Normalized_Train.cluster == 4]

# now train logistic regression on each of the cluster


from sklearn import linear_model
logistic_0 = linear_model.LogisticRegression(C=1e5)
logistic_1 = linear_model.LogisticRegression(C=1e5)
logistic_2 = linear_model.LogisticRegression(C=1e5)
logistic_3 = linear_model.LogisticRegression(C=1e5)
logistic_4 = linear_model.LogisticRegression(C=1e5)

logistic_0.fit(cluster_0_10_Feat.drop(['feature'],axis=1).as_matrix(columns=None), cluster_0_10_Feat.feature.as_matrix(columns=None))
logistic_1.fit(cluster_1_10_Feat.drop(['feature'],axis=1).as_matrix(columns=None), cluster_1_10_Feat.feature.as_matrix(columns=None))
logistic_2.fit(cluster_2_10_Feat.drop(['feature'],axis=1).as_matrix(columns=None), cluster_2_10_Feat.feature.as_matrix(columns=None))
logistic_3.fit(cluster_3_10_Feat.drop(['feature'],axis=1).as_matrix(columns=None), cluster_3_10_Feat.feature.as_matrix(columns=None))
logistic_4.fit(cluster_4_10_Feat.drop(['feature'],axis=1).as_matrix(columns=None), cluster_4_10_Feat.feature.as_matrix(columns=None))


logistic.predict(month_8_CL_Raw_Oversampled_test.drop(['feature'],axis=1).as_matrix(columns=None)[1,:].reshape(1,-1))

# thing above works!

from sklearn.metrics import confusion_matrix
# note that labels here are just for two classes!
labels = ['0','1']
cm = confusion_matrix(testLabels_8_months_Array,logistic.predict(overSampled_8_mon_test_N_Array))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
ax.set_title('Logistic Regression Confusion Matrix')
plt.show()

# Logistic Regression Accuracy, ################################## 

counterC = 0
predictConfusion = []
for j in xrange(len(overSampled_8_mon_test_N_Array)):
	prediction = logistic.predict(overSampled_8_mon_test_N_Array[j,:].reshape(1,-1))
	if prediction == testLabels_8_months_Array[j,0]:
		counterC += 1
	predictConfusion.append(prediction)

float(counterC) / len(overSampled_8_mon_test_N_Array)


# now do the whole thing:

# 1. predict cluster
# 2. apply logistic regression trained on that cluster

AccPredfeature_0 = 0
AccPredfeature_1 = 0
AccPredfeature_2 = 0
AccPredfeature_3 = 0
AccPredfeature_4 = 0

# how many times each cluster was predicted
storeClusterPrediction_0 = 0
storeClusterPrediction_1 = 0
storeClusterPrediction_2 = 0
storeClusterPrediction_3 = 0
storeClusterPrediction_4 = 0


for f in xrange(len(month_8_CL_Raw_Oversampled_test)):

	predictedCluster = k_means.predict(month_8_CL_Raw_Oversampled_test.drop(['feature'],axis=1).as_matrix(columns=None)[f,:].reshape(1,-1))

	if predictedCluster == 0:
		storeClusterPrediction_0 += 1
		churnPrediction_0 = logistic_0.predict(month_8_CL_Raw_Oversampled_test.drop(['feature'],axis=1).as_matrix(columns=None)[f,:].reshape(1,-1))
		if churnPrediction_0 == month_8_CL_Raw_Oversampled_test.churn.as_matrix()[f]:
			AccPredChurn_0 += 1
	elif predictedCluster == 1:
		storeClusterPrediction_1 += 1
		churnPrediction_1 = logistic_1.predict(month_8_CL_Raw_Oversampled_test.drop(['feature'],axis=1).as_matrix(columns=None)[f,:].reshape(1,-1))
		if churnPrediction_1 == month_8_CL_Raw_Oversampled_test.churn.as_matrix()[f]:
			AccPredChurn_1 += 1
	elif predictedCluster == 2:
		storeClusterPrediction_2 += 1
		churnPrediction_2 = logistic_2.predict(month_8_CL_Raw_Oversampled_test.drop(['feature'],axis=1).as_matrix(columns=None)[f,:].reshape(1,-1))
		if churnPrediction_2 == month_8_CL_Raw_Oversampled_test.churn.as_matrix()[f]:
			AccPredChurn_2 += 1
	elif predictedCluster == 3:
		storeClusterPrediction_3 += 1
		churnPrediction_3 = logistic_3.predict(month_8_CL_Raw_Oversampled_test.drop(['feature'],axis=1).as_matrix(columns=None)[f,:].reshape(1,-1))
		if churnPrediction_3 == month_8_CL_Raw_Oversampled_test.churn.as_matrix()[f]:
			AccPredChurn_3 += 1
	elif predictedCluster == 4:
		storeClusterPrediction_4 += 1
		churnPrediction_4 = logistic_4.predict(month_8_CL_Raw_Oversampled_test.drop(['feature'],axis=1).as_matrix(columns=None)[f,:].reshape(1,-1))
		if churnPrediction_4 == month_8_CL_Raw_Oversampled_test.churn.as_matrix()[f]:
			AccPredChurn_4 += 1


# finally, get the accuracy of each logistic regression

AccCluster_0 = AccPredfeature_0 /  float(storeClusterPrediction_0)
AccCluster_1 = AccPredfeature_1 / float(storeClusterPrediction_1)
AccCluster_2 = AccPredfeature_2 / float(storeClusterPrediction_2)
AccCluster_3 = AccPredfeature_3 / float(storeClusterPrediction_3)
AccCluster_4 = AccPredfeature_4 / float(storeClusterPrediction_4)
























# convert to numpy arrays to feed to neural network ### TRAINING DATA
import numpy as np
overSampled_8_mon_train_N_Array = overSampled_8_mon_train_N.as_matrix(columns=None)
trainLabels_8_months_Array      = trainLabels_8_months.as_matrix(columns=None)
a = np.zeros([len(trainLabels_8_months_Array),1])
for j in xrange(len(a)):
	a[j,0] = trainLabels_8_months_Array[j]
trainLabels_8_months_Array = a

### TESTING DATA
overSampled_8_mon_test_N_Array = overSampled_8_mon_test_N.as_matrix(columns=None)
testLabels_8_months_Array      = testLabels_8_months.as_matrix(columns=None)
a = np.zeros([len(testLabels_8_months_Array),1])
for j in xrange(len(a)):
	a[j,0] = testLabels_8_months_Array[j]
testLabels_8_months_Array = a

### Neural Network

# from NeuralNetwork_2_Layers import NeuralNetwork_2_Layers
# Network_2 = NeuralNetwork_2_Layers()
# Network_2.doBoldDriver = True
# Network_2.initializeBoldDriver(1,0.5,0.03,10e-5)
# Network_2.initializeBoldDriver(2,0.25,0.03,10e-5)
# Network_2.initializeWeights_Layer1(overSampled_8_mon_train_N_Array[0:1000].T.shape[0]+1,overSampled_8_mon_train_N_Array[0:1000].T.shape[1]) # add 1 for bias
# Network_2.initializeWeights_Layer2(len(overSampled_8_mon_train_N_Array[0:1000]))

# storeErrorsFinal, storeErrors_1, inputLabels = Network_2.updateWeights(overSampled_8_mon_train_N_Array[0:200],trainLabels_8_months_Array[0:200],1000,0.2,0.03,0.01,200,True)

# import matplotlib.pyplot as plt
# plt.plot(storeErrorsFinal)
# plt.show()

# plt.plot(storeErrors_1)
# plt.show()

# # compute predictions

# Network_2.computePredictionNN(np.hstack((1,overSampled_8_mon_test_N_Array[0,:])))

# # test the accuracy

# storePredictions = []

# for j in xrange(len(overSampled_8_mon_test_N_Array)):
# 	storePredictions.append(Network_2.computePredictionNN(np.hstack((1,overSampled_8_mon_test_N_Array[j,:]))))

# # add intercept here!

# accuracy = Network_2.chooseLevel(0.5,testLabels_8_months_Array,np.hstack((np.ones([len(overSampled_8_mon_test_N_Array),1]),overSampled_8_mon_test_N_Array)))

print 'Doing Neural Network'

from NeuralNetwork_2_Layers_INIT import NeuralNetwork_2_Layers
Network_2 = NeuralNetwork_2_Layers()
Network_2.doBoldDriver = True
Network_2.initializeBoldDriver(1,0.25,0.03,10e-6)
Network_2.initializeBoldDriver(2,0.025,0.003,10e-6)
Network_2.initializeWeights_Layer1(overSampled_8_mon_train_N_Array[0:150].T.shape[0],overSampled_8_mon_train_N_Array[0:150].T.shape[1]) # add 1 for bias
Network_2.initializeWeights_Layer2(len(overSampled_8_mon_train_N_Array[0:150]))

storeErrorsFinal, storeErrors_1 = Network_2.updateWeights(overSampled_8_mon_train_N_Array,trainLabels_8_months_Array,1500,0.2,0.2,0.003,20000,True)

import matplotlib.pyplot as plt
plt.plot(storeErrorsFinal)
plt.show()

storePredictions = []
counterC = 0
level    = 0.5
predictConfusion = []
for j in xrange(len(overSampled_8_mon_test_N_Array)):
	storePredictions.append(Network_2.computePredictionNN(overSampled_8_mon_test_N_Array[j,:]))
	prediction = Network_2.computePredictionNN(overSampled_8_mon_test_N_Array[j,:])
	if prediction < level:
		predFinal = 0
	elif prediction >= level:
		predFinal = 1
	if predFinal == testLabels_8_months_Array[j,0]:
		counterC += 1
	predictConfusion.append(predFinal)

float(counterC) / j



from sklearn.metrics import roc_curve, auc
figure = plt.figure()
ax = figure.add_subplot(111)
fpr,tpr, _ = roc_curve(testLabels_8_months_Array,np.asarray(storePredictions))
roc_auc    = auc(fpr, tpr)
labelString = 'Curve Area %0.2f ' % roc_auc
ax.plot(fpr, tpr, label=labelString)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import confusion_matrix
# note that labels here are just for two classes!
labels = ['NOT Feature','Feature']
cm = confusion_matrix(testLabels_8_months_Array,np.asarray(predictConfusion))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.set_title('Neural Network Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# do the loop for 3D picture

choiceC  = np.arange(0,1,0.01)
classAcc = np.zeros([len(choiceC),1])
FalseNeg = np.zeros([len(choiceC),1])
for g in xrange(len(choiceC)):
	counterC = 0
	predictConfusion = []
	for j in xrange(len(overSampled_8_mon_test_N_Array)):
		prediction = Network_2.computePredictionNN(overSampled_8_mon_test_N_Array[j,:])
		if prediction < choiceC[g]:
			predFinal = 0
		elif prediction >= choiceC[g]:
			predFinal = 1
		if predFinal == testLabels_8_months_Array[j,0]:
			counterC += 1
		predictConfusion.append(predFinal)

	classAcc[g,0] = float(counterC) / len(overSampled_8_mon_test_N_Array)
	cm            = confusion_matrix(testLabels_8_months_Array,np.asarray(predictConfusion))
	FalseNeg[g,0] = float(cm[1][0]) / (cm[1][0]+cm[1][1])

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(choiceC, classAcc, FalseNeg,shade=True)
ax.set_xlabel('Decision Boundary')
ax.set_ylabel('Overall Accuracy')
ax.set_zlabel(' False Negative / All Negative')
ax.set_title('Neural Network Calibration')
plt.show()

# now do random forest ############# DECISION TREE STUFF

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(max_depth=6)
clf.fit(overSampled_8_mon_train_N_Array, trainLabels_8_months_Array)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=3)
# clf.fit(overSampled_8_mon_train_N_Array, trainLabels_8_months_Array)

from sklearn.metrics import confusion_matrix
# note that labels here are just for two classes!
labels = ['0','1']
cm = confusion_matrix(testLabels_8_months_Array,clf.predict(overSampled_8_mon_test_N_Array))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
ax.set_title('Random Forest Confusion Matrix')
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, auc
figure = plt.figure()
ax = figure.add_subplot(111)
fpr,tpr, _ = roc_curve(testLabels_8_months_Array,clf.predict(overSampled_8_mon_test_N_Array))
roc_auc    = auc(fpr, tpr)
labelString = 'Curve Area %0.2f ' % roc_auc
ax.plot(fpr, tpr, label=labelString)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC Curve')
plt.legend(loc="lower right")
plt.show()

# this piece recreates the decision tree classifier accuracy
RF_acc   = np.zeros([20,1])
classAcc = np.zeros([20,1])
FalseNeg = np.zeros([20,1])

for k in xrange(1,20):
	clf = DecisionTreeClassifier(max_depth=k)
	clf.fit(overSampled_8_mon_train_N_Array, trainLabels_8_months_Array)
	counterC = 0
	predictConfusion = []
	for j in xrange(len(overSampled_8_mon_test_N_Array)):
		prediction = clf.predict(overSampled_8_mon_test_N_Array[j,:].reshape(1,-1))
		if prediction == testLabels_8_months_Array[j,0]:
			counterC += 1
		predictConfusion.append(prediction)

	classAcc[k,0] = float(counterC) / len(overSampled_8_mon_test_N_Array)
	cm            = confusion_matrix(testLabels_8_months_Array,np.asarray(predictConfusion))
	FalseNeg[k,0] = float(cm[1][0]) / (cm[1][0]+cm[1][1])

# end of recreation

# check the feature importance using Random Forest

importances =  clf.feature_importances_
from numpy import argsort
indices = argsort(importances)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure()
titleString = 'Feature Importance, Decision Tree '
plt.title(titleString)
plt.barh(range(len(indices[-10:])),importances[indices][-10:],color='r')
plt.yticks(range(len(indices[-10:])),overSampled_8_mon_train_N.columns[indices][-10:])
plt.xlabel('Relative Importance')
plt.show()


#### LOGISTIC REGRESSION CLASSIFIER

from sklearn import linear_model
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(overSampled_8_mon_train_N_Array, trainLabels_8_months_Array)
logistic.predict(overSampled_8_mon_test_N_Array[1,:].reshape(1,-1))

from sklearn.metrics import confusion_matrix
# note that labels here are just for two classes!
labels = ['0','1']
cm = confusion_matrix(testLabels_8_months_Array,logistic.predict(overSampled_8_mon_test_N_Array))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
ax.set_title('Logistic Regression Confusion Matrix')
plt.show()

# Logistic Regression Accuracy

counterC = 0
predictConfusion = []
for j in xrange(len(overSampled_8_mon_test_N_Array)):
	prediction = logistic.predict(overSampled_8_mon_test_N_Array[j,:].reshape(1,-1))
	if prediction == testLabels_8_months_Array[j,0]:
		counterC += 1
	predictConfusion.append(prediction)

float(counterC) / len(overSampled_8_mon_test_N_Array)


