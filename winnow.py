"""
winnow algorithm for Python
this code is based on:

- users.soe.ucsc.edu/~manfred/pubs/J38.pdf
- cs.nyu.edu/~mohri/mlu_lecture_6.pdf


"""

from sklearn import datasets
import pandas as pd
from numpy import ones, zeros, exp, sign
import matplotlib.pyplot as plt
plt.style.use('ggplot')

irisData = datasets.load_iris()

irisDF = pd.DataFrame.from_dict({'sepalLength':irisData['data'][:,0],'sepalWidth':irisData['data'][:,1],'petalLength':irisData['data'][:,2],'petalWidth':irisData['data'][:,3],'label':irisData['target']})

# remove the third class what so ever

irisDF = irisDF[irisDF.label != 2]

# change labels to 1 and -1

irisDF.label = irisDF.label.map(lambda x: -1 if x == 0 else 1)

# do winnow algorithm

shuffledFrame = irisDF.sample(frac=1)
featureCols = irisDF.columns.tolist()
featureCols.remove('label')
featFrame = shuffledFrame[featureCols]
labels = shuffledFrame.label

# normalize featureCols

for col in featureCols:
	featFrame[col] = featFrame[col].map(lambda x: (x-featFrame[col].mean()) / featFrame[col].std())

w = ones([1,featFrame.shape[1]])
w = w/w.shape[1]
eta = 0.01
predicted = []
storeWeights = zeros([len(featFrame),featFrame.shape[1]])
absAccuracy = 0

for t in xrange(0,len(featFrame)):
	x = featFrame.iloc[t,:].values
	ypred = sign(w.dot(x))
	predicted.append(ypred)
	# update weights if needed
	if ypred != labels[t]:
		delta = ypred - labels[t]
		Z = (w*exp(eta*labels[t]*x)).sum()
		# update each weight
		for i in xrange(0,w.shape[1]):
			w[0,i] = w[0,i]*exp(-eta*delta*labels[t]*x[i])/Z
	else:
		w = w
		absAccuracy += 1
	storeWeights[t,:] = w

print 'Absolute Accuracy: ', absAccuracy / float(len(featFrame))

# plot some figures

plt.figure()
plt.plot(labels.values,'o',label='Actual')
plt.plot(predicted,'o',label='Predicted')
plt.xlabel('Number of Samples')
plt.ylabel('Predicted vs. Actual')
plt.legend(loc='upper left')
plt.show()

plt.figure()
plt.plot(storeWeights[:,0],label='weight -> Petal Length')
plt.plot(storeWeights[:,1],label='weight -> Petal Width')
plt.plot(storeWeights[:,2],label='weight -> Sepal Length')
plt.plot(storeWeights[:,3],label='weight -> Sepal Width')
plt.xlabel('Number of Samples')
plt.ylabel('Values of Weights')
plt.legend(loc='lower right')
plt.show()















