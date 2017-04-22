import pandas as pd 
import glob
import numpy as np 
import pickle
import re
from itertools import permutations
from itertools import product
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn import model_selection
from sklearn import metrics
#from matplotlib import pyplot as plt 
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import statsmodels.api as sm
import sys
import pickle
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV


#This function takes the list of chomosomes from where we are to take the negative and positive samples 
#stored in different dataFrames and then appends them together and returns a full big data frame
#these dataframes which initially have frequencies in them are converted to tf-idf values
def makeDataFrameTFIDF(listOfChromoToUse):
	listOfDataFrames = [] #keeps a list of  both positive and negative frames
	#for every chromosome take the positive dataframe, take the negative dataframe,and append them together
	for chromoName in listOfChromoToUse:
		positiveDataFrame = pd.read_csv(('DataFramesSeqClassification/6/Positive/' + chromoName + '.csv'))
		negativeDataFrame = pd.read_csv(('DataFramesSeqClassification/6/Negative/' + chromoName + '.csv'))
		listOfDataFrames.append(positiveDataFrame)
		listOfDataFrames.append(negativeDataFrame)

	#once we have the list of dataFrames we have to concatenate them to one dataFrame
	fullDataFrame = pd.concat(listOfDataFrames)
	print(fullDataFrame.shape)
	fullDataMatrix = fullDataFrame.as_matrix()#convert the data frame to matrix as it is easier to perform operations
	print(np.shape(fullDataMatrix))
	#convert frequencies to probabilities ie we get the Term-Frequency
	columnCount = np.size(fullDataMatrix,axis=1)
	fullDataMatrix[:,0:columnCount-1] = fullDataMatrix[:,0:columnCount-1] / fullDataMatrix[:,0:columnCount-1].sum(axis=1,keepdims=True)#divide every row element by sum of row
	#calculate inverse document frequency(IDF) = loge(totaldocs/#docs with term T)
	totalNumberOfSeq = np.size(fullDataMatrix,axis=0)
	for columnNumber in range(columnCount - 1):
		numberOfSeqWithKMer = np.count_nonzero(fullDataMatrix[:,columnNumber])
		fullDataMatrix[:,columnNumber] *= np.log(totalNumberOfSeq/numberOfSeqWithKMer)
	#print(totalNumberOfSeq,'  ',numberOfSeqWithKMer,' ',np.log(totalNumberOfSeq/numberOfSeqWithKMer))
	return pd.DataFrame(fullDataMatrix,columns=fullDataFrame.columns)# return the full big dataframe
	#for row in range(totalNumberOfSeq):
	#	print(fullDataMatrix[row,1])

	#for row in range(np.size(fullDataMatrix,axis=0)):
	#	print(np.sum(fullDataMatrix[row,:]))
	
	#positiveDataFrame

#--------------------------------END of makeDataFrameTFIDF---------------------------------------


# load the dataframe that only contain raw frequencies
def makeDataFrameFrequency(listOfChromoToUse):
	listOfDataFrames = [] #keeps a list of  both positive and negative frames
	#for every chromosome take the positive dataframe, take the negative dataframe,and append them together
	for chromoName in listOfChromoToUse:
		positiveDataFrame = pd.read_csv(('DataFramesSeqClassification/6/Positive/' + chromoName + '.csv'))
		negativeDataFrame = pd.read_csv(('DataFramesSeqClassification/6/Negative/' + chromoName + '.csv'))
		listOfDataFrames.append(positiveDataFrame)
		listOfDataFrames.append(negativeDataFrame)

	#once we have the list of dataFrames we have to concatenate them to one dataFrame
	fullDataFrame = pd.concat(listOfDataFrames)
	print(fullDataFrame.shape)
	return fullDataFrame




def runNaiveBayes(XTrain,XTest,yTrain,yTest,fullX,fullY):
	
	clf = GaussianNB()
	kf = KFold(n_splits=10)
	fullX = fullX.as_matrix()
	fullY = fullY.as_matrix()
	
	#	Xtrain,XTest,yTrain,yTest = fullX[train],fullX[test],fullY[train].ravel(),fullY[test].ravel()
	trainAccuracyList = []
	testAccuracyList = []
	sensitivityList = []
	specificityList = []
	AUCList = []

	kf = KFold(n_splits=5)
	for trainIndex,testIndex in kf.split(fullX):
		XTrain,XTest = fullX[trainIndex],fullX[testIndex]
		yTrain,yTest = fullY[trainIndex],fullY[testIndex]
		#print(np.shape(XTrain),'  ',np.shape(yTrain))

		clf.fit(XTrain, yTrain.ravel())

		trainAccuracyList.append(clf.score(XTrain,yTrain))
		testAccuracyList.append(clf.score(XTest,yTest))

		#print('Training Accuracy',clf.score(XTrain,yTrain),'','Test Accuracy',clf.score(XTest,yTest))
		
		pred = clf.predict(XTest)
		pr = precision_recall_fscore_support(yTest,pred)
	
	#print(pr.precision,'',pr.recall,'',pr.fbeta_score)
		proba = (clf.predict_proba(XTest))
		lw = 2
		#fpr,tpr,thresholds = metrics.roc_curve(yTrain, clf.predict_proba(XTrain)[:,1], pos_label=1)


		#print('Sensitivity ',pr[1][1])   #recall
		conf = confusion_matrix(yTest,pred)
		
		sensitivityList.append(pr[1][1])
		specificityList.append(conf[0,0]/(conf[0,0] + conf[0,1]))
		#AUCList.append(roc_auc_score(yTest,proba[:,1]))
		#print('Specificity',conf[0,0]/(conf[0,0] + conf[0,1]),'  sensitivity ',conf[1,1]/(conf[1,0] + conf[1,1]),'AUC-',roc_auc_score(yTest,proba[:,1]))

		#print(conf[0,0],'',conf[0,1],'\n',conf[1,0],'',conf[1,1])

	print('Training Accuarcy - ',sum(trainAccuracyList)/len(trainAccuracyList))
	print('Test Accuracy - ',sum(testAccuracyList) / len(testAccuracyList))
	print('Sensitivity - ',sum(sensitivityList)/len(sensitivityList))
	print('Specificity - ',sum(specificityList)/len(specificityList))
	#print('AUC - ',sum(AUCList)/len(AUCList))

#----------------------------------------------------------------------------

def runLinearSVM(XTrain,XTest,yTrain,yTest,fullX,fullY,C=1,gamma=1):
	clf = svm.LinearSVC(C=C)
	
	kf = KFold(n_splits=5)
	
	trainingAccuracyList = []
	CVAccuracyList = []
	CVSensitivityList = []
	CVSpecificityList = []
	XTrainFolds = []
	yTrainFolds = []
	XCVFold = []
	yCVFold = []
	XTrain = XTrain.values
	yTrain = yTrain.values
	for trainIndex,CVIndex in kf.split(XTrain):
		XTrainFolds,XCVFold = XTrain[trainIndex],XTrain[CVIndex]
		yTrainFolds,yCVFold = yTrain[trainIndex],yTrain[CVIndex]
		#XCVFold = XTrain[CVIndex]
		#yCVFold = yTrain[CVIndex]
		clf.fit(XTrainFolds,yTrainFolds.ravel())
		trainingAccuracyList.append(clf.score(XTrainFolds,yTrainFolds))
		CVAccuracyList.append(clf.score(XCVFold,yCVFold))
		pred = clf.predict(XCVFold)
		pr = precision_recall_fscore_support(yCVFold,pred)
		CVSensitivityList.append(pr[1][1])
		conf = confusion_matrix(yCVFold,pred)
		CVSpecificityList.append(conf[0,0]/(conf[0,0] + conf[0,1]))

	print('Training Accuracy - ',sum(trainingAccuracyList)/len(trainingAccuracyList))
	print('CV Accuracy - ',sum(CVAccuracyList)/len(CVAccuracyList))
	print('CV Sensitivity - ',sum(CVSensitivityList)/len(CVSensitivityList))
	print('CV Specificity - ',sum(CVSpecificityList)/len(CVSpecificityList))
		#print('Training Accuracy',clf.score(XTrain,yTrain),'','Test Accuracy',clf.score(XTest,yTest))
	
	clf.fit(XTrain,yTrain.ravel())
	
	pred = clf.predict(XTest)
	print('Test Accuracy - ',clf.score(XTest,yTest))
	pr = precision_recall_fscore_support(yTest,pred)
	print('Test Sensitivity ',pr[1][1])
	conf = confusion_matrix(yTest,pred)
	print('Test Specificity ',conf[0,0]/(conf[0,0] + conf[0,1]))
	
	#pr = precision_recall_fscore_support(yTest,pred)
	
	#print(pr.precision,'',pr.recall,'',pr.fbeta_score)
	#proba = (clf.predict_proba(XTest))
	#lw = 2
	#fpr,tpr,thresholds = metrics.roc_curve(yTrain, clf.predict_proba(XTrain)[:,1], pos_label=1)


	#print('Sensitivity ',pr[1][1])   #recall
	#conf = confusion_matrix(yTest,pred)
	#print('Specificity',conf[0,0]/(conf[0,0] + conf[0,1]))#,'  sensitivity ',conf[1,1]/(conf[1,0] + conf[1,1]),'AUC-',roc_auc_score(yTest,proba[:,1]))

	#print(conf[0,0],'',conf[0,1],'\n',conf[1,0],'',conf[1,1])

	parameter_candidates = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}]


#	clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
	# Train the classifier on data1's feature and target data
#	clf.fit(XTrain,yTrain)
	# View the accuracy score
#	print('Best score for data1:', clf.best_score_)
#	print('Best C:',clf.best_estimator_.C)
#	print('Best Kernel:',clf.best_estimator_.kernel)

#	print('The grid search results ',clf.cv_results_)
#	print('Best Gamma:',clf.best_estimator_.gamma)
#----------------------------------------------------------------------------




#---------------------------------------------------------------------------

def runSVMGaussian(XTrain,XTest,yTrain,yTest,XCV,yCV,fullX,fullY,C=1,gamma=1):
	clf = svm.SVC(C=C,gamma=gamma)
	clf.fit(XTrain,yTrain.ravel())
	
	trainAccuracy = clf.score(XTrain,yTrain)
	CVAccuracy = clf.score(XCV,yCV)
	testAccuracy = clf.score(XTest,yTest)

	print('TestTrainAccuracy ',trainAccuracy)
	print('CVAccuracy ',CVAccuracy)
	print('Test Accuracy ',testAccuracy)
	pred = clf.predict(XTest)
	pr = precision_recall_fscore_support(yTest,pred)
	
	print('Test Sensitivity ',pr[1][1])
	conf = confusion_matrix(yTest,pred)
	print('Test Specificity ',conf[0,0]/(conf[0,0] + conf[0,1]))


#--------------------------------------------------------------------------------------------------------------

fullDataFrame = makeDataFrameTFIDF(['chr1'])
fullDataFrame = makeDataFrameFrequency(['chr1'])
fullDataFrame = fullDataFrame.iloc[np.random.permutation(len(fullDataFrame))]#shuffle the data
trainSetDataFrame,testSetDataFrame = model_selection.train_test_split(fullDataFrame,test_size=0.30)# divide the full dataset into two
#let us save the datasets for further use
#trainSetDataFrame.to_csv('TrainSet1_19_3.csv',index = False)
#testSetDataFrame.to_csv('TestSet1_19_3.csv',index = False)
fullX = fullDataFrame.iloc[:,0:len(fullDataFrame.columns) - 1]
fullY = fullDataFrame.iloc[:,len(fullDataFrame.columns) - 1]
#fullY = fullDataFrame.target
#fullDataFrame.to_csv('chr1_19_3Data.csv',index=False)
XTrain,XTest,yTrain,yTest = model_selection.train_test_split(fullX,fullY,test_size=0.3)

#trainSetDataFrame = pd.read_csv('TrainSet1_19_3.csv')
#testSetDataFrame = pd.read_csv('TestSet1_19_3.csv')

#XTrain = trainSetDataFrame.iloc[:,0:len(trainSetDataFrame.columns) - 1]
#XTest = testSetDataFrame.iloc[:,0:len(testSetDataFrame.columns) - 1]
#yTrain = trainSetDataFrame.iloc[:,len(trainSetDataFrame.columns) - 1]
#yTest = testSetDataFrame.iloc[:,len(testSetDataFrame.columns) - 1]
runNaiveBayes(XTrain,XTest,yTrain,yTest,fullX,fullY)
for c in [1,5,10,15,20]:
#      :
#		print('C - gamma ',c,'  ',gamma)
               runSVM(XTrain,XTest,yTrain,yTest,None,None,C=c,gamma=gamma)
#XTrain,XCV,yTrain,yCV = model_selection.train_test_split(XTrain,yTrain,test_size=0.2)
for gamma in [0.1,1,2,3,4,5]:
	for C in [6,1,5,10,15]:
		print('For gamma,C ',gamma,'  ',C)
		runSVMGaussian(XTrain,XTest,yTrain,yTest,XCV,yCV,fullX=None,fullY=None,C=C,gamma=gamma)
#		break


#for c in [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]:
#	print('C = ',c)
#	runLinearSVM(XTrain,XTest,yTrain,yTest,None,None,C=c,gamma=1)


#print(fullY)
#print(yTrain,yTest)

#fullDataMatrix = fullDataFrame.as_matrix()


#for row in range(np.size(fullDataMatrix,axis=0)):
#	print((fullDataMatrix[row,1]))
