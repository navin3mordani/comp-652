import glob
import pandas as pd 
import re
import pickle
import random
import os
import numpy as np 




def makeTransitionMatrix(flankLength,KMerLength,meanLengthPositive,meanLengthNonRegulatory):
	
	#numberOfStates = (flankLength - KMerLength + 1) * 2 + 1 + 1
	#1. 1 for non-regulatory
	#2. (flankLength - KMerLength + 1) for left flank
	#3. 1 for regulatory
	#4. (flankLength - KMerLength + 1) for right flank


	numberOfStates = (flankLength - KMerLength + 1) * 2 + 1 + 1
	
	#make transitionMatrix of size numberOfStates*numberOfStates
	transitionMatrix = np.zeros((numberOfStates,numberOfStates))

	#fill the first row for state non-regulatory state 0
	#1/meanLengthNonRegulatory for going to state 1(1st state of left flank) (column - 1)
	#1 - 1/meanLengthNonRegulatory for remaining in the same state 0 (column - 0)
	transitionMatrix[0,0] = 1 - 1/meanLengthNonRegulatory
	transitionMatrix[0,1] = 1/meanLengthNonRegulatory

	#Next for the left flank states
	#for the next flankLength - KMerLength + 1 rows 
	#fill the value one for the entry (row,row+1) = 1
	for row in range(1,flankLength - KMerLength + 1 + 1):
		transitionMatrix[row,row + 1] = 1

	#Next we reach the state of regulatory
	#fill with 1/meanLengthPositive to go to next state(1st of right flank)(column = row+1)
	#fill with 1 - 1/meanLengthPositive to remain in the state(row,row)

	row = (flankLength - KMerLength + 1) + 1
	transitionMatrix[row,row] = 1 - 1/meanLengthPositive
	transitionMatrix[row,row+1] = 1/meanLengthPositive

	#Next we come to the right flank
	#fill all with row,row+1 = 1
	#except the last one as the last one points to state zero
	for row in range((flankLength - KMerLength + 1) + 1 + 1:numberOfStates):
		transitionMatrix[row,(row+1)%numberOfStates] = 1

	transitionMatrix[transitionMatrix == 0] = proxyProb
	transitionMatrix = transitionMatrix / transitionMatrix.sum(axis=1,keepdims=True)

#------------------------------------------------

#read the length distributions
chromoKeyList = pickle.load(open('chromoKeyList/chromoKeyList.p','rb')) #get the list of chromosome names chr1-chr22
for chromoKey in chromoKeyList:
	positiveLengthDistribution = pickle.load(open('PositiveLisLenPickle/lisLen/' + chromoKey + '.p','rb'))
	nonRegulatoryLengthDistribution = pickle.load(open('NonRegulatoryLisLenPickle/lisLen/147/' + chromoKey + '.p','rb'))
	meanLengthNonRegulatory = np.mean(nonRegulatoryLengthDistribution)
	meanLengthPositive = np.mean(positiveLengthDistribution)

#positiveLength.append(positiveLengthDistribution)
#nonRegulatoryLength.append(nonRegulatoryLengthDistribution)
