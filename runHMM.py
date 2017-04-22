import glob
import pandas as pd 
import re
import pickle
import random
import os
import numpy as np 
from hmmlearn import hmm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from hsmmlearn.hsmm import HSMMModel 
from hsmmlearn.emissions import GaussianEmissions
from hsmmlearn.emissions import MultinomialEmissions
from scipy.stats import poisson
from scipy.stats import bernoulli



def runHMM(flankLength,KMerLength,TransitionMatrix,EmissionMatrix,InitialProbabilityMatrix,numberOfStates,ObservationList,outputFilePath):

	model = hmm.MultinomialHMM(n_components=numberOfStates,algorithm='viterbi')

	model.startprob_  = InitialProbabilityMatrix
	model.transmat_ = TransitionMatrix
	model.emissionprob_ = EmissionMatrix
	# Predict the optimal sequence of internal hidden state
	#print(model.startprob_)
	#print(model.transmat_)
	#X = np.atleast_2d(ObservationList).T
	X = np.reshape(np.array(ObservationList),(-1,1))
	
	S = (model.predict(X))
	pickle.dump(S,open(outputFilePath,'wb'))
	print(S)
	print('done ',type(S))
	#generate an array with strings
	x_arrstr = np.char.mod('%f', S)
	#combine to a string
	x_str = ",".join(x_arrstr)
        #S = ' '.join(list(S))
	txt_file = open('ans.txt','w')
	txt_file.write(x_str)
	txt_file.close()
	return S

#-------------------------------------END of runHMM function-----------------------------------------------------------

def runHSMM(flankLength,KMerLength,Durations,TransitionMatrix,EmissionMatrix,InitialProbabilityMatrix,numberOfStates,ObservationList,outputFilePath):
	emissions = MultinomialEmissions(EmissionMatrix)
	Durations = [poisson(1)] * numberOfStates
	hsmm = HSMMModel(
    emissions, Durations, TransitionMatrix,startprob = InitialProbabilityMatrix
)
	X = np.reshape(np.array(ObservationList),(-1,1))
	decoded_states = hsmm.decode(X)
	return decoded_states



#----------------------------------------------------------------------
def calculateMatches(actualStateSeq,ObservedStateSeq,flankLength,KMerLength):
	
	ObservedStateSeqList = list(ObservedStateSeq)
	regulatoryStateNumber = (flankLength - KMerLength + 1) + 1#set the regulatory state number
	#prepare a list of indices at which we predict regulatory
	indicesRegulatory = [i for i, x in enumerate(ObservedStateSeqList) if x == regulatoryStateNumber]
	countCorrectRegulatory = 0#count how many regulatory are correctly predicted
	numberOfRegulatoryInActualStateSeq = actualStateSeq.count('R')#how many Rs are there in actual state seq
	#for all the Regulatory we predict check if it is actually an R in actual seq
	for indexRegulatory in indicesRegulatory:
		if(actualStateSeq[indexRegulatory] == 'R'):#if predicted and actual Regulatory match then inceremnt correctRegulatory count
			countCorrectRegulatory += 1
	print('Regulatory = ',str(countCorrectRegulatory),' out of ',str(numberOfRegulatoryInActualStateSeq),' total Regulatory Predicted',str(len(indicesRegulatory)))

	#Non-regulatory state number = 0
	nonRegulatoryStateNumber = 0
	indicesNonRegulatory = [i for i, x in enumerate(ObservedStateSeqList) if x == nonRegulatoryStateNumber]
	countCorrectNonRegulatory = 0#count the number of NonRegulatory correctly predicted
	numberOfNonRegulatoryInActualStateSeq = actualStateSeq.count('O') + actualStateSeq.count('A') + actualStateSeq.count('C') + actualStateSeq.count('G') + actualStateSeq.count('T')
	for indexNonRegulatory in indicesNonRegulatory:
		if(actualStateSeq[indexNonRegulatory] in ['A','C','G','T','O']):
			countCorrectNonRegulatory += 1
	print('NonRegulatory = ',str(countCorrectNonRegulatory),' out of ',str(numberOfNonRegulatoryInActualStateSeq),' total NonRegulatory predicted ',str(len(indicesNonRegulatory)))

	
#------------------------------------------END of CALCULATE MATCHES----------------------------------------------

def modifyTransitionMatrixForHSMM(flankLength,KMerLength,numberOfStates,TransitionMatrix):
	modifiedTransitionMatrix = TransitionMatrix
	#set the diagonal elements to zero
	#np.fill_diagonal(modifiedTransitionMatrix,0)
	#only for non-regulatory(state 0) and regulatory(state (flankLength - KMerLength + 1) + 1)
	#set transition to next (state,state+1) = (state,state) + (state,state+1) 
	#(flankLength - KMerLength + 1) + 1
	print(np.size(TransitionMatrix) - np.count_nonzero(TransitionMatrix))
	for i in range(np.size(modifiedTransitionMatrix,axis=0)):
		modifiedTransitionMatrix[i,(i+1)%numberOfStates] = TransitionMatrix[i,i] + TransitionMatrix[i,(i+1)%numberOfStates]
		
	for i in range(np.size(modifiedTransitionMatrix,axis=0)):
		#print(TransitionMatrix[i,:])
		break
		if(not np.sum(TransitionMatrix[i,:]) == np.sum(TransitionMatrix[:,i])):
			print('Yes ',np.sum(TransitionMatrix[i,:]),'  ',np.sum(TransitionMatrix[:,i]))
			print(TransitionMatrix[i,:])
			break
		if (not np.sum(modifiedTransitionMatrix[i,:]) == np.sum(modifiedTransitionMatrix[:,i])):
			print(i)
			print(np.sum(modifiedTransitionMatrix[i,:]),' ',np.sum(modifiedTransitionMatrix[:,i]))
	return modifiedTransitionMatrix

flankLength = 147
KMerLength = 6
chromoToTrain = 'chr1'
chromoToTest = 'chr2'
TransitionMatrix = pickle.load(open('TransitionMatrices/' + str(flankLength) + '/' + str(KMerLength) + '/' + chromoToTrain + '.p','rb'))
EmissionMatrix = pickle.load(open('EmissionProbabilityVectors/FullEmissionProbMatrix/' + str(flankLength) + '/' + str(KMerLength) + '/' + chromoToTrain + '.p','rb'))

numberOfStates = (flankLength - KMerLength + 1) * 2 + 1 + 1
InitialProbabilityMatrix = [0] * numberOfStates
InitialProbabilityMatrix[0] = 0.8
InitialProbabilityMatrix[(flankLength - KMerLength + 1) + 1] = 0.2
proxyProb = 1e-08

InitialProbabilityMatrix = np.array(InitialProbabilityMatrix) 

InitialProbabilityMatrix[InitialProbabilityMatrix == 0] = proxyProb
InitialProbabilityMatrix = InitialProbabilityMatrix / InitialProbabilityMatrix.sum(axis=0,keepdims=True)

ObservationList = pickle.load(open('ObservationsForHMM/' + str(KMerLength) + '/' + chromoToTest + '.p','rb'))

actualStateSeq = None
"""genomeWithoutLeftRightAndRegFilePath = 'genomeWithoutLeftRightAndReg/147/' + chromoToTest  + '.fa' 
for seq_record in SeqIO.parse(genomeWithoutLeftRightAndRegFilePath, "fasta"):
	X = seq_record.seq
	actualStateSeq = str(X).upper()

actualStateList = []#annotate the actual sequence as states giving out KMers 
actualStateSeq = actualStateSeq.replace("A","O")
actualStateSeq = actualStateSeq.replace("C","O")
actualStateSeq = actualStateSeq.replace("G","O")
actualStateSeq = actualStateSeq.replace("T","O") #
#Lets scan the actual seq using window of KMerLength 
for loop in range(len(actualStateSeq) - KMerLength + 1):
	#replace position by the last alphabet of the KMer as that will be the state(R,L,F,O,N)
	
	KMer = actualStateSeq[loop:loop+KMerLength]
	if "N" in KMer:
		continue
	#if len(set(list(KMer))) == 1:
	actualStateList.append(KMer[KMerLength-1])
actualStateSeq = ''.join(actualStateList)
actualStateSeq = actualStateSeq.replace("N","")
print('lens - ',len(ObservationList),' ',len(actualStateSeq),'  ',len(actualStateList))

#pickle.dump(actualStateSeq,open("ActualStateSeq_" + chromoToTest +".p","wb"))
"""
actualStateSeq = pickle.load(open("ActualStateSeq_" + chromoToTest +".p","rb"))
for i in range(20):
	#break
	ObservationList1 = ObservationList[i*100000:(i+1)*100000]
	outputFilePath = 'StateSeq/' + str(flankLength) + '/' + str(KMerLength) + '/' + "0_1000" + chromoToTest + '.p'
	ObservedStateSeq = runHMM(flankLength ,KMerLength ,TransitionMatrix ,EmissionMatrix,InitialProbabilityMatrix,numberOfStates,ObservationList1,outputFilePath)		
	
	actualStateSeq1 = actualStateSeq[i*100000:(i+1) * 100000]
	calculateMatches(actualStateSeq1,ObservedStateSeq,147,6)
	print(len(ObservationList1),' is len of observed')
	print(len(actualStateSeq1),' is len of actual')
	print(list(actualStateSeq1).count('R')+list(actualStateSeq1).count('L')+list(actualStateSeq1).count('F'))
	print(list(actualStateSeq1).count('A')+list(actualStateSeq1).count('C')+list(actualStateSeq1).count('G')+list(actualStateSeq1).count('T'))
	pickle.dump(ObservedStateSeq,open("OBS" + str(i)+".p","wb"))
	#pickle.dump(actualStateSeq1,open("ACTS"+str(i)+".p","wb"))
	print(list(set(list(actualStateSeq1))))


#read the length distributions
chromoKeyList = pickle.load(open('chromoKeyList/chromoKeyList.p','rb')) #get the list of chromosome names chr1-chr22
positiveLength = []
nonRegulatoryLength = []
for chromoKey in chromoKeyList:
	positiveLengthDistribution = pickle.load(open('PositiveLisLenPickle/lisLen/' + chromoKey + '.p','rb'))
	nonRegulatoryLengthDistribution = pickle.load(open('NonRegulatoryLisLenPickle/lisLen/147/' + chromoKey + '.p','rb'))
	#positiveLength.append(positiveLengthDistribution)
	print(type(positiveLengthDistribution))
	positiveLength += (positiveLengthDistribution)
	nonRegulatoryLength += (nonRegulatoryLengthDistribution)
print(type(positiveLength),type(nonRegulatoryLength))
meanLengthNonRegulatory = np.mean(np.array(nonRegulatoryLength))
meanLengthPositive = np.mean(np.array(positiveLength))
print(meanLengthNonRegulatory,'  ',meanLengthPositive)


flankDurationList = [bernoulli(1)] * (flankLength - KMerLength + 1)
Durations = [poisson(meanLengthNonRegulatory)]
Durations = Durations + flankDurationList + [poisson(meanLengthPositive)] + flankDurationList
print(len(Durations),'Duration')
#avgLengthNonRegulatory = 1/TransitionMatrix[0,0]
#row = (flankLength - KMerLength + 1) + 1
#avgLengthRegulatory = 1 / TransitionMatrix[row,row]
#print(avgLengthNonRegulatory,'',avgLengthRegulatory)
modifiedTransitionMatrix = modifyTransitionMatrixForHSMM(flankLength,KMerLength,numberOfStates,TransitionMatrix)
#runHSMM(flankLength,KMerLength,Durations,modifiedTransitionMatrix,EmissionMatrix,InitialProbabilityMatrix,numberOfStates,ObservationList,outputFilePath)runHSMM(flankLength,KMerLength,Durations,TransitionMatrix,EmissionMatrix,InitialProbabilityMatrix,numberOfStates,ObservationList,outputFilePath):
for i in range(20):
	#break
	ObservationList1 = ObservationList[i*1000:(i+1)*1000]
	outputFilePath = 'StateSeq/' + str(flankLength) + '/' + str(KMerLength) + '/' + "0_1000" + chromoToTest + '.p'
	
	ObservedStateSeq = runHSMM(flankLength,KMerLength,Durations,TransitionMatrix,EmissionMatrix,InitialProbabilityMatrix,numberOfStates,ObservationList,outputFilePath)
	actualStateSeq1 = actualStateSeq[i*1000:(i+1) * 1000]
	calculateMatches(actualStateSeq1,ObservedStateSeq,147,6)
	print(len(ObservationList1),' is len of observed')
	print(len(actualStateSeq1),' is len of actual')
	print(list(actualStateSeq1).count('R')+list(actualStateSeq1).count('L')+list(actualStateSeq1).count('F'))
	print(list(actualStateSeq1).count('A')+list(actualStateSeq1).count('C')+list(actualStateSeq1).count('G')+list(actualStateSeq1).count('T'))
	#pickle.dump(ObservationList1,open("OBS" + str(i)+".p","wb"))
	#pickle.dump(actualStateSeq1,open("ACTS"+str(i)+".p","wb"))
	print(list(set(list(actualStateSeq1))))

