import glob
import pandas as pd 
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from itertools import permutations
from itertools import product
import pickle
import random
import os
import numpy as np 

def makeEmissionProbMatrixForOneState(inputPickleFilePath,outputPickleFilePath,KMerLength):
	try:
		inputIndex = pickle.load(open(inputPickleFilePath,'rb'))
		KMerColumnNumberDict = pickle.load(open('KMerColumnNumberPickle/' + str(KMerLength) + '/columnNumberDict.p','rb'))
	
	except Exception as e:
		raise e
	emissionProbabilityVector = [0] * len(KMerColumnNumberDict.keys())
	totalKMerCount = sum(inputIndex.values())
	print(totalKMerCount)
	for KMerKey in inputIndex.keys():
		prob = inputIndex[KMerKey] / totalKMerCount
		columnNumber = KMerColumnNumberDict[KMerKey]
		emissionProbabilityVector[columnNumber] = prob
		if not prob == 0:
			print(KMerKey,' ',prob,' ',emissionProbabilityVector[columnNumber])
	emissionProbabilityVector = np.array(emissionProbabilityVector)
	pickle.dump(emissionProbabilityVector,open(outputPickleFilePath,'wb'))

#-----------------------End of makeEmissionProbMatrixForOneState-----------------------

def makeEmissionProbMatrixForFlanks(inputPickleFilePath,outputPickleFilePath,KMerLength,flankLength):
	try:
		inputIndex = pickle.load(open(inputPickleFilePath,'rb'))
		KMerColumnNumberDict = pickle.load(open('KMerColumnNumberPickle/' + str(KMerLength) + '/columnNumberDict.p','rb'))
	except Exception as e:
		raise e
	permutations = [''.join(p) for p in product('ACGT',repeat=KMerLength)] #get permutations of a specfied length of ACGT
	flankEmissionProbMatrix = np.zeros((4096,142))
	#print(np.shape(flankFrequencyMatrix))
	#print(sum(inputIndex.values()))
	KMerWise2dArray = []  # get an array of (4096,142)
	for KMerKey in (sorted(KMerColumnNumberDict.keys())):
		KMerWise2dArray.append(inputIndex[KMerKey])
	print(inputIndex['AAAAAA'])
		
	KMerWise2dArray = np.array(KMerWise2dArray)#shape is (4096,142)
	print(np.shape(KMerWise2dArray))
	sumOfEachColumnList = np.array((np.sum(KMerWise2dArray,axis=0)))
	for row in range(np.size(KMerWise2dArray,axis=0)):
		for col in range(np.size(KMerWise2dArray,axis=1)):
			flankEmissionProbMatrix[row][col] = KMerWise2dArray[row][col] / sumOfEachColumnList[col]#one column per state
	print(sumOfEachColumnList[0],'	',flankEmissionProbMatrix[0][0])
	pickle.dump(flankEmissionProbMatrix,open(outputPickleFilePath,'wb'))

	#for KMerKey in KMerColumnNumberDict.keys():


	#for loop in range(flankLength - KMerLength + 1):
	#	for KMerKey in inputIndex.keys():
	#		a=1
			





def makeFullEmisionProbMatrix(flankLength,KMerLength,chromoName,proxyProb):
	numberOfStates = 1 + 1 + (flankLength - KMerLength + 1) * 2 #one for NonRegulatory,1 for Regulatory and rest for flanks
	
	"""
	Let us read all the prob vectors
	"""
	nonRegulatoryProbVector = pickle.load(open('EmissionProbabilityVectors/NonRegulatoryIndex/' + str(flankLength) + '/' + str(KMerLength) + '/' + str(KMerLength) + chromoName + '.p','rb'))
	regulatoryProbVector = pickle.load(open('EmissionProbabilityVectors/positiveIndex/' + str(KMerLength) + '/' + str(KMerLength) + chromoName + '.p','rb'))
	leftFlankProbMatrix = pickle.load(open('EmissionProbabilityVectors/FlankIndex/Left/' + str(flankLength) + '/' + str(KMerLength) + '/' + chromoName + '.p','rb'))
	rightFlankProbMatrix = pickle.load(open('EmissionProbabilityVectors/FlankIndex/Right/' + str(flankLength) + '/' + str(KMerLength) + '/' + chromoName + '.p','rb'))

	#order of states in the matrix 
	#0. NonRegulatory
	#1-(flankLength - KMerLength + 1): for Left Flank
	#(flankLength - KMerLength + 1) + 1: for Regulatory
	#(flankLength - KMerLength + 1) + 1 + 1 to (flankLength - KMerLength + 1)*2 + 1:RightFlank

	fullEmissionProbMatrix = []   # size numberOfStates * 4^KMerLength
	
	#1. Add the Emission prob vector of NonRegulatory
	fullEmissionProbMatrix.append(list(nonRegulatoryProbVector))

	#2. Add the left flanks
	"""Extract the column from leftFlankProbMatrix and add it to row of fullEmissionProbMatrix"""
	for columnNumber in range(np.size(leftFlankProbMatrix,axis=1)):
		fullEmissionProbMatrix.append(list(leftFlankProbMatrix[:,columnNumber]))

	#3. Add the regulatoryProbVector
	fullEmissionProbMatrix.append(list(regulatoryProbVector))

	#4. Add the right flanks 
	"""
	Extract columns from right flank matrix and add it to emission prob rows
	"""
	for columnNumber in range(np.size(rightFlankProbMatrix,axis=1)):
		fullEmissionProbMatrix.append(list(rightFlankProbMatrix[:,columnNumber]))

	print(np.shape(np.array(fullEmissionProbMatrix)))
	fullEmissionProbMatrix = (np.array(fullEmissionProbMatrix))


	print(fullEmissionProbMatrix)
	fullEmissionProbMatrix[fullEmissionProbMatrix == 0] = proxyProb
	fullEmissionProbMatrix = fullEmissionProbMatrix / fullEmissionProbMatrix.sum(axis=1,keepdims=True)
	pickle.dump(fullEmissionProbMatrix,open('EmissionProbabilityVectors/FullEmissionProbMatrix/' + str(flankLength) + '/' + str(KMerLength) + '/' + chromoName + '.p','wb'))
	'''for row in range(np.size(fullEmissionProbMatrix,axis=0)):
		#numberOfZerosInRow = np.size(fullEmissionProbMatrix,axis=1) - np.count_nonzero(fullEmissionProbMatrix[row,:])
		#print(numberOfZerosInRow)
		#fullEmissionProbMatrix[fullEmissionProbMatrix==0] = proxyProb
		sumOfRow = np.sum(fullEmissionProbMatrix[row,:])
		#funp.divide(fullEmissionProbMatrix[row,:],sumOfRow)
		print(np.min(fullEmissionProbMatrix[row,:]),' ',sumOfRow)
	'''
#---------------------------
#for indexFile in glob.glob('positiveIndex/6/*.p'):
#	makeEmissionProbMatrixForOneState(indexFile,'EmissionProbabilityVectors/' + indexFile,6)

#for indexFile in glob.glob('NonRegulatoryIndex/147/6/*.p'):
#	makeEmissionProbMatrixForOneState(indexFile,'EmissionProbabilityVectors/' + indexFile,6)
#for indexFile in glob.glob('FlankIndex/Left/147/6/*.p'):
#	makeEmissionProbMatrixForFlanks(indexFile,'EmissionProbabilityVectors/' + indexFile,6,147)
#for indexFile in glob.glob('FlankIndex/Right/147/6/*.p'):
#	print(indexFile)
#	makeEmissionProbMatrixForFlanks(indexFile,'EmissionProbabilityVectors/' + indexFile,6,147)
makeFullEmisionProbMatrix(147,6,'chr1',proxyProb = 1e-07)