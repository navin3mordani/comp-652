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



def EncodeChromoAsObservations(inputFastaFilePath,chromoName,KMerLength):

	if not chromoname in inputFastaFilePath:
		print('The chromoName in inputFastaFilePath check failed. Check the arguments')
		return
	KMerColumnNumberDict = pickle.load(open('KMerColumnNumberPickle/' + str(KMerLength) + '/columnNumberDict.p','rb'))
	observationList = [] # the list to store the final list of observations
	observationStringList = []#stores the observations but in the form of strings 
	KMer = None #variable to store the current KMer while scanning
	for seq_record in SeqIO.parse(inputFastaFilePath, "fasta"): #get the sequences of regularory regions along with their ids
		X = seq_record.seq
		#lets work with strings instead
		#so cast the seq object to a string
		seqString = str(X).upper()
		for loop in range(len(seqString)-KMerLength + 1):
			KMer = seqString[loop:loop + KMerLength]
			try:
				observation = KMerColumnNumberDict[KMer]#get the column number which is also the observation
				observationList.append(observation)
				observationStringList.append(str(observation))
			except KeyError as e:
				if('N' in KMer or 'R' in KMer or 'F' in KMer or 'L' in KMer):
					continue
				else : raise e


	#now the observationList contains all observation

	#store the list as pickle file and also write it to a txt file
	pickle.dump(observationList,open('ObservationsForHMM/' + KMerLength + '/' + chromoName + '.p','wb'))
	#convert the observationStringList to a big string
	observationString = ' '.join(observationStringList)
	#and write it to a txt file
	text_file = open('ObservationsForHMM/' + KMerLength + '/' + chromoName + '.txt', "w")
	text_file.write(observationString)
	text_file.close()
	print('done' + 'chromoname')

#--------------------------End of EncodeChromoAsObservations -----------------------


EncodeChromoAsObservations(inputFastaFilePath = 'chromFa/chr2.fa',chromoName='chr2',KMerLength=6)
