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




def makeIndexOfPositiveExamples(KMerLength):
	chromoKeyList = pickle.load(open('chromoKeyList/chromoKeyList.p','rb')) #get the list of chromosome names chr1-chr22
	permutations = [''.join(p) for p in product('ACGT',repeat=KMerLength)] #get permutations of a specfied length of ACGT
	
	'''
	for every chromosome get the set of regulatory regions of the GM12878
	the sequence of the regulatory regions is stored in the folder named 
	RegulatorySeq
	'''
	for chromoKey in chromoKeyList:
		positiveIndex = {perm: 0 for perm in permutations}#make a dictionary that will save the frequency of K-Mers of positive Examples
		regulatoryFastaFilePath = 'RegulatorySeq/' + chromoKey + '.fa'
		for seq_record in SeqIO.parse(regulatoryFastaFilePath, "fasta"): #get the sequences of regularory regions along with their ids
				X = seq_record.seq
				#lets work with strings instead
				#so cast the seq object to a string
				seqString = str(X).upper()
				#print(seqString)
				'''
				loop through the K-Mers and then store them in the dictionary
				'''
				for loop in range(len(seqString)-KMerLength + 1):
					try:
											
						positiveIndex[seqString[loop:loop + KMerLength]] += 1
					except KeyError as e:
						if('N' in  seqString[loop:loop + KMerLength]):
							continue
						else: raise e
						
				
		outputFile = str(KMerLength) + chromoKey + '.p'
		outputFilePath = 'positiveIndex/'+ str(KMerLength) + '/' + outputFile
		pickle.dump(positiveIndex,open(outputFilePath,'wb'))
		print(chromoKey)
		break 






makeIndexOfPositiveExamples(6)