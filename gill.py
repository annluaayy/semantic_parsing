#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import os, glob
import re
from operator import itemgetter
import numpy as np
import math

class InitialData:

	"""
	to extract data in pair (utterance, MR) from directory 'one-to-one-mapping-data';
	MR is a set of meanings corresponding to a certain utterance

	>>> utterance: a block moves across the red region
	>>> MR: (block obj-0)
			(movement mov-0 obj-0)
			(move-across-event path-0 mov-0 obj-0 reg-0)
			(red reg-0)
			(region reg-0)
	"""

	def __init__(self, path_utte, path_meaning):
		
		# path is the directory stocking the files of utterances and meanings
		self.path_utte = path_utte
		self.path_meaning = path_meaning

		# initiate a list for stocking the pairs of utterance and corresponding meaning
		self.pairs = []

	def makePair(self):
		"""
		to extract utterances paired with their corresponding sets of meaning from the directory path
		"""
		# initiate a dic for all the utterances with key as their numero of file
		utterances = {}

		utteranceFiles = glob.glob(self.path_utte)
		meaningFiles = glob.glob(self.path_meaning)

		
		for file in utteranceFiles:
			# to identifier the numero of running utterance file 
			num_file = re.sub(r'\D','',file)
			with open(file) as fp:
				for line in fp:
					# remove '-' from sentence
					while '-' in line:
						line = line.replace('-', ' ')
					utterances[num_file] = line
		
		
		for file in meaningFiles:
			# to identifier the numero of running meaning file 
			num_file_m = re.sub(r'\D','',file)
			with open(file) as fp:
				# initiate a list for a set of concepts of an utterance
				mrSet = []
				for line in fp:
					# replace the instantiated digital numbers with variable like x,y,z by recalling function variableReplace
					line = self.variableReplace(line)
					mrSet.append(line)
				utte_meaning = (utterances[num_file_m],mrSet)
				self.pairs.append(utte_meaning)
				

	def variableReplace(self, sentence):
		"""
		to replace all the digital number of instantialization by variable x, y ,z ......
	
		>>> obj-0 replaced by obj-x
		>>> mov-0 replaced by mov-y
		>>> reg-0 replaced by reg-z
		>>> path-0 replaced by path-m
		"""	
		if "obj-" in sentence:
			sentence = re.sub(r'obj-\d*','obj-x', sentence)
		if "mov-" in sentence:
			sentence = re.sub(r'mov-\d*','mov-y',sentence)
		if "reg-" in sentence:
			sentence = re.sub(r'reg-\d*','reg-z',sentence)
		if "path-" in sentence:
			sentence = re.sub(r'path-\d*','path-m',sentence)
		return sentence


class Gill:

	"""
	GIll (graph intersection lexicon learning) algorithm is mentioned in Chen.L.David's dissertation;
	using the intersections between the pairs of navigation plans to learn a lexicon

	here what we have is more of a tempo-spatial language representation
	"""

	def __init__(self, dataPairs, k=10):

		# set k for selecting the most plausible meanings
		self.k = k		

		self.dataPairs = dataPairs

		# a list for all the unigrams in the utterances
		self. unigrams = []

		# a list for all the bigrams in the utterances
		self.bigrams = []

		# create a dictionary of the words(unigrams or bigrams) with all their possible meanings 
		self.initMeaning = {}

		# create dictionnary for all the intersections of a certain word(n_Gram)
		self.intersections = {}

		# creat a dictionary for the count of all the ngrams
		# we count how many examples contain the word(ngrams) by igoring mutiple occurrences in a single example
		self.countUnigrams={}
		self.countBigrams={}
		self.countNgrams = {}
		self.total_words = len(self.dataPairs)

		# creat a dictionary for the count of all the temporal_spatial meaning representations
		self.countMeaning = {}

		# creat a dictionary for storing the final entries of each word
		self.lexicon = {}

		# this version of dictionary is created for the application of idf in the program
		self.lexicon_idf = {}

	def mainAlgo(self,threshold=0.4):
		"""
		Main algorithm for the implementation of GILL

		k: set k for selecting the most plausible meanings		
		threshold: set a threshold for selecting the final candidates
		"""

		# to collect the unigrams and th bigrams by recalling the two functions
		self.unigramsCollect()
		#self.bigramsCollect()

		nGramsLs = self.unigrams + self.bigrams
		
		for ngram in nGramsLs:
			# initiate all the possible meanings for a certain n_Grams
			self.initMeaning[ngram]=[]
			for pair in self.dataPairs:
				# add possible meanings for a certain word(ngram)
				if ngram in pair[0]:
					self.initMeaning[ngram].append(pair[1])	

			# find the intersections between each pair of meaning sets
			candidates = self.findIntersect(ngram)
			# iterate this searching of intersections till no new overlapping
			while len(candidates) >0:
				self.initMeaning[ngram]=[]
				newMeanings = [x[0] for x in candidates]
				self.initMeaning[ngram].extend(newMeanings)
				# preserve score list in case it's the last iteration
				scoreLs = []
				for ele in candidates:
					scoreLs.append(ele)

				candidates = self.findIntersect(ngram)
			
			# storing the final results
			self.lexicon[ngram] = []
			for couple in scoreLs:
				if couple[1] > threshold or couple[1] == threshold:
					self.lexicon[ngram].append(couple[0])

			# storing with the implementation of idf
			self.lexicon_idf[ngram] = []
			for meaning_set in self.lexicon[ngram]:
				for element in meaning_set:
					if self.tf_idf(element) != 0:
						self.lexicon_idf[ngram].append(element)
			

	

	def tf_idf(self,meaning):
		"""
		to use term weighting solution tf_idf(term frequency and inverse document frequency)

		to check how important a word is to a document, in this way, to discriminate the word
		>>> like "a", "the", "move", "the" appear in all document


		p.s. for now, I ignore the tf, just use the idf, to avoid the interference of the common words'meanings
		"""
		# number for the total documents in the corpus
		D = len(self.dataPairs)
		# number of the documents containing the meaning term
		n_t = 0
		# initiate the idf value to 0.0
		idf = 0.0
		for pair in self.dataPairs:
			if meaning in pair[1]:
				n_t += 1
		
		return math.log(float(D)/float(n_t),2)
		
	
	def findIntersect(self,word):
		"""
		to compute the intersections between each pair of all the possible meanings for a certain n_Gram 
		"""
		# initiate the intersections list for a certain word as a vacant list
		self.intersections[word] = []

		# find intersections between every pair of meanings in initMeaning[word] list
		for ind_1,meaning1 in enumerate(self.initMeaning[word]):
			for ind_2,meaning2 in enumerate(self.initMeaning[word]):
				if meaning1 != meaning2:
					# greedy way to find the intersections by finding the larger common parts firstly
					largestCommonPart = self.largestCommon(meaning1,meaning2)

					lsmeaning1=[x for x in meaning1]
					lsmeaning2=[x for x in meaning2]

					# find all the intersections between this meaning pair
					while len(largestCommonPart) > 0 :
						for common in largestCommonPart:
							if common not in self.intersections[word]:
								self.intersections[word].append(common)
						
						# remove the common parts found so far
						for common in largestCommonPart:
							ind_begin = lsmeaning1.index(common[0])
							ind_end = lsmeaning1.index(common[-1])
							
							ind_debut = lsmeaning2.index(common[0])
							ind_fin = lsmeaning2.index(common[-1])
							
							if ind_end != len(lsmeaning1)-1:
								lsmeaning1 = lsmeaning1[:ind_begin]+lsmeaning1[(ind_end+1):]
							else:
								lsmeaning1 = lsmeaning1[:ind_begin]

							if ind_debut != len(lsmeaning2)-1:
								lsmeaning2 = lsmeaning2[:ind_debut]+lsmeaning2[(ind_fin+1):]
							else:
								lsmeaning2 = lsmeaning2[:ind_debut]
						
						# continue to find the next largest common parts
						largestCommonPart = self.largestCommon(lsmeaning1,lsmeaning2)

						
		# give a scoring function for every pair of (word,meaning)
		# to get the candidate meanings, k highest-scoring entries for meaning(word)
		if len(self.intersections[word]) > 0:
			candidates = self.scoreFunction(self.intersections[word],word)		
		else:
			candidates = []
		return candidates
				

	def largestCommon(self,meaning1,meaning2):
		"""
		LCSubstring (Longest Common Substring) Problem;
		Except here, we compare two lists instead of two strings,
		to find the longest common sublist

		return a set of largest common parts or intersect lists
		"""
		m = len(meaning1)
		n = len(meaning2)
		# create hash table
		dic_common = {}
		#initiate the longest longueur
		longest = 0
		# create a list for storing the longest common substring
		longest_sub = []
		
		for i_a,mr_a in enumerate(meaning1):
			for i_b,mr_b in enumerate(meaning2):
				if mr_a == mr_b:
					if i_a == 0 or i_b == 0:
						dic_common[(i_a,i_b)] = 1
					else:
						if (i_a-1,i_b-1) in dic_common:
							dic_common[(i_a,i_b)] = dic_common[(i_a-1,i_b-1)] +1
						else:
							dic_common[(i_a,i_b)] = 1

					# storing the longest common sublist by now
					if dic_common[(i_a,i_b)] > longest:
						longest = dic_common[(i_a,i_b)]
						longest_sub = [meaning1[(i_a-longest+1):(i_a+1)]]
					elif dic_common[(i_a,i_b)] == longest:
						longest_sub.append(meaning1[(i_a-longest+1):(i_a+1)])
				# if mr_a not= mr_b, we ignore them

		# return the list containing the largest common part of two compared lists
		return longest_sub
					
					  
		
	def scoreFunction(self,intersectionLs,word):
		"""
		Given an intersection list of a word(ngram)
		return a list of scores for each intersection meaning

		>>> score(w,g) = p(g|w)-p(g|^w)
		>>> keep k highest-scoring entries of meaning(word)
		"""	

		# initiate count of co-occurrence of certain meanings and this word as dictionary
		dic_count_occ = {}
		# initiate count of certain meanings as dictionary
		dic_count_m = {}

		# initiate a list for all the intersection meaning socre
		scoreLs = []

		for ind,ele in enumerate(intersectionLs):
			for pair in self.dataPairs:
				# check if this intersection meaning in the pair
				if set(ele).issubset(set(pair[1])):
					list(ele)
					list(pair[1])
					# caz list cannot be a key in a dictionary in python, use its index instead
					if ind in dic_count_m:
						dic_count_m[ind] +=1
					else:
						dic_count_m[ind] = 1

					# check if there's any co-occurrence for word and the intersection meaning
					utteranceWords = self.lemmatizer(pair[0])	
					if word in utteranceWords:
						if (word,ind) in dic_count_occ:
							dic_count_occ[(word,ind)] +=1
						else:
							dic_count_occ[(word,ind)] =1
		

		# calculate the scores
		for ind,ele in enumerate(intersectionLs):
			
			# in the case where words like 'a','move','the','region' appear in all the examples
			###??? how can we to discriminate their meanings
			if self.total_words == self.countUnigrams[word]:
				score = 0.0
			else:
				# where meaning and word occur
				occ_positive = float(dic_count_occ[(word,ind)])/float(self.countUnigrams[word])
				# where meaning and word don't occur
				occ_negative = float(dic_count_m[ind]-dic_count_occ[(word,ind)])/float(self.total_words-self.countUnigrams[word])
				score = float(occ_positive) - float(occ_negative)
			
			scoreLs.append((ele,score))

		# to reorder the score list by descreasing
		sorted_scoreLs = sorted(scoreLs,key=itemgetter(1),reverse=True)
		resultLs = []
		if len(sorted_scoreLs) > self.k:
			resultLs = sorted_scoreLs[:(k+1)]
		elif len(sorted_scoreLs) == self.k or len(sorted_scoreLs) < self.k:
			resultLs = sorted_scoreLs
		return resultLs
		

	def lemmatizer(self, utterance):
		wordLs = utterance.split()
		# lemmatize the words in the utterance
		lmtzr = WordNetLemmatizer()
		lemmas = [lmtzr.lemmatize(x) for x in wordLs]
		return lemmas

	def unigramsCollect(self):
		
		"""
		to collect all the unigrams in the utterances
		"""

		for pair in self.dataPairs:
			# lemmatize the words in the utterance
			lemmas = self.lemmatizer(pair[0])
			# store them in list of unigrams
			self.unigrams.extend(lemmas)
			
		# dedouble the repetition in the list
		list(set(self.unigrams))

		# count the occurrence for each unigram
		for word in self.unigrams:
			self.countUnigrams[word] = self.countOcc(word)

	
	def bigramsCollect(self):

		"""
		to collect all the bigrams in the utterances
		"""
		for pair in self.dataPairs:
			# lemmatize the words in the utterance
			lemmas = self.lemmatizer(pair[0])
			bigrams = nltk.bigrams(lemmas)
			# store them in list of bigrams
			self.bigrams.extend(bigrams)
	
		# dedouble the repetition in the list
		list(set(self.bigrams))

		# count the occurrence for each bigram
		for word in self.bigrams:
			self.countBigrams[word] = self.countOcc(word)

		
	def countOcc(self,word):
		"""
		count the occurrence of the words of a list in dataPairs
		"""	
		count = 0
		for pair in self.dataPairs:
			utteranceWords=self.lemmatizer(pair[0])
			if word in utteranceWords:
				count += 1
		return count	
		

########################################################################


##################### jeu d'essai


########################################################################

if __name__ == '__main__':
	
	from optparse import OptionParser

	usage = """Implementation of the algorithm of learning a lexicon
			GILL (graph intersection lexicon learning)
           """
	
	parser = OptionParser(usage=usage)
	parser.add_option("--k", dest="k", default=10, help='Nb (maximum) de candidates after the scoring function. Default=10')
	parser.add_option("--threshold", dest="threshold", default=0.4, help='final scores higher than the threshold will enter into the entries. Default=0.4')

	(opts,args)=parser.parse_args()
	k = int(opts.k)
	threshold = float(opts.threshold)

	path_utte = './one-to-one-mapping-data/utterance-*.txt'
	path_meaning= './one-to-one-mapping-data/meaning-*.txt'
	
	# to re-structure data into utterance_meaning pair
	app1 = InitialData(path_utte,path_meaning)
	app1.makePair()
	dataPairs = app1.pairs

	# create a GILL model 
	app2 = Gill(dataPairs, k=k)
	app2.mainAlgo(threshold=threshold)

	newfile = open("lexicon_test3.txt", 'w')
	for key in app2.lexicon_idf:
		newfile.write(str(key)+"\n")
		newfile.write(str(app2.lexicon_idf[key])+"\n")
		newfile.write(u"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"+"\n")
	newfile.close()



























