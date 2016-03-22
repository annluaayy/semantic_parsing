#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import os, glob
import re
import operator

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


class IBM_1:
	
	"""
	IBM Model 1 for modeling the alignements between words(unigrams for now) and meaning representations

	Here, we take the utterance as the source language and the mr(meaning representation) as the target language in our translation model
	"""
	def __init__(self, dataPairs):
		"""
		dataPairs is a list of pair with utterance and its corresponding set of meanings in the initial SONY-CNRS data;

		lexicon is a list for stocking the entries after the convergence of EM algorithm;

		alignments is a list for stocking all the possible pair of word_meaningRepresentation;

		unigramsLex is a list for stocking all the words(unigrams) apprearing in the utterances;

		mrLs is a list for stocking all the MR(meaning representataions) in data;
	
		dic_transProba is a dictionary for stocking the translation probabilites in the algo EM.
		"""
		self.dataPairs = dataPairs

		# create a lexicon for stocking the entries later
		self.lexicon = []

		# initiate a list of tuples of aligned utterances and MR
		self.alignments = []

		# initiate the unigrams in all the utterances
		self.unigramsLex = []

		# initiate a list for all the MR representations
		self.mrLs = []

		# initiate the translation probability for all the pairs
		self. dic_transProba = {}

		# create a dictionary for stocking the model probabilities later
		self.dic_modelProba = {}

	
	def lemmatizer(self, utterance):
		"""
		to lemmatize words in an utterance
		"""
		lmtzr = WordNetLemmatizer()
		wordsLs = utterance.split()
		lemmas = [lmtzr.lemmatize(x) for x in wordsLs]
		return lemmas

	def initLex(self):
		"""
		to extact all the unigrams and MR apprearing in the utterances and meaning sets respectively
		"""	
		for pair in self.dataPairs:
			lemmasLs = self.lemmatizer(pair[0])
			# stock unigram lemmas and mr in two list separately unigramslex list and mrLs list
			self.unigramsLex.extend(lemmasLs)
			self.mrLs.extend(pair[1])

		# for avoiding the repetition of unigrams or mrs in the lists, dedouble them
		list(set(self.unigramsLex))
		list(set(self.mrLs))	

	def mainAlgo(self):
		"""
		the main program for the implementation of IBM tranlsation model 1
		"""

		## initiate all the t(e|f) uniformly, each t(e|f) = 1/s where s is the number of translated lexicons(here MR)
		# recall the function initLex
		self.initLex()
		# initiate alignments
		for pair in self.dataPairs:
			lemmasLs = self.lemmatizer(pair[0])
			self.initAlign(lemmasLs,pair[1])
		

		initialFile = open("initialFile.txt",'w')
		# for avoiding the repetition, dedouble the alignment pairs
		list(set(self.alignments))
		# calculate the initial t(e|f) and stock them in a dictionary of translation probability:dic_transProba
		for alignedPair in self.alignments:
			self.dic_transProba[alignedPair] = float(1)/float(len(self.mrLs))
			initialFile.write(str(alignedPair)+"	"+str(self.dic_transProba[alignedPair])+"\n")
			initialFile.write(u"                                                          "+"\n")
		initialFile.close()
		
	
		i=0	

		while i < 10:

			# run EM-like algorithm
			self.emAlgo()
			# re-order the dictionary of model probability, sorted function turns dictionary into a list of tuples
			#sorted_dic = sorted(self.dic_transProba.items(), key=operator.itemgetter(1))

			
			filename = "iterate"+str(i)+".txt"
			newfile = open(filename,'w')
			for key in self.dic_transProba:
				newfile.write(str(key)+"	"+str(self.dic_transProba[key])+"\n")
				newfile.write("                "+"\n")
			newfile.close()
			i+=1	
			
			# threshold 0.5 for deciding to put entry into the lexicon
			#if sorted_dic[-1][1]>0.5:
				#print sorted_dic[-1]
				#break
				#self.lexicon.append((sorted_dic[-1][0][0],sorted_dic[-1][0][1])
				#self.unigramsLex.remove(sorted_dic[-1][0][0])
				#self.alignments.remove(sorted_dic[-1][0])
				#sorted_dic.remove(sorted_dic[-1])
				
		
	def initAlign(self, lemmasLs, mrLs):
		"""
		Given lemmas in an utterance and its corresponding set of mr(meaning representations),
		make the initial alignments as if all alignments are equally likely
		"""
		for lemma in lemmasLs:
			# initialize the alignments for this lemma with all concepts in this MR set
			alignLs = [(lemma,x) for x in mrLs]
			# stock the alignments concerning this lemma in the list of initiated alignments
			self.alignments.extend(alignLs)	

	
	def emAlgo(self):
		"""
		iterative process of the main program by using EM-like algorithm
		"""
		# initiate all the count(e|f) values to 0
		dic_count_e_f = {}
		for element in self.alignments:
			dic_count_e_f[element] = 0.0
		
		# initiate total(f) to 0 for all f
		dic_total_f = {}
		for word in self.unigramsLex:
			dic_total_f[word] = 0.0
		
		for pair in self.dataPairs:	
		
			wordsLs = self.lemmatizer(pair[0])
			# compute normalization (?? normalize p(a,f|e) values to yield p(a|f,e) values)
			dic_totalMr={}
			for mr in pair[1]:
				total_mr = 0.0
				for word in wordsLs:
					if (word,mr) in self.dic_transProba:
						total_mr += self.dic_transProba[(word,mr)]
				dic_totalMr[mr] = total_mr		

			# collect counts
			for mr in pair[1]:
				for word in wordsLs:
					if (word,mr) in self.dic_transProba:
						# count of an MR is a tranlsation of a word
						count_mr_word = float(self.dic_transProba[(word,mr)])/float(dic_totalMr[mr])
						dic_count_e_f[(word,mr)] += count_mr_word

						# count of total(f)
						dic_total_f[word] += count_mr_word
		
		self.dic_transProba = {}	
		# estimate the model or the probabilites
		for word in self.unigramsLex:
			for mr in self.mrLs:
				if (word,mr) in self.alignments:
					self.dic_transProba[(word,mr)] = float(dic_count_e_f[(word,mr)])/float(dic_total_f[word])	

		
		 



#####################################################################


############ jeu d'essai

#####################################################################

if __name__ == '__main__':			
				
	path_utte = './one-to-one-mapping-data/utterance-*.txt'
	path_meaning= './one-to-one-mapping-data/meaning-*.txt'
	
	# to re-structure data into utterance_meaning pair
	app1 = InitialData(path_utte,path_meaning)
	app1.makePair()
	dataPairs = app1.pairs

	# to train the data in the IBM 1 translation model, for output of better (word, meaning) pairs
	ibm_app = IBM_1(dataPairs)				
	ibm_app.mainAlgo()
								
			
			
	
		

	
		
