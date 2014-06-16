import numpy as np


def toDicVec(filename):
	dic = {}
	first = True
	vecfile = open(filename,"r")
	vecfile.readline(); #1st line = useless
	for line in vecfile:
		for word in line.split(" "):
			if(first):
				key = word
				dic[key]=[]
				first = False
			else:
				dic[key].append(word)
		dic[key].pop()
		first = True

	return dic


def wordCorpusSum(filename,corpus,gramsize,hashbangs):
	dic = toDicVec(filename)
	wordDic = {}

	cfile = open(corpus,"r")

	for line in cfile:
		for word in line.split(" "):
			key = word

			if(hashbangs):
				word = '#'+word+'#'

			start=0
			end=gramsize-1





