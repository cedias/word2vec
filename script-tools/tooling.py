import numpy as np
import struct
import sys
from num2words import num2words
import re


def toDicVec(filename):
    """
    Creates a dic with 'word':np_array from text vector files.
    """
    dic = {}
    first = True
    vecfile = open(filename, "r")
    vecfile.readline()  # 1st line = useless
    for line in vecfile:
        for word in line.split(" "):
            if(first):
                key = word
                dic[key] = []
                first = False
            else:
                dic[key].append(word)
        dic[key].pop()
        first = True

    for key in dic:
        dic[key] = np.array(dic[key], float)

    return dic


def wordCorpusSum(filename, corpus, gramsize, hashbangs, vsize):
    """
    creates a word dictionnary of word in corpus with vectors being the sumation of the ngrams (overlapping) in file filename
    """
    dic = toDicVec(filename)
    wordDic = {}
    errorCpt = 0

    cfile = open(corpus, "r")

    for line in cfile:
        for word in line.split(" "):
            if word in wordDic:
                continue

            key = word

            if(hashbangs):
                word = '#'+word+'#'

            start = 0
            end = gramsize
            vec = np.zeros(vsize)

            while end <= len(word):
                try:
                    vec = np.add(vec, dic[word[start:end]])
                except:
                    #print "the %d-gram %s from word %s is not in the dictionnary "%(gramsize,word[start:end],word)
                    end = end+1
                    start = start+1
                    errorCpt += 1
                    continue

                end = end+1
                start = start+1

            wordDic[key] = vec

    print "%d grams where missing from vocabulary" % (errorCpt)
    return wordDic


def bin2dic(filename,wordlist=[]):
    """
    transform binary file from word2vec in a dictionnary (with only words from wordlist if not empty):
    returns: return {"dic" "vocab_size" "vector_size" "ngram_size" "hashbang" "position"}

    """

    f = open(filename, "rb")
    dic = {}
    gram = ""

    try:
        line = f.readline()
        infoline = line.split(" ")
        vocab_size = int(infoline[0])
        vector_size = int(infoline[1])
        try:
            ngram_size = int(infoline[2])
            hashbang = int(infoline[3])
            position = int(infoline[4])
        except:
            ngram_size = 0
            hashbang = 0
            position = 0

        line = f.readline()

        while line != "":
            if len(gram) > 0 and wordlist != [] and gram not in wordlist:
                del dic[gram]
            else:
                print gram

            fullline = line.split(" ", 1)
            gram = fullline[0]
            dic[gram] = []

            if len(fullline) < 2:
                nextline = f.readline()
                fullline.append(nextline)

            bs = bytearray(fullline[1])
            i = 0

            while True:

                while len(bs) < vector_size*4+1:
                    nextline = bytearray(f.readline())
                    for b in nextline:
                        bs.append(b)

                num = struct.unpack('f', bs[i:i+4])
                dic[gram].append(num[0])
                i += 4

                if i >= len(bs)-1:
                    break

            if len(dic[gram]) != vector_size:
                print "error on vec gram: %s lenght %d instead of %s" % (gram, len(dic[gram]), vector_size)

            line = f.readline()
    finally:
        f.close()

    return {"dic": dic, "vocab_size": vocab_size, "vector_size": vector_size, "ngram_size": ngram_size, "hashbang": hashbang, "position": position}



def cleanfile(filename, output, stats=True):
    """
    cleans a corpus from punctation.
    """
    if stats:
        num_lines = sum(1 for line in open(filename))
        i = 0

    fin = open(filename, "r")
    fou = open(output, "w")
    delChars = [".", "?", "!", ",", "\"", "(", ")", "{", "}", ":", ";", "#", "*", "/", "\\", "'", "-"]

    for line in fin:
        if len(line) > 1:
            for char in delChars:
                line = line.replace(char, "")
            line = line.replace("&", "and")

            nums = re.findall(r'\d+', line)
            for num in nums:
                o = num
                n = str(" "+num2words(int(num))+" ")

                line = line.replace(o, n)
            fou.write(line)

        if stats:
            i += 1
            percent = i/float(num_lines)*100
            sys.stdout.write("\r (%d/%d)  %d%%  " % (i, num_lines, percent))
            sys.stdout.flush()
    print "\n"