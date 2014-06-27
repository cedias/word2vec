import numpy as np
from sklearn import cluster


def toDicVec(filename):
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
        dic[key] = np.array(dic[key],float)

    return dic


def wordCorpusSum(filename,corpus,gramsize,hashbangs,vsize):
    dic = toDicVec(filename)
    wordDic = {}
    errorCpt = 0;

    cfile = open(corpus,"r")

    for line in cfile:
        for word in line.split(" "):
            
            if(wordDic.has_key(word)):
                continue

            key = word


            if(hashbangs):
                word = '#'+word+'#'

            start=0
            end=gramsize
            vec = np.zeros(vsize)
            while end <= len(word):
                
                try:
                    vec = np.add(vec,dic[word[start:end]])
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


def kmean(wordDic):
    km = cluster.KMeans(n_clusters=100, init='k-means++', n_init=10, max_iter=30, tol=0.0001, precompute_distances=True, verbose=1, n_jobs=8)
    km.fit(wordDic.values())
    return km.predict(wordDic.values())


def writeWordDicBin(wordDic,size):
    f = open("/tmp/vec.bin","wb")
    string = str(len(wordDic))+" "+str(size) 
    f.write(bytearray(string))

    for word in wordDic.keys():
        f.write(bytearray(word+" "))
        for num in wordDic[word]:
            f.write(bytearray(str(num)))
        f.write(bytearray("\n"))
    f.close()