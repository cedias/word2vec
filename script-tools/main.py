#main.py
import tooling as tools


vectors = tools.toDicVec("vec.txt");
wordDic = tools.wordCorpusSum("vec.txt","text8",3,True,200)

print "wordDic has %d words in it" % (len(wordDic))
"""print "Starting kmean:"

a = tools.kmean(wordDic)
clusters = zip(a,wordDic.keys())
clusters = sorted(clusters)

f = open("/tmp/Clusts","w")

for clust,word in clusters:
	line = str(clust) + " => " + word + "\n"
	f.write(line)

f.close()"""

tools.writeWordDicBin(wordDic, 200)


