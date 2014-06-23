#test-ngram-w2vec.py
import subprocess as sp
import numpy as np

sizes = range(200,350,50)
samples = ["0","1e-5"]
negatives = range(0,10,5)
alphas = np.arange(0.025,0.060,0.015)
ngrams = range(2,5,1)
hashbs = [0,1]
cbows = [0,1]
hsE = [0,1]


cpt = 1
logFile = open("results.txt" , "w")
lofFile2 = open("parameters.txt", "w")
lofFile2.write("size\tsample\tnegative\talpha\tngram\thashbang\tcbow\ths\n");
for size in sizes:
	for sample in samples:
		for negative in negatives:
			for hs in hsE:
				if negative == 0 and hs == 0:
					continue;
				for alpha in alphas:
					for ngram in ngrams:
						for hashb in hashbs:
							for cbow in cbows:
									print "iteration %d on 649" % (cpt)
									argsLine= "./testNgrams.sh %s %s %s %s %s %s %s %s" % (str(size),str(sample),str(negative),str(alpha),str(ngram),str(hashb),str(cbow),str(hs))
									argu= "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (str(size),str(sample),str(negative),str(alpha),str(ngram),str(hashb),str(cbow),str(hs))
									lofFile2.write(argu);
									sp.call(args=argsLine,shell=True,stdout=logFile)
									cpt = cpt+1
	

