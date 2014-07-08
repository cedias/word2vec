#test-ngram-w2vec.py
import subprocess as sp

sizes = range(300, 400, 50)
negatives = range(0, 15, 5)
ngrams = range(3, 6, 1)
cbows = [0, 1]
positions = [0, 1, 2]
windows = [10]
min_count = [0, 5, 10]
sample = ["0", "1e-4"]

cpt = 1
logFile = open("results.txt", "w")

for size in sizes:
    for negative in negatives:
        for ngram in ngrams:
            for mini in min_count:
                for cbow in cbows:
                    for position in positions:
                        for window in windows:
                            for samp in sample:
                                print "iteration %d on 649" % (cpt)
                                outputFile = "/local/dias/v_%s_%s_%s_%s_%s_%s_%s_%s.bin" % \
                                    (str(size), str(negative), str(ngram), str(cbow), str(mini), str(position), str(window), samp)
                                argsLine = "./word2gram -train text8 -output %s -threads 12 -negative %s -window %s -cbow %s -sample %s -binary 1 -size %s -min-count %s -pos %s -ngram %s" % \
                                    (outputFile, str(negative), str(window), str(cbow), samp, str(size), str(mini), str(position), str(ngram))
                                logFile.write(argsLine+"\n");
                                sp.call(args=argsLine, shell=True, stdout=logFile)
                                cpt = cpt + 1
