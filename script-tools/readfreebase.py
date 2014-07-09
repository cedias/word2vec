#readfreebase.py
import gzip


f = gzip.open("/local/dias/freebase.gz", "r")
savefile = open("/local/dias/descriptions.txt", "w")
i = 0

for line in (line for line in f if line.split("\t")[1] == "<http://rdf.freebase.com/ns/common.topic.description>"):
    obj = line.split("\t")[2]
    if(obj[-3:len(obj)] == "@en"):
    	desc = obj[:-3]
        savefile.write(obj[:-3])
        i += 1
        print "wrote %d description" % (i)


f.close
savefile.close
