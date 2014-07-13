#readfreebase.py
import gzip
import sys

f = gzip.open("/local/dias/freebase.gz", "r")
savefile = open("/local/dias/descriptions.txt", "w")
i = 0

for line in (line for line in f if line.split("\t")[1] == "<http://rdf.freebase.com/ns/common.topic.description>"):
    obj = line.split("\t")[2]
    if(obj[-3:len(obj)] == "@en"):
        desc = obj[:-3]
        savefile.write(obj[:-3]+"\n")
        i += 1
        sys.stdout.write("\r wrote %d descriptions  " % (i))
        sys.stdout.flush()


f.close
savefile.close
