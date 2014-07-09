import sys

try:
    filename = sys.argv[1]
except:
    print "missing filename argument"
    exit()

fin = open(filename, "r")
fout = open("clean.txt", "w")

for line in fin:
    print "hello"
    print line

fin.close
fout.close
