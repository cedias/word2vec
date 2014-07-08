from os import listdir
from os import remove
import subprocess as sp
import time

fparams = open("acc.txt", "a")
while True:
    for vec in listdir("."):
        if(vec[-3:len(vec)] == "bin"):
            argsLine = "../compute-accuracy %s 30000 0 < ../questions-words.txt" % (vec)
            sp.call(args=argsLine, shell=True, stdout=fparams)
            remove(vec)

    time.sleep(60)
