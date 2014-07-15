import struct
import sys


def bin2dic(filename):

    f = open(filename, "rb")
    dic = {}

    try:
        line = f.readline()
        infoline = line.split(" ")
        vocab_size = int(infoline[0])
        vector_size = int(infoline[1])
        ngram_size = int(infoline[2])
        hashbang = int(infoline[3])
        position = int(infoline[4])

        line = f.readline()

        while line != "":
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


res = bin2dic(sys.argv[1])
