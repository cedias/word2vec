import struct


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
            bs = bytearray(fullline[1])
            i = 0
            print "nb bytes: %d" % len(bs)
            print "g: %s" %gram
            while i < len(bs)-1:

                if len(bs) != vector_size*4+1:
                    print "not enough bytes, \\n inside"
                    nextline = bytearray(f.readline())
                    for b in nextline:
                        bs.append(b)
                    print "new nb bytes: %d" % len(bs)

                num = struct.unpack('f', bs[i:i+4])
                dic[gram].append(num[0])
                i += 4

            print dic[gram]

            line = f.readline()
    finally:
        f.close()

    return {"dic": dic, "vocab_size": vocab_size, "vector_size": vector_size, "ngram_size": ngram_size, "hashbang": hashbang, "position": position}


bin2dic("../vecTest.bin")
