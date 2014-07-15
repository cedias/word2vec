import sys


def getTopicList():
    fin = open("/home/dias/word2vec/questions-words.txt", "r")
    hs = set()
    for l in fin:
        if l[0] == ':':
                continue
        for word in l.split(" "):
                hs.add(word.strip())
    fin.close()
    return list(hs)

num_lines = sum(1 for line in open("/local/dias/dataset/cleaned_descriptions.txt"))

fin = open("/local/dias/dataset/cleaned_descriptions.txt", "r")
fout = open("/local/dias/intopic.txt", "w")

topics = getTopicList()
i = 0
in_topic = 0
allLines = 0

for line in fin:
    allLines += 1
    for word in line.split(" "):
        if word in topics:
            in_topic += 1
            fout.write(line)
            break

    i += 1
    percent = i/float(num_lines)*100
    sys.stdout.write("\r (%d/%d)  %d%%  - %d in topics on %d (%d%%)" % (i, num_lines, percent, in_topic, allLines, in_topic/float(allLines)*100))
    sys.stdout.flush()
