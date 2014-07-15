import csv
import sys
import re
from num2words import num2words

def csvtotweetlist(filename):
    with open(filename, "rb") as csvfile:
        freader = csv.reader(csvfile, delimiter=',')
        for row in freader:
            yield row[len(row)-1]
        csvfile.close()


def csvtweets2corpus(filename, output):
    fout = open(output,"w")
    num_lines = sum(1 for line in open(filename))
    i = 0

    delChars = [".", "?", "!", ",", "\"", "(", ")", "{", "}", ":", ";", "#", "*", "/", "\\", "'", "-"]
    username = re.compile("@(\w)+")
    url = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")
    #hashtag = re.compile("[^\/]{1}#(\w)+")
    escapeChars = re.compile("&.+;")

    for tweet in csvtotweetlist(filename):
        tweet = re.sub(escapeChars, "", tweet)
        tweet = re.sub(username, "", tweet)
        tweet = re.sub(url, "", tweet)

        for char in delChars:
            tweet = tweet.replace(char, "")

        nums = re.findall(r'\d+', tweet)
        for num in nums:
            o = num
            n = str(" "+num2words(int(num))+" ")
            tweet = tweet.replace(o, n)

        fout.write(tweet+"\n")

        i += 1
        percent = i/float(num_lines)*100
        sys.stdout.write("\r (%d/%d)  %d%%  " % (i, num_lines, percent))
        sys.stdout.flush()

    print "\n"


csvtweets2corpus(sys.argv[1],sys.argv[2])
