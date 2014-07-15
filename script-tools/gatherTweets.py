# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:30:55 2014

@author: balikasg
"""
from twitter import *
import sys
import time
import random

fin = open("../questions-words.txt", "r")
fou = open("/local/dias/collectedTweets.txt", "a")
hs = set()
for l in fin:
    if l[0] == ':':
            continue
    for word in l.split(" "):
            hs.add(word.strip())
fin.close()

print "Gathering tweets for %d words" % len(hs)


#yourKeyword = "Music"  # the spaces are like AND <-- tweets with both jobs and statistics will be returned, in case you want OR use commas e.g. "jobs,statistics"

tweetSet = set()
nbKw = len(hs)
kwNum = 0
total_tweets = 0
PERSUB = 1000
MAXSEC = 300

wordlist = []
for x in hs:
    wordlist.append(x)

random.shuffle(wordlist)
#We need the authentication parameters of your application. Register an application at https://dev.twitter.com/
auth = OAuth(consumer_key='g4BYmg5ba04Sv54ED37QT4SSc',
             consumer_secret='2p26W2zht3AFiG7adlPCJv0A6urEPMTkSybMg442T8h8Yg0Kvh',
             token='109399833-ujMwysmsdCtY6a4NVL8jrQBwMy8YIFKVw0QB8NwL',
             token_secret='AlXJI3cRPMtFE31iITRoZLZ5HKCtTCtkMvrPSbLhIq8OW')    # authentication parameters of your application
ts = TwitterStream(auth=auth)  # Initialize the twitter srteam
for kw in wordlist:
    kwNum += 1
    tweetcount = 0
    iterator = ts.statuses.filter(track=kw, language="en")  # Start gathering tweets
    ur, cnt = [], 0
    now = time.time()
    for item in iterator:  # Do something with the tweets, HEre we jus save them in the memory and we count them.
        tweet = item["text"].encode("utf-8")

        if tweet not in tweetSet:
            tweetSet.add(tweet)
            fou.write(tweet+"\n")
            tweetcount += 1
            total_tweets += 1
            sys.stdout.write("\r Keyword: %s (%d/%d - %d%%) - Number of collected tweets %d (%d%%) - Total collected: %d" %
                            (kw, kwNum, nbKw, kwNum/float(nbKw)*100, tweetcount, tweetcount/float(PERSUB)*100, total_tweets))

            sys.stdout.flush()
            later = time.time()
        if tweetcount >= PERSUB or int(later-now) > MAXSEC:
            fou.flush()
            break


fout.close()
