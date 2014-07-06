CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -march=native -Wall  -Wno-unused-result -funroll-loops -Ofast
#

all: word2gram word2vec word2phrase distance word-analogy compute-accuracy compute-accuracy-syntax cleano

vocab.o : vocab.c vocab.h
	$(CC) -c vocab.c
ngram_tools.o : ngram_tools.c ngram_tools.h
	$(CC) -c ngram_tools.c
trainingThread.o : trainingThread.c trainingThread.h
	$(CC) -c trainingThread.c
word2gram : word2gram.c vocab.o ngram_tools.o trainingThread.o
	$(CC) word2gram.c vocab.o ngram_tools.o trainingThread.o -o word2gram $(CFLAGS)
word2vec : word2vec.c vocab.o ngram_tools.o trainingThread.o
	$(CC) word2vec.c vocab.o ngram_tools.o trainingThread.o -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
compute-accuracy-syntax : compute-accuracy-syntax.c
	$(CC) compute-accuracy-syntax.c -o compute-accuracy-syntax $(CFLAGS)
	chmod +x *.sh

cleano: word2vec word2gram
	rm *.o

clean:
	rm -rf word2vec word2gram word2phrase distance word-analogy compute-accuracy compute-accuracy-syntax