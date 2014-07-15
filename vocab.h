#ifndef VOCAB
#define VOCAB
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DEBUG_MODE 2

struct vocab_word {
 	long long cn; //times of occurence in train file
 	int *point;
 	char *word, *code, codelen;
};

typedef struct vocabulary_struct
{
	int vocab_hash_size;
	unsigned long long int train_words;
	long long vocab_max_size;
	long long vocab_size;
	int* vocab_hash;
	struct vocab_word *vocab;

} vocabulary;

/*Inits a vocabulary */
vocabulary *InitVocabulary(int vocab_hash_size, int vocab_max_size);
/*Reads a word from file descriptor fin*/
void ReadWord(char *word, FILE *fin);

/*Reads a word and adds #hashbangs# around it from file descriptor fin*/
void ReadWordHashbang(char *word, FILE *fin);

/* Returns hash value of a word*/
int GetWordHash(vocabulary* voc,char *word);

/*free the vocab structure*/ //TODO
void DestroyVocab(vocabulary* voc);

/* Returns position of a word in the vocabulary;
 if the word is not found, returns -1*/
int SearchVocab(vocabulary* voc,char *word);

/* Reads a word and returns its index in the vocabulary*/
int ReadWordIndex(vocabulary* voc,FILE *fin);

/* Adds a word to the vocabulary*/
int AddWordToVocab(vocabulary* voc,char *word);

/* Used for sorting by word counts */
int VocabCompare(const void *a, const void *b);

/* Sorts the vocabulary by frequency using word counts - EXCEPT 0 -*/
void SortVocab(vocabulary* voc, int min_count);

/* Reduces the vocabulary by removing infrequent tokens */
void ReduceVocab(vocabulary* voc,  int min_reduce);

/*Look if word already in vocab, 
if not add, if yes, increment. --- REDUCE VOCAB DEACTIVATED */
void searchAndAddToVocab(vocabulary* voc, char* word);

/*Create a vocab from train file - returns file size*/
long long LearnVocabFromTrainFile(vocabulary* voc, char* train_file,int min_count);

/*Create a vocab of ngram from train file returns file size*/
long long LearnNGramFromTrainFile(vocabulary* voc, char* train_file,int min_count, int ngram, int hashbang, int position, int overlap);

/*Saves vocab & Occurences*/
void SaveVocab(vocabulary* voc, char* save_vocab_file);

/*Reads a saved vocab file ------------ MIN COUNT DEACTIVATED*/
long long ReadVocab(vocabulary* voc, char* read_vocab_file, char* train_file, int min_count);

/* Create binary Huffman tree using the word counts
 Frequent words will have short uniqe binary codes*/
void CreateBinaryTree(vocabulary* voc);



#endif