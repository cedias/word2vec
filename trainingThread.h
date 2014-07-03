#ifndef TRAIN_THREADS
#define TRAIN_THREADS

#define MAX_SENTENCE_LENGTH 100
#define MAX_CODE_LENGTH 40

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "vocab.h"

typedef float real;

typedef struct threadParameters_struct {
    vocabulary *voc;
    real *syn0 , *syn1, *syn1neg, *expTable, *alpha,starting_alpha;
    real sample; 
    int threadNumber,num_threads,hs,negative,file_size,max_string,exp_table_size,ngram,max_exp,window, layer1_size,table_size;
    int *table;
    long long int *word_count_actual;
    char* train_file;
} threadParameters;

threadParameters * CreateParametersStruct(vocabulary* voc,
	real *syn0,
	real *syn1,
	real *syn1neg,
	real *expTable,
	real *alpha,
	real starting_alpha,
	real sample,
	long long int *word_count_actual,
	int *table,
	int threadNumber,
	int num_threads,
	int file_size,
	int max_string,
	int exp_table_size,
	int ngram,
	int layer1_size,
	int window,
	int max_exp,
	int hs,
	int negative,
	int table_size,
	char* train_file
	); 

void *TrainCBOWModelThread(void *arg);
void *TrainSKIPModelThread(void *arg);


#endif