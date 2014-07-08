//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


/**
* For more details see:
*
* http://arxiv.org/abs/1402.3722
* http://yinwenpeng.wordpress.com/2013/12/18/word2vec-gradient-calculation/
* http://yinwenpeng.wordpress.com/2013/09/26/hierarchical-softmax-in-neural-network-language-model/
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include "vocab.h"
#include "trainingThread.h"
#include "ngram_tools.h"

#define MAX_EXP 6
#define MAX_STRING 100

typedef float real;     // Precision of float numbers

int EXP_TABLE_SIZE = 1000;

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 0, num_threads = 1, min_reduce = 0, ngram = 3, hashbang = 1, group_vec = 0;
int layer1_size = 100, position = 1;

long long word_count_actual = 0, file_size = 0, classes = 0;

real alpha = 0.025, starting_alpha, sample = 0;
//syn0 = vectors table
real *syn0, *syn1, *syn1neg, *expTable;

clock_t start;

int hs = 1, negative = 0;

const int table_size = 1e8;

int *table;

void InitUnigramTable(vocabulary * voc) {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));


	for (a = 0; a < voc->vocab_size; a++)
		train_words_pow += pow(voc->vocab[a].cn, power); //occurences^power

	i = 0;
	d1 = pow(voc->vocab[i].cn, power) / (real)train_words_pow; //normalize

	for (a = 0; a < table_size; a++) {

		table[a] = i;

		if (a / (real)table_size > d1) {
		  i++;
		  d1 += pow(voc->vocab[i].cn, power) / (real)train_words_pow;
		}

		if (i >= voc->vocab_size)
			i = voc->vocab_size - 1;
		}
}

void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}

void InitNet(vocabulary * voc) {
	long long a, b;
	a = posix_memalign((void **)&syn0, 128, (long long)voc->vocab_size * layer1_size * sizeof(real));

	if (syn0 == NULL) {
		printf("Memory allocation failed\n"); 
		exit(1);
	}

	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)voc->vocab_size * layer1_size * sizeof(real));

		if (syn1 == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}

		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < voc->vocab_size; a++)
				 syn1[a * layer1_size + b] = 0;
	}

	if (negative>0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)voc->vocab_size * layer1_size * sizeof(real));

		if (syn1neg == NULL){
			printf("Memory allocation failed\n");
			exit(1);
		}

		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < voc->vocab_size; a++)
		 		syn1neg[a * layer1_size + b] = 0;
	}

	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < voc->vocab_size; a++)
			syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

	CreateBinaryTree(voc);
}

void TrainModel(vocabulary* voc) {
	long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	starting_alpha = alpha;
	InitNet(voc);

	if (negative > 0)
		InitUnigramTable(voc);

	start = clock();

	threadParameters* params; 

	for (a = 0; a < num_threads; a++){
		params = CreateParametersStruct(
			voc,
			syn0,
			syn1,
			syn1neg,
			expTable,
			(&alpha),
			starting_alpha,
			sample,
			(&word_count_actual),
			table,
			a,
			num_threads,
			file_size,
			MAX_STRING,
			EXP_TABLE_SIZE,
			ngram,
			layer1_size,
			window,
			MAX_EXP,
			hs,
			negative,
			table_size,
			position,
			train_file
			);

		/*NB: The parameters struct are freed by each thread.*/

		if(cbow)
			pthread_create(&pt[a], NULL, TrainCBOWModelThreadGram, (void *)params);
		else
			pthread_create(&pt[a], NULL, TrainSKIPModelThreadGram, (void *)params);
	}

	for (a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);

	if(debug_mode > 0)
		printf("Training Ended !\n");


	free(pt);

}


int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {

			if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
			}

		return a;
		}

	return -1;
}

int main(int argc, char **argv) {
	int i;

	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
		printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 0\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
		printf("\t-ngram <int> (default 0 - use words) \n");
		printf("\t\tUse N-GRAM model instead of words to train vectors \n");
		printf("\t-hashbang <0-1> (default 0)\n");
		printf("\t\tUse hashbang on n-grams - i.e #good# -> #go,goo,ood,od#\n");
		printf("\t-group <0-5> (default 0)\n");
		printf("\t\tHow word vectors are computed with n-grams - 0:Sum (default); 1:Mean; 2:Min; 3:Max; 4:Trunc; 5:FreqSum\n");
		printf("\t-pos <0-1-2> (default 0) - 1: #good# -> #g go- -oo- -od d# 2: -> #g 01-go 02-oo 03-od d# \n");
		printf("\t\tAdds position indication to ngrams\n");
		
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
		return 0;
	}

	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;

	if ((i = ArgPos((char *)"-size", argc, argv)) > 0)layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos ((char *) "-ngram", argc, argv)) > 0 ) ngram = atoi(argv[i + 1]);
	if ((i = ArgPos ((char *) "-hashbang", argc, argv)) > 0 ) hashbang = atoi(argv[i + 1]);
	if ((i = ArgPos ((char *) "-group", argc, argv)) > 0 ) group_vec = atoi(argv[i + 1]);
	if ((i = ArgPos ((char *) "-pos", argc, argv)) > 0 ) position = atoi(argv[i + 1]);
	
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));

	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	/**
	Fixed starting Parameters:
	**/
	int vocab_hash_size =  3000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
	int vocab_max_size = 1000;


	//1: init vocabulary
	vocabulary* vocab = InitVocabulary(vocab_hash_size,vocab_max_size);

	//2: load vocab
	file_size = LearnNGramFromTrainFile(vocab,train_file,min_count,ngram,hashbang,position);

	if (output_file[0] == 0) //nowhere to output => quit
		return 0;

	//3: train_model
	TrainModel(vocab);
	
	//4: make word vectors
	printf("Creating word vectors.\n");
	gramVocToWordVec(vocab,syn0,MAX_STRING,layer1_size,ngram,hashbang,group_vec,binary,position,train_file,output_file);

	free(expTable);
	DestroyNet();
	DestroyVocab(vocab);

	return 0;
}