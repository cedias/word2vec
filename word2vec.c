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

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40


const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
 	long long cn; //times of occurence in train file
 	int *point;
 	char *word, *code, codelen;
}typedef vword;

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1, ngram = 0, hashbang = 0, group_vec = 0;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
//syn0 = vectors table
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));

	for (a = 0; a < vocab_size; a++)
		train_words_pow += pow(vocab[a].cn, power); //occurences^power

	i = 0;
	d1 = pow(vocab[i].cn, power) / (real)train_words_pow; //normalize

	for (a = 0; a < table_size; a++) {

		table[a] = i;

		if (a / (real)table_size > d1) {
		  i++;
		  d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
		}

		if (i >= vocab_size)
			i = vocab_size - 1;
		}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, character;
	
	while (!feof(fin)) {
		character = fgetc(fin);



		if (character == 13) //Carriage Return
			continue;

		if ((character == ' ') || (character == '\t') || (character == '\n')) {
			
			if (a > 0) {

		    	if (character == '\n')
		    		ungetc(character, fin); //we don't want the new line char.
		    break;
		  	}

		 	if (character == '\n') { 
			    strcpy(word, (char *)"</s>");  //newline become </s> in corpus
			    return;
		  	}
		 	else
		  		continue;
		}

		word[a] = character;
		a++;

		if (a >= MAX_STRING - 1)
			a--;   // Truncate too long words
	}

	word[a] = '\0';

	if(hashbang > 0)
	{

		a = strlen(word); //'\0'
		word[a] = '#';
		a++;
		word[a] = '\0';
		a++;


 		while(a>0)
 		{
 			word[a] = word[a-1];
 			a--;
 		}

 		word[0] ='#';
	}


	return;
}

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;

	for (a = 0; a < strlen(word); a++) 
		hash = hash * 257 + word[a];

	hash = hash % vocab_hash_size;
	return hash;
}

void DestroyVocab() {
  int a;

  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      free(vocab[a].word);
    }
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab[vocab_size].word);
  free(vocab);
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

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);

	while (1) {
		if (vocab_hash[hash] == -1)
			return -1;

		if (!strcmp(word, vocab[vocab_hash[hash]].word))
			return vocab_hash[hash];

		hash = (hash + 1) % vocab_hash_size;
	}

	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);

	if (feof(fin)) 
		return -1;

	return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {

	unsigned int hash, length = strlen(word) + 1;

	if (length > MAX_STRING)
		length = MAX_STRING;

		
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));

	strcpy(vocab[vocab_size].word, word);

	vocab[vocab_size].cn = 0;
	vocab_size++;

	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);

	while (vocab_hash[hash] != -1)
		hash = (hash + 1) % vocab_hash_size;

	vocab_hash[hash] = vocab_size - 1;

	return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;

	if(debug_mode > 2)
		printf("Sorting Vocab...\n");

	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);

	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;

	size = vocab_size;
	train_words = 0;

	for (a = 1; a < size; a++) {
	// Words occuring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) {
			vocab_size--;
			free(vocab[vocab_size].word); 
			//free(vocab[a].word);
			vocab[a].word = NULL;
		}
		else {
		// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word);

			while (vocab_hash[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;

			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}


	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));

	// Allocate memory for the binary tree construction
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}

}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {

	int a, b = 0;
	unsigned int hash;

	for (a = 0; a < vocab_size; a++){
		if (vocab[a].cn > min_reduce) {

			vocab[b].cn = vocab[a].cn;
			vocab[b].word = vocab[a].word;
			b++;

		} else
			free(vocab[a].word);
	}

	vocab_size = b;

	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;

	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);

		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;

		vocab_hash[hash] = a;
	}

	fflush(stdout);
	min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

	for (a = 0; a < vocab_size; a++)
		count[a] = vocab[a].cn;

	for (a = vocab_size; a < vocab_size * 2; a++) //sets rest of count array to 1e15
		count[a] = 1e15;

	pos1 = vocab_size - 1; //end of word occurences
	pos2 = vocab_size; //start of other end

	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < vocab_size - 1; a++) { //vocab is already sorted by frequency
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {

			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}

		} else {
			min1i = pos2;
			pos2++;
		}

		if (pos1 >= 0) {

			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}

		} else {
			min2i = pos2;
			pos2++;
		}

		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;

		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];

			if (b == vocab_size * 2 - 2)
				break;
		}

		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;

		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

//Look if word already in vocab, if not add, if yes, increment.
void searchAndAddToVocab(char* word){
	long long a,i;
	i = SearchVocab(word);

		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		} else
			vocab[i].cn++;

		if (vocab_size > vocab_hash_size * 0.7)
			ReduceVocab();
}

void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	int i,start,end,lenWord;

	char gram[ngram+1];

	for (i = 0; i < vocab_hash_size; i++) //init vocab hashtable
		vocab_hash[i] = -1;

	fin = fopen(train_file, "rb");

	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	vocab_size = 0;
	AddWordToVocab((char *)"</s>");

	while (1) {
		ReadWord(word, fin);

		if(ngram > 0) //learn ngrams instead of words
		{
			lenWord = strlen(word);

			if(lenWord<=ngram){ //word smaller or equal to ngram var.
				searchAndAddToVocab(word);
				//printf("smaller\n");

				if (feof(fin))
					break;
				else
					continue;
			}

 			start = 0;
			end = ngram-1;
			i=0;
			//printf("%s\n",word );
		

			while(end<lenWord)
			{

				for (i = 0; i < ngram; i++)
				{
					gram[i] = word[start+i];
				}
				gram[ngram] = '\0';


				//printf("%s\n",gram );

				searchAndAddToVocab(gram);

				end++;
				start++;
			}
		}
		else
		{
			searchAndAddToVocab(word);
		}

		if (feof(fin))
			break;

		train_words++;

		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
	}

	SortVocab();

	if (debug_mode > 1) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}

	file_size = ftell(fin);
	fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");

  for (i = 0; i < vocab_size; i++)
  	fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);

  fclose(fo);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");

	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}

	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;

	vocab_size = 0;

	while (1) {
		ReadWord(word, fin);

		if (feof(fin))
			break;

		a = AddWordToVocab(word);
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		i++;
	}

	SortVocab();

	if (debug_mode > 1) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fin = fopen(train_file, "rb");

	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}

void InitNet() {
	long long a, b;
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));

	if (syn0 == NULL) {
		printf("Memory allocation failed\n"); 
		exit(1);
	}

	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));

		if (syn1 == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}

		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < vocab_size; a++)
				 syn1[a * layer1_size + b] = 0;
	}

	if (negative>0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));

		if (syn1neg == NULL){
			printf("Memory allocation failed\n");
			exit(1);
		}

		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < vocab_size; a++)
		 		syn1neg[a * layer1_size + b] = 0;
	}

	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

	CreateBinaryTree();
}

void *TrainModelThread(void *id) {
	long long a, b, d, i, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;

	char wordToGram[MAX_STRING];
	char gram[ngram+1];
	int start = 0;
	int end = ngram-1;
	int newWord = 1;
	int wordLength = 0;

	real *neu1 = (real *)calloc(layer1_size, sizeof(real)); //one vector
	real *neu1e = (real *)calloc(layer1_size, sizeof(real)); 
	FILE *fi = fopen(train_file, "rb");

	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

	while (1) {


		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;

			if ((debug_mode > 1)) {
				now=clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
				word_count_actual / (real)(train_words + 1) * 100,
				word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}

			alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));

			if (alpha < starting_alpha * 0.0001)
				alpha = starting_alpha * 0.0001;
		}

		if (sentence_length == 0) {

			while (1) {
				

				if (feof(fi))
					break;
				

				if(ngram > 0) //learn ngrams instead of words
				{
					
					
					if(newWord){
						ReadWord(wordToGram, fi);
						start = 0;
						end = ngram-1;
						wordLength = strlen(wordToGram);
					
						newWord = 0;
					}
					

					if(wordLength <= ngram){
						word =  SearchVocab(wordToGram);
						newWord = 1;
						continue;
					}

					
					for (i = 0; i < ngram; i++)
					{
						gram[i] = wordToGram[start+i];
					}
					gram[ngram] = '\0';


					word = SearchVocab(gram);
					
					end++;
					start++;

					if(end == wordLength)
						newWord = 1;
					
				}
				else
				{
				word = ReadWordIndex(fi); 

				}
				

				if (word == -1)
					continue;

				word_count++;

				if (word == 0)
					break;

				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample > 0) {
					real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
					next_random = next_random * (unsigned long long)25214903917 + 11;
					
					if (ran < (next_random & 0xFFFF) / (real)65536)
						continue;
				}
				sen[sentence_length] = word;
				sentence_length++;

				if (sentence_length >= MAX_SENTENCE_LENGTH)
					break;
			}
			
			sentence_position = 0;
		}

		if (feof(fi)) //end file
			break;

		if (word_count > train_words / num_threads) //trained all word
			break;

		word = sen[sentence_position]; //index

		if (word == -1) 
			continue;

		for (c = 0; c < layer1_size; c++)
			neu1[c] = 0;

		for (c = 0; c < layer1_size; c++)
			neu1e[c] = 0;

		next_random = next_random * (unsigned long long)25214903917 + 11;

		b = next_random % window;


		if (cbow) {  //train the cbow architecture
			// in -> hidden
			for (a = b; a < window * 2 + 1 - b; a++) //a = [0 window]->[(window*2+1)-rand] -> dynamic window
				if (a != window) {

					c = sentence_position - window + a;
					
					if (c < 0) continue;

					if (c >= sentence_length) continue;

					last_word = sen[c]; //index of word

					if (last_word == -1) continue;

					for (c = 0; c < layer1_size; c++) // c is each vector index
						neu1[c] += syn0[c + last_word * layer1_size]; //sum of all vectors in input window (fig cbow) -> vectors on hidden
			}

			if (hs)
				for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size; //offset of word
					// Propagate hidden -> output
					for (c = 0; c < layer1_size; c++)
						f += neu1[c] * syn1[c + l2]; //sum vectors input window * word weights on syn1 -> output vectors

					if (f <= -MAX_EXP) //sigmoid activation function - precalculated in expTable
						continue;
					else if (f >= MAX_EXP)
						continue;
					else
						f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha; 
					// Propagate errors output -> hidden
					for (c = 0; c < layer1_size; c++)
						neu1e[c] += g * syn1[c + l2]; //save to modify vectors
					// Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++)
						syn1[c + l2] += g * neu1[c]; //modify weights
				}
			// NEGATIVE SAMPLING
			if (negative > 0)
				for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1; //(w,c) in corpus
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % table_size];

						if (target == 0) 
							target = next_random % (vocab_size - 1) + 1;

						if (target == word)
							continue;

						label = 0; //(w,c) not in corpus
					}

					l2 = target * layer1_size; //get word vector index
					f = 0;

					for (c = 0; c < layer1_size; c++)
						f += neu1[c] * syn1neg[c + l2]; //vector*weights

					if (f > MAX_EXP) //sigmoid
						g = (label - 1) * alpha;
					else if (f < -MAX_EXP)
						g = (label - 0) * alpha;
					else
						g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

					for (c = 0; c < layer1_size; c++)
						neu1e[c] += g * syn1neg[c + l2]; //saving error

					for (c = 0; c < layer1_size; c++)
						syn1neg[c + l2] += g * neu1[c];
				}
			// hidden -> in
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = sentence_position - window + a;

				if (c < 0)
					continue;

				if (c >= sentence_length)
					continue;
				last_word = sen[c];

				if (last_word == -1)
					continue;

				for (c = 0; c < layer1_size; c++)
					syn0[c + last_word * layer1_size] += neu1e[c];  //modify word vectors with error
			}
		} else { 
				//SKIP-GRAM
			
			for (a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {

					c = sentence_position - window + a;

					if (c < 0)
						continue;
					if (c >= sentence_length)
						continue;

					last_word = sen[c];

					if (last_word == -1)
						continue;

					l1 = last_word * layer1_size; //word index

					for (c = 0; c < layer1_size; c++)
						neu1e[c] = 0;

					// HIERARCHICAL SOFTMAX
					if (hs)
						for (d = 0; d < vocab[word].codelen; d++) {
							f = 0;
							l2 = vocab[word].point[d] * layer1_size; //other words
							// Propagate hidden -> output
							for (c = 0; c < layer1_size; c++)
								f += syn0[c + l1] * syn1[c + l2];

							if (f <= -MAX_EXP)
								continue;
							else if (f >= MAX_EXP)
								continue;
							else
								f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

							// 'g' is the gradient multiplied by the learning rate
							g = (1 - vocab[word].code[d] - f) * alpha;
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++)
								neu1e[c] += g * syn1[c + l2];
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++)
								syn1[c + l2] += g * syn0[c + l1];
						}
					// NEGATIVE SAMPLING
					if (negative > 0)
						for (d = 0; d < negative + 1; d++) {
							if (d == 0) {
								target = word;
								label = 1;
							} else {
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table[(next_random >> 16) % table_size];

								if (target == 0)
									target = next_random % (vocab_size - 1) + 1;

								if (target == word)
									continue;
								label = 0;
							}

						l2 = target * layer1_size;
						f = 0;

						for (c = 0; c < layer1_size; c++)
							f += syn0[c + l1] * syn1neg[c + l2];

						if (f > MAX_EXP)
							g = (label - 1) * alpha;
						else if (f < -MAX_EXP)
							g = (label - 0) * alpha;
						else
							g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						
						for (c = 0; c < layer1_size; c++)
							neu1e[c] += g * syn1neg[c + l2];
						
						for (c = 0; c < layer1_size; c++)
							syn1neg[c + l2] += g * syn0[c + l1];
					}
					// Learn weights input -> hidden
					for (c = 0; c < layer1_size; c++)
						syn0[c + l1] += neu1e[c];
			}
		}

		sentence_position++;

		if (sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}

	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

void TrainModel() {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	if(debug_mode>0)
		printf("Starting training using file %s\n", train_file);

	starting_alpha = alpha;

	if (read_vocab_file[0] != 0)
		ReadVocab();
	else
		LearnVocabFromTrainFile();

	if (save_vocab_file[0] != 0)
		SaveVocab();

	if (output_file[0] == 0) //quit
		return;

	InitNet();

	if (negative > 0)
		InitUnigramTable();

	start = clock();

	for (a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);

	for (a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);

	if(debug_mode > 0)
		printf("Training Ended !\n");

	if(ngram > 0)
		return;


	fo = fopen(output_file, "wb");


	if (classes == 0) {
		// Save the word vectors
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			fprintf(fo, "%s ", vocab[a].word);

			if (binary)
				for (b = 0; b < layer1_size; b++)
					fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			else
				for (b = 0; b < layer1_size; b++)
					fprintf(fo, "%lf ", syn0[a * layer1_size + b]);

			fprintf(fo, "\n");
		}
	} else {
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));
		int *cl = (int *)calloc(vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

		for (a = 0; a < vocab_size; a++)
			cl[a] = a % clcn;

		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * layer1_size; b++)
				cent[b] = 0;

			for (b = 0; b < clcn; b++)
				centcn[b] = 1;

			for (c = 0; c < vocab_size; c++) {

				for (d = 0; d < layer1_size; d++)
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];

				centcn[cl[c]]++;
			}

			for (b = 0; b < clcn; b++) {
				closev = 0;

				for (c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}

				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++)
					cent[layer1_size * b + c] /= closev;
			}

			for (c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < layer1_size; b++)
						x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];

					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes

		for (a = 0; a < vocab_size; a++){
			fprintf(fo, "%s %d", vocab[a].word, cl[a]);

			for (b = 0; b < layer1_size; b++){
				fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			}
			fprintf(fo, "\n");
		}

		free(centcn);
		free(cent);
		free(cl);
	}

	fclose(fo);
}

/*group by sum*/
void sumGram(int offset, real* vector)
{
	int i;
	for (i=0; i < layer1_size;i++)
	{
		vector[i]+=syn0[offset+i];
	}
}

/*1->min 0->max*/
void minmaxGram(int offset,real *vector,int min)
{
	int i;

	if(min){
		for (i=0; i < layer1_size;i++)
		{
			if(vector[i]>syn0[offset+i])
				vector[i]=syn0[offset+i];
		}
	}
	else
	{
		for (i=0; i < layer1_size;i++)
		{
			if(vector[i]<syn0[offset+i])
				vector[i]=syn0[offset+i];
		}
	}
}

/*Divide vector by ngrams*/
void truncGram(int offset, real *vector, int wordLength, int gramPos)
{
	//nbGram = wordLength - ngram + 1;
	int nbDiv = layer1_size/(wordLength - ngram + 1);
	int i;
	// if nbdiv*nbGram =! layer1size
	for(i=gramPos*nbDiv;i<layer1_size;i++)//overrides - make condition cleaner 
	{ 
		vector[i] = syn0[offset+i];
	}
}

void createWordVectorFile(){
	char grama[ngram+1];
	int hash =0;
	char word[MAX_STRING];
	FILE *fin, *fo;
	int i,start,end,lenWord,indGram, offset;
	int *hashset;
	long long unsigned int cptWord=0;
	int skipCpt=0;
	int unexistCpt=0;
	int gramCpt=0;

	
	

	hashset = calloc(vocab_hash_size,sizeof(int));
	real wordVec[layer1_size];

	for(i=0;i<vocab_hash_size;i++)
		hashset[i] = -1;

	fin = fopen(train_file, "rb");
	fo = fopen(output_file, "wb");

	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	/* Counting number of words in file*/
	while (1) {
		
		if (feof(fin))
			break;

		ReadWord(word, fin);
		hash = GetWordHash(word);

		if (hashset[hash] != -1)
			continue;
		else
			hashset[hash] = 1;
		cptWord++;
	}

	fprintf(fo, "%lld %lld\n", cptWord, layer1_size); //prints size
	
	if(debug_mode > 0)
		printf("number of words: %lld\n",cptWord );

 	

	/*reset*/
	rewind(fin);
	for(i=0;i<vocab_hash_size;i++)
		hashset[i] = -1;
	cptWord=0;

	/*write </s>*/
	indGram = SearchVocab("</s>");
	offset = indGram * layer1_size;
	fprintf(fo, "</s> ");
	for (i = 0; i < layer1_size; i++){
		if (binary)
			fwrite(&wordVec[i], sizeof(real), 1, fo);
		else
			fprintf(fo, "%lf ", wordVec[i]);
	}
	fprintf(fo, "\n");


	/*Writing vectors*/
	while (1) {
		
		if (feof(fin))
			break;

		ReadWord(word, fin);

		hash = GetWordHash(word);
		for(i=0;i<layer1_size;i++) 
			wordVec[i] = 0;

		lenWord = strlen(word);
		start = 0;
		end = ngram-1;

		if(hashset[hash] != -1){
			skipCpt++;
			continue;
		}

		while(end<lenWord)
		{

			for (i = 0; i < ngram; i++)
			{
				grama[i] = word[start+i];
			}
			grama[ngram] = '\0';

			
			

			indGram = SearchVocab(grama);

			if(indGram > -1)
				offset = indGram * layer1_size;
			else
			{
				unexistCpt++;
				end++;
				start++;
				continue;
			}
			switch(group_vec){
				case 0:
				case 1:
					sumGram(offset,wordVec);
					break;
				case 2:
					minmaxGram(offset,wordVec,1);
					break;
				case 3:
					minmaxGram(offset,wordVec,0);
					break;
				case 4:
					truncGram(offset,wordVec,lenWord,gramCpt);
					break;
			}
			//printf("gram: %s\n",grama );
			for(i=0;i<layer1_size;i++){
				wordVec[i] += syn0[offset+i];
			}

			gramCpt++;

			end++;
			start++;
		}


		if(group_vec==0) //Mean
		{
			//normalization
			for(i=0;i<layer1_size;i++){
					wordVec[i] /= gramCpt;
			}
		}
		hashset[hash] = 1;
		cptWord++;
		gramCpt = 0;



		//removes #bangs
		if(hashbang > 0){
			for(i=1;i<lenWord;i++){
				word[i-1]=word[i];
			}
			word[lenWord-2]='\0';
		}


		fprintf(fo, "%s ", word);
		for (i = 0; i < layer1_size; i++){
			if (binary)
					fwrite(&wordVec[i], sizeof(real), 1, fo);
			else
					fprintf(fo, "%lf ", wordVec[i]);
		}
		
		fprintf(fo, "\n");
		
	}
	if(debug_mode > 0)
		printf("Saved %lld word vectors, %d grams weren't in dictionnary, %d words were skipped (doubles)\n",cptWord,unexistCpt,skipCpt);
	
	fclose(fo);
	fclose(fin);
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
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-classes <int>\n");
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
		printf("\t-ngram <int> (default 0 - use words) \n");
		printf("\t\tUse N-GRAM model instead of words to train vectors \n");
		printf("\t-hashbang <0-1> (default 0)\n");
		printf("\t\tUse hashbang on n-grams - i.e #good# -> #go,goo,ood,od#\n");
		printf("\t-group <0-5> (default 0)\n");
		printf("\t\tHow word vectors are computed with n-grams - 0:Mean (default); 1:Sum; 2:Min; 3:Max; 4:Trunc; 5:FreqSum\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
		return 0;
	}

	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;

	if ((i = ArgPos((char *)"-size", argc, argv)) > 0)layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
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
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	if ((i = ArgPos ((char *) "-ngram", argc, argv)) > 0 ) ngram = atoi(argv[i + 1]);
	if ((i = ArgPos ((char *) "-hashbang", argc, argv)) > 0 ) hashbang = atoi(argv[i + 1]);
	if ((i = ArgPos ((char *) "-group", argc, argv)) > 0 ) group_vec = atoi(argv[i + 1]);
	

	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));

	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));

	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	
	TrainModel();
	if(ngram > 0)
		createWordVectorFile();
	return 0;
}