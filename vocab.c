#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#include "vocab.h"

/*Inits a vocabulary*/
vocabulary* InitVocabulary( int vocab_hash_size, unsigned long long int vocab_max_size){
	unsigned long long i;

	vocabulary* voc = (vocabulary*) malloc(sizeof(vocabulary));
	if(voc == NULL){
		printf("vocabulary couldn't be created, memory problem, exiting...\n");
		exit(1);
	}
	voc->vocab_hash = (unsigned long long int *) calloc(vocab_hash_size, sizeof(unsigned long long int));
	if(voc == NULL){
		printf("vocabulary hash couldn't be created, memory problem, exiting...\n");
		exit(1);
	}

	voc->vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
	if(voc == NULL){
		printf("vocabulary hash couldn't be created, memory problem, exiting...\n");
	}
	voc->vocab_size = 0;
	voc->vocab_hash_size = vocab_hash_size;
	voc->vocab_max_size = vocab_max_size;

	for(i=0;i<voc->vocab_hash_size;i++)
		voc->vocab_hash[i] = -1;

	for(i=0;i<voc->vocab_max_size;i++)
		voc->vocab[i].cn = 0;

	return voc;
}


/*Reads a word from file descriptor fin*/
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
	return;
}

/*Reads a word and adds #hashbangs# around it from file descriptor fin*/
void ReadWordHashbang(char *word, FILE *fin) {
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

 	//adding #word#
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
	return;
}


/* Returns hash value of a word*/
int GetWordHash(vocabulary* voc, char *word) {
	unsigned long long a, hash = 0;

	for (a = 0; a < strlen(word); a++) 
		hash = hash * 257 + word[a];

	hash = hash % voc->vocab_hash_size;
	return hash;
}

/*free the vocab structure*/ //TODO
void DestroyVocab(vocabulary* voc) {
  int a;

  for (a = 0; a < voc->vocab_size; a++) {
    if (voc->vocab[a].word != NULL) {
      free(voc->vocab[a].word);
    }
    if (voc->vocab[a].code != NULL) {
      free(voc->vocab[a].code);
    }
    if (voc->vocab[a].point != NULL) {
      free(voc->vocab[a].point);
    }
  }
  free(voc->vocab[voc->vocab_size].word);
  free(voc->vocab);
}

/* Returns position of a word in the vocabulary;
 if the word is not found, returns -1*/
int SearchVocab(vocabulary* voc, char *word) {
	unsigned int hash = GetWordHash(voc, word);

	while (1) {
		if (voc->vocab_hash[hash] == -1)
			return -1;

		if (!strcmp(word, voc->vocab[voc->vocab_hash[hash]].word))
			return voc->vocab_hash[hash];

		hash = (hash + 1) % voc->vocab_hash_size;
	}

	return -1;
}

/* Reads a word and returns its index in the vocabulary*/
int ReadWordIndex(vocabulary* voc, FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);

	if (feof(fin)) 
		return -1;

	return SearchVocab(voc, word);
}

/* Adds a word to the vocabulary
	Returns the vocabulary size */
int AddWordToVocab(vocabulary* voc, char *word) {

	unsigned int hash, length = strlen(word) + 1;

	if (length > MAX_STRING)
		length = MAX_STRING;

	
	voc->vocab[voc->vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(voc->vocab[voc->vocab_size].word, word);

	voc->vocab[voc->vocab_size].cn = 0;
	voc->vocab_size++;

	// Reallocate memory if needed
	if (voc->vocab_size + 2 >= voc->vocab_max_size) {
		voc->vocab_max_size += 1000;
		voc->vocab = (struct vocab_word *)realloc(voc->vocab, voc->vocab_max_size * sizeof(struct vocab_word));
	}
	
	hash = GetWordHash(voc,word);

	while (voc->vocab_hash[hash] != -1)
		hash = (hash + 1) % voc->vocab_hash_size;

	voc->vocab_hash[hash] = voc->vocab_size - 1;
	return voc->vocab_size - 1;
}

/* Used for sorting by word counts */
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

/* Sorts the vocabulary by frequency using word counts*/
void SortVocab(vocabulary* voc, int min_count) {
	int a, size;
	unsigned int hash;

	if(DEBUG_MODE > 2)
		printf("Sorting Vocab...\n");

	// Sort the vocabulary and keep </s> at the first position
	qsort(&voc->vocab[1], voc->vocab_size - 1, sizeof(struct vocab_word), VocabCompare);

	for (a = 0; a < voc->vocab_hash_size; a++)
		voc->vocab_hash[a] = -1;

	size = voc->vocab_size;
	voc->train_words = 0;

	for (a = 1; a < size; a++) {
	// Words occuring less than min_count times will be discarded from the vocab
		if (voc->vocab[a].cn < min_count) {
			voc->vocab_size--;
			//free(vocab[vocab_size].word); 
			//free(vocab[a].word);
			voc->vocab[a].word = NULL;
		}
		else {
		// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(voc, voc->vocab[a].word);

			while (voc->vocab_hash[hash] != -1)
				hash = (hash + 1) % voc->vocab_hash_size;

			voc->vocab_hash[hash] = a;
			voc->train_words += voc->vocab[a].cn;
		}
	}


	voc->vocab = (struct vocab_word *)realloc(voc->vocab, (voc->vocab_size + 1) * sizeof(struct vocab_word));

	// Allocate memory for the binary tree construction
	for (a = 0; a < voc->vocab_size; a++) {
		voc->vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		voc->vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

/* Reduces the vocabulary by removing infrequent tokens */
void ReduceVocab(vocabulary* voc, int min_reduce) {

	int a, b = 0;
	unsigned int hash;

	for (a = 0; a < voc->vocab_size; a++){
		if (voc->vocab[a].cn > min_reduce) {

			voc->vocab[b].cn = voc->vocab[a].cn;
			voc->vocab[b].word = voc->vocab[a].word;
			b++;

		} else
			free(voc->vocab[a].word);
	}

	voc->vocab_size = b;

	for (a = 0; a < voc->vocab_hash_size; a++)
		voc->vocab_hash[a] = -1;

	for (a = 0; a < voc->vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(voc, voc->vocab[a].word);

		while (voc->vocab_hash[hash] != -1)
			hash = (hash + 1) % voc->vocab_hash_size;

		voc->vocab_hash[hash] = a;
	}

	fflush(stdout);
	min_reduce++;
}

/*Look if word already in vocab, if not add, if yes, increment. */
void searchAndAddToVocab(vocabulary* voc, char* word){
	long long a,i;
	i = SearchVocab(voc, word);

		if (i == -1) {
			a = AddWordToVocab(voc, word);
			voc->vocab[a].cn = 1;
		} else
			voc->vocab[i].cn++;

		if (voc->vocab_size > voc->vocab_hash_size * 0.7)
			ReduceVocab(voc,0);  //////////////////CAUTION
}

/*Create a vocab from train file*/
long long LearnVocabFromTrainFile(vocabulary* voc, char* train_file,int min_count) {
	int i;
	char word[MAX_STRING];
	FILE * fin;

	for (i = 0; i < voc->vocab_hash_size; i++) //init vocab hashtable
		voc->vocab_hash[i] = -1;

	fin = fopen(train_file, "rb");

	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	voc->vocab_size = 0;
	AddWordToVocab(voc, (char *)"</s>");

	while (1) {

		ReadWord(word, fin);
		searchAndAddToVocab(voc,word);
		
		if (feof(fin))
			break;

		voc->train_words++;

		if ((DEBUG_MODE > 1) && (voc->train_words % 100000 == 0)) {
			printf("%lldK%c", voc->train_words / 1000, 13);
			fflush(stdout);
		}
	}

	SortVocab(voc,min_count);

	if (DEBUG_MODE > 1) {
		printf("Vocab size: %lld\n", voc->vocab_size);
		printf("Words in train file: %lld\n", voc->train_words);
	}

	long long file_size = ftell(fin);
	fclose(fin);
	return file_size;
}

/*Create a vocab of ngram from train file*/
long long LearnNGramFromTrainFile(vocabulary* voc, char* train_file,int min_count, int ngram, int hashbang) {
	char word[MAX_STRING];
	int i,start,end,lenWord;
	FILE * fin;
	
	char gram[ngram+1];

	for (i = 0; i < voc->vocab_hash_size; i++) //init vocab hashtable
		voc->vocab_hash[i] = -1;

	fin = fopen(train_file, "rb");

	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	voc->vocab_size = 0;
	AddWordToVocab(voc, (char *)"</s>");

	while (1) {

		if(hashbang)
			ReadWordHashbang(word,fin);
		else
			ReadWord(word,fin);

		lenWord = strlen(word);

		if(lenWord<=ngram){ //word smaller or equal to ngram var.
			searchAndAddToVocab(voc,word);

			if (feof(fin))
				break;
			else
				continue;
		}

		start = 0;
		end = ngram-1;
		i=0;

		while(end<lenWord)
		{

			for (i = 0; i < ngram; i++)
			{
				gram[i] = word[start+i];
			}
			gram[ngram] = '\0';

			searchAndAddToVocab(voc,gram);

			end++;
			start++;
		}

		if (feof(fin))
			break;

		voc->train_words++;

		if ((DEBUG_MODE > 1) && (voc->train_words % 100000 == 0)) {
			printf("%lldK%c", voc->train_words / 1000, 13);
			fflush(stdout);
		}
	}

	SortVocab(voc,min_count);

	if (DEBUG_MODE > 1) {
		printf("Vocab size: %lld\n", voc->vocab_size);
		printf("Words in train file: %lld\n", voc->train_words);
	}

	long long file_size = ftell(fin);
	fclose(fin);
	return file_size;
}

/*Saves vocab & Occurences*/
void SaveVocab(vocabulary* voc, char* save_vocab_file) {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");

  for (i = 0; i < voc->vocab_size; i++)
  	fprintf(fo, "%s %lld\n", voc->vocab[i].word, voc->vocab[i].cn);

  fclose(fo);
}

/*Reads a saved vocab file*/
long long ReadVocab(vocabulary* voc, char* read_vocab_file, char* train_file,int min_count) {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");

	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}

	for (a = 0; a < voc->vocab_hash_size; a++)
		voc->vocab_hash[a] = -1;

	voc->vocab_size = 0;

	while (1) {
		ReadWord(word, fin);

		if (feof(fin))
			break;

		a = AddWordToVocab(voc,word);
		fscanf(fin, "%lld%c", &voc->vocab[a].cn, &c);
		i++;
	}

	SortVocab(voc,min_count);

	if (DEBUG_MODE > 1) {
		printf("Vocab size: %lld\n", voc->vocab_size);
		printf("Words in train file: %lld\n", voc->train_words);
	}

	fin = fopen(train_file, "rb");

	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	fseek(fin, 0, SEEK_END);
	long long file_size = ftell(fin);
	fclose(fin);
	return file_size;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree(vocabulary* voc) {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(voc->vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(voc->vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(voc->vocab_size * 2 + 1, sizeof(long long));

	for (a = 0; a < voc->vocab_size; a++)
		count[a] = voc->vocab[a].cn;

	for (a = voc->vocab_size; a < voc->vocab_size * 2; a++) //sets rest of count array to 1e15
		count[a] = 1e15;

	pos1 = voc->vocab_size - 1; //end of word occurences
	pos2 = voc->vocab_size; //start of other end

	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < voc->vocab_size - 1; a++) { //vocab is already sorted by frequency
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

		count[voc->vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = voc->vocab_size + a;
		parent_node[min2i] = voc->vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < voc->vocab_size; a++) {
		b = a;
		i = 0;

		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];

			if (b == voc->vocab_size * 2 - 2)
				break;
		}

		voc->vocab[a].codelen = i;
		voc->vocab[a].point[0] = voc->vocab_size - 2;

		for (b = 0; b < i; b++) {
			voc->vocab[a].code[i - b - 1] = code[b];
			voc->vocab[a].point[i - b] = point[b] - voc->vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}
