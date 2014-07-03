#ifndef TRAIN_THREADS
#define TRAIN_THREADS
#define DEBUG_MODE 2

typedef float real;

void CreateWordVector(vocabulary* voc,
	real* syn0,
	int max_string,
	int layer1_size,
	int ngram,
	int hashbang,
	int group_vec,
	int binary,
	char* train_file,
	char* output_file
	);

void sumGram(real* syn0, int layer1_size, int offset, real* vector);
void sumFreqGram(real* syn0, int layer1_size,int offset, real* vector,int cn);
void minmaxGram(real* syn0, int layer1_size,int offset,real *vector,int min);
void truncGram(real* syn0, int layer1_size,int ngram,int offset, real *vector, int wordLength, int gramPos);

#endif