#ifndef NGRAMTOOLS
#define NGRAMTOOLS
#define DEBUG_MODE 2

typedef float real;

void gramVocToWordVec(vocabulary* voc, real* syn0,int max_string, int layer1_size, int ngram, int hashbang,int group_vec, int binary,int position,int overlap, char* train_file, char* output_file);
void writeGrams(vocabulary* voc,real *syn0,int layer1_size,int ngram,int hashbang,int position,char* output_file, int binary);

/*string to ngrams*/
int getGrams(char* word, char* gram, int index, int size,int overlap,int position,int hashbang);

/*adds position info to n-gram*/
void addGramPosition(char* word, char * gram, int size, int index, int position, int overlap, int hashbang);


/*utility*/
void sumGram(real* syn0, int layer1_size, int offset, real* vector);
void sumFreqGram(real* syn0, int layer1_size,int offset, real* vector,int cn);
void minmaxGram(real* syn0, int layer1_size,int offset,real *vector,int min);
void truncGram(real* syn0, int layer1_size,int ngram,int offset, real *vector, int wordLength, int gramPos);

#endif