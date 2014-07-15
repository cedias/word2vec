#include "vocab.h"
#include "ngram_tools.h"
#include <math.h>

void writeGrams(vocabulary* voc,real *syn0,int layer1_size,int ngram,int hashbang,int position,char* output_file, int binary){
	FILE *fo = fopen(output_file,"wb");
	int a,b;


	fprintf(fo, "%lld %d %d %d %d\n", voc->vocab_size, layer1_size, ngram, hashbang, position);
	for (a = 0; a < voc->vocab_size; a++) {
		fprintf(fo, "%s ", voc->vocab[a].word);

		if (binary)
			for (b = 0; b < layer1_size; b++)
				fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
		else
			for (b = 0; b < layer1_size; b++)
				fprintf(fo, "%lf ", syn0[a * layer1_size + b]);

		fprintf(fo, "\n");
	}

	fclose(fo);
}

void gramVocToWordVec(vocabulary* voc, real* syn0,int max_string, int layer1_size, int ngram, int hashbang,int group_vec, int binary,int position,int overlap, char* train_file, char* output_file){

	FILE *fin, *fo;
	char grama[ngram+3];
	int hash = 0;
	char word[max_string];
	int i,start,end,lenWord, offset;
	int *hashset;
	long long unsigned int cptWord=0;
	long long int indGram;
	int skipCpt=0;
	int unexistCpt=0;
	int gramCpt=0;

	hashset = calloc(voc->vocab_hash_size,sizeof(int));
	real wordVec[layer1_size];

	for(i=0;i<voc->vocab_hash_size;i++)
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
		hash = GetWordHash(voc,word);

		if (hashset[hash] != -1)
			continue;
		else
			hashset[hash] = 1;

		cptWord++;
	}
	
	fprintf(fo, "%lld %d\n", cptWord+1, layer1_size); //prints size + 1 for </s>

	if(DEBUG_MODE > 0)
		printf("number of words: %lld\n",cptWord+1 );

	/*reset*/
	rewind(fin);
	for(i=0;i<voc->vocab_hash_size;i++)
		hashset[i] = -1;

	for(i=0;i<layer1_size;i++) 
		wordVec[i] = 0;

	cptWord=0;

	
	/*write </s>*/

	for(i=0;i<layer1_size;i++)
	{
		wordVec[i] = 0; //syn0[offset+i];
	}
	
	fprintf(fo, "</s> ");
	for (i = 0; i < layer1_size; i++){
		if (binary)
			fwrite(&wordVec[i], sizeof(real), 1, fo);
		else
			fprintf(fo, "%lf ", wordVec[i]);
	}
	fprintf(fo, "\n");
	
	cptWord=1;

	/*Writing vectors*/

	while (1) {
		
		if (feof(fin))
			break;

		if (hashbang)
			ReadWordHashbang(word, fin);
		else
			ReadWord(word,fin);

		hash = GetWordHash(voc,word);

		for(i=0;i<layer1_size;i++) 
			wordVec[i] = 0;

		lenWord = strlen(word);

		if(hashset[hash] != -1){
			skipCpt++;
			continue;
		}

		while(getGrams(word,grama,gramCpt, ngram, overlap, position,hashbang))
		{

			indGram = SearchVocab(voc,grama);
			
			if(indGram > -1)
				offset = indGram * layer1_size;
			else
			{
				unexistCpt++;
				continue;
			}
			
			switch(group_vec){
				case 0:
				case 1:
					sumGram(syn0, layer1_size, offset,wordVec);
					break;
				case 2:
					minmaxGram(syn0, layer1_size,offset,wordVec,1);
					break;
				case 3:
					minmaxGram(syn0, layer1_size,offset,wordVec,0);
					break;
				case 4:
					truncGram(syn0, layer1_size,ngram,offset,wordVec,lenWord,gramCpt);
					break;
				case 5:
					sumFreqGram(syn0, layer1_size,offset,wordVec,voc->vocab[indGram].cn);
			}

			gramCpt++;
			end++;
			start++;
		}

		if(group_vec==1){
			for(i=0;i<layer1_size;i++){
				wordVec[i] /= 2;
			}
		}

		//normalization
		for(i=0;i<layer1_size;i++){
				wordVec[i] /= gramCpt;
		}
		
		hashset[hash] = 1;		
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
		
		
		
		cptWord++;
		
	}

	if(DEBUG_MODE > 0)
		printf("Saved %lld word vectors, %d grams weren't in dictionnary, %d words were skipped (doubles)\n",cptWord,unexistCpt,skipCpt);

	fclose(fo);
	fclose(fin);
	free(hashset);
}


/* Adds position to gram Ngram - gram tab size is ngram+3 index: [0->ngram+2]*/
void addGramPosition(char* word, char * gram, int size, int index, int position, int overlap,int hashbang){
	int i;
	char num[3];
	int lenWord = strlen(word);
	int lastIndex;

	if(overlap)
		lastIndex = lenWord-size;
	else
		lastIndex = lenWord/size-1;
	


	if(position==1){
		/*	Adds '-' /!\ intended for no hashbangs */

		if(index==0) //first index
		{
			gram[size]='-';
			gram[size+1]='\0'; 
			return;
		}

		if(index == lastIndex) //last index
		{
			for(i=size+1;i>0;i--){
				gram[i]=gram[i-1];
			}
			gram[0]='-';

			return;
		}

		for(i=size+1;i>0;i--){
			gram[i]=gram[i-1];
		}
			gram[0]='-';
			gram[size+1]='-';
			gram[size+2]='\0';
	}
	else
	{
		/* adds #index- /!\ start must be <= 99 */

		if(index==0 && gram[0]=='#')
			return;

		if(index == lastIndex && hashbang)
			return;

		for(i=size+3;i>=3;i--)
		{
			gram[i]=gram[i-3];
		}
		
		sprintf(num,"%d",index);
		if(index>=10){
			gram[0] = num[0];	
			gram[1] = num[1];
		}else{
			gram[0] = '0';
			gram[1] = num[0];
		}
		gram[2] = '-';

	}
	
	return;
}

int getGrams(char* word, char* gram, int index, int size,int overlap,int position,int hashbang){
	int lenWord = strlen(word);
	int lastIndex;

	if(overlap)
		lastIndex = lenWord-size;
	else
		lastIndex = lenWord/size-1;
	

	if(index > lastIndex)
		return 0;

	if(lenWord <= size){
		return -1;
	}

	int start,i;

	for(i = 0;i<size+4;i++) //gram = ngram+4 (convention)
		gram[i]=0; //reset gram

	if(overlap){
		start = index;
		
		for (i = 0; i < size; i++)
		{
			gram[i] = word[start+i];
		}
		gram[size] = '\0';

		if(position > 0)
			addGramPosition(word,gram,size,index,position,overlap,hashbang);
	}
	else
	{
		
		start = index * size;
	
		i=0;
		int f
		if(start+size > lenWord || lenWord-(start+size) <= 2){
			while(word[start+i] != '\0'){
				gram[i] = word[start+i];
				i++;
			}
			gram[i] = '\0';

		}
		else 
		{
			for(i=0;i<size;i++)
				gram[i]=word[start+i];
			gram[size]='\0';
		}

		
		
		

		if(position > 0)
			addGramPosition(word,gram,size,index,position,overlap, hashbang);
	}
	

	return 1;
}

/*group by sum*/
void sumGram(real* syn0, int layer1_size, int offset, real* vector)
{
	int i;
	for (i=0; i < layer1_size;i++)
	{
		vector[i]+=syn0[offset+i];
	}
}

/*group by sum*/
void sumFreqGram(real* syn0, int layer1_size,int offset, real* vector,int cn)
{
	int i;
	for (i=0; i < layer1_size;i++)
	{
		vector[i]+=(syn0[offset+i]* (1.00 / cn));
	}

}

/*1->min 0->max*/
void minmaxGram(real* syn0, int layer1_size,int offset,real *vector,int min)
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
void truncGram(real* syn0, int layer1_size,int ngram,int offset, real *vector, int wordLength, int gramPos)
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


/* in case of double training
void mergeAndSaveVectors(){

	int i = 0;
	int j = 0;
	int indWord;
	int offset;
	char * word;
	real wordVec[layer1_size];
	int cptSkipped = 0;
	FILE *fo;

	fo = fopen(output_file, "wb");
	fprintf(fo, "%lld %lld\n", saveArrayLength, layer1_size); //prints size

	for(i=0;i<saveArrayLength;i++){
		word = saveArray[i].word;

		for (j = 0; j < layer1_size; j++)
		{
			wordVec[j] = saveArray[i].vector[j];
		}
		
		indWord = SearchVocab(word);
		
		if(indWord == -1)
		{
			cptSkipped++;
		}
		else
		{
			offset = indWord*layer1_size;

			for (j = 0; j < layer1_size; j++)
			{
				wordVec[j] = (wordVec[j] + syn0[offset+j])/2; //mean
			}
		}
		
		//save to file

		fprintf(fo, "%s ", word);

			for (j = 0; j < layer1_size; j++){
				if (binary)
						fwrite(&wordVec[j], sizeof(real), 1, fo);
				else
						fprintf(fo, "%lf ", wordVec[j]);
			}

		fprintf(fo, "\n");
	}
	printf("skipped %d/%lld words, they were down-sampled by word training - they only have syntactic vectors \n",cptSkipped,saveArrayLength );
}
*/