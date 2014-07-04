#include "vocab.h"
#include "ngram_tools.h"


void gramVocToWordVec(vocabulary* voc, real* syn0,int max_string, int layer1_size, int ngram, int hashbang,int group_vec, int binary, char* train_file, char* output_file){

	FILE *fin, *fo;
	char grama[ngram+1];
	int hash = 0;
	char word[max_string];
	int i,start,end,lenWord,indGram, offset;
	int *hashset;
	long long unsigned int cptWord=0;
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

		ReadWord(word, fin,hashbang);
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

		ReadWord(word, fin,hashbang);

		hash = GetWordHash(voc,word);

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
			indGram = SearchVocab(voc,grama);

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

		if(group_vec==0 || group_vec==5) //Mean
		{
			//normalization
			for(i=0;i<layer1_size;i++){
					wordVec[i] /= gramCpt;
			}
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