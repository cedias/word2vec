make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
time ./word2vec -train text8 -output /tmp/vectors.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 0 -threads 12 -binary 1 -ngram 8 -hashbang 1 -min-count 0 
./compute-accuracy /tmp/vectors.bin 30000 < questions-words.txt
# to compute accuracy with the full vocabulary, use: ./compute-accuracy vectors.bin < questions-words.txt
