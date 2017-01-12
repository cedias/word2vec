make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
time ./word2gram -train text8 -output vectors.bin -ngram 2 -group -1 -hashbang 1 -cbow 0 -size 200 -window 10 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1 -min-count 0
./distance vectors.bin
