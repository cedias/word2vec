if [ "$#" -ne 8 ]; then
    echo "Illegal number of parameters"
    echo "Usage: testNgram size sample neg alpha ngram hashbang cbow hs"
fi

p_size=$1
p_sample=$2
p_neg=$3
p_alpha=$4
p_ngram=$5
p_hashb=$6
p_cbow=$7
p_hs=$8

./word2vec -train text8 -output /tmp/vectors.bin -debug 0 -min-count 0 -window 5 -threads 12 -binary 1 -cbow $p_cbow -size $p_size -negative $p_neg -hs $p_neg -sample $p_sample -ngram $p_ngram -hashbang $p_hashb -alpha $p_alpha 
./compute-accuracy-syntax /tmp/vectors.bin 10000 2 < questions-words-syntax.txt
