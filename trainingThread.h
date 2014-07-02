#ifndef TRAIN_THREADS
#define TRAIN_THREADS



struct threadParameters {
    vocabulary *voc;
    int threadNumber;
};

void *TrainModelThread(void *arg);


#endif