// mem_gpu_setup.h
// function:

#ifndef MEM_GPU_SETUP_H
#define MEM_GPU_SETUP_H



#include <vector>
#include "library.h"





using namespace std;




void mem_gpu_init();
void mem_gpu_release();

void mem_gpu_settissue(int);
void mem_gpu_destroytissue(int);






#endif

// end of mem_gpu_setup.h
