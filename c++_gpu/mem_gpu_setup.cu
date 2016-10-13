// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <random>
#include <chrono>		/* sys time */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>


// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// private
#include "global.h"
#include "library.h"
#include "mem_gpu_setup.h"




using namespace std;




void mem_gpu_init()
{
	int dimension1, dimension2;


	//== d_X_batch, d_Z_batch, d_Y_batch, d_Y_batch_exp, d_cellfactor_batch, d_cellfactor_batch_new
	//
	dimension2 = X.get_dimension2();
	checkCudaErrors(cudaMalloc((void **) &d_X_batch, (size_batch*dimension2)*sizeof(float)));
	//
	dimension2 = Z.get_dimension2();
	checkCudaErrors(cudaMalloc((void **) &d_Z_batch, (size_batch*dimension2)*sizeof(float)));
	//
	dimension2 = Y.get_dimension3();
	checkCudaErrors(cudaMalloc((void **) &d_Y_batch, (size_batch*dimension2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_Y_batch_exp, (size_batch*dimension2)*sizeof(float)));
	//
	checkCudaErrors(cudaMalloc((void **) &d_cellfactor_batch, (size_batch*D)*sizeof(float)));
	//
	checkCudaErrors(cudaMalloc((void **) &d_cellfactor_batch_new, (size_batch*(D+1))*sizeof(float)));




	//== list_cis_start, d_list_cis_start
	int * list_cis_start = mapping_cis.get_list_start();
	checkCudaErrors(cudaMalloc((void **) &d_list_cis_start, J*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_list_cis_start, list_cis_start, J*sizeof(int), cudaMemcpyHostToDevice));
	//== list_cis_end, d_list_cis_end
	int * list_cis_end = mapping_cis.get_list_end();
	checkCudaErrors(cudaMalloc((void **) &d_list_cis_end, J*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_list_cis_end, list_cis_end, J*sizeof(int), cudaMemcpyHostToDevice));
	//== list_beta_cis_start, d_list_beta_cis_start
	int * list_beta_cis_start = beta_cis.get_list_start();
	checkCudaErrors(cudaMalloc((void **) &d_list_beta_cis_start, J*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_list_beta_cis_start, list_beta_cis_start, J*sizeof(int), cudaMemcpyHostToDevice));




	//== d_beta_cis_sub
	int amount = beta_cis.get_amount();
	checkCudaErrors(cudaMalloc((void **) &d_beta_cis_sub, amount*sizeof(float)));

	//== d_beta_batch, d_beta_batch_reshape
	int dimension1_beta_batch = beta_batch.get_dimension1();
	int dimension2_beta_batch = beta_batch.get_dimension2();
	checkCudaErrors(cudaMalloc((void **) &d_beta_batch, (dimension1_beta_batch*dimension2_beta_batch)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_beta_batch_reshape, (dimension1_beta_batch*dimension2_beta_batch)*sizeof(float)));

	//== d_beta_cellfactor1, d_beta_cellfactor1_reshape
	int dimension1_beta_cellfactor1 = beta_cellfactor1.get_dimension1();
	int dimension2_beta_cellfactor1 = beta_cellfactor1.get_dimension2();
	checkCudaErrors(cudaMalloc((void **) &d_beta_cellfactor1, (dimension1_beta_cellfactor1*dimension2_beta_cellfactor1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_beta_cellfactor1_reshape, (dimension1_beta_cellfactor1*dimension2_beta_cellfactor1)*sizeof(float)));

	//== d_beta_cellfactor2_sub, d_beta_cellfactor2_sub_reshape
	int dimension1_beta_cellfactor2 = beta_cellfactor2.get_dimension2();
	int dimension2_beta_cellfactor2 = beta_cellfactor2.get_dimension3();
	checkCudaErrors(cudaMalloc((void **) &d_beta_cellfactor2_sub, (dimension1_beta_cellfactor2*dimension2_beta_cellfactor2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_beta_cellfactor2_sub_reshape, (dimension1_beta_cellfactor2*dimension2_beta_cellfactor2)*sizeof(float)));



	return;
}



void mem_gpu_release()
{
	checkCudaErrors(cudaFree(d_X_batch));
	checkCudaErrors(cudaFree(d_Z_batch));
	checkCudaErrors(cudaFree(d_Y_batch));
	checkCudaErrors(cudaFree(d_Y_batch_exp));
	checkCudaErrors(cudaFree(d_cellfactor_batch));
	checkCudaErrors(cudaFree(d_cellfactor_batch_new));

	checkCudaErrors(cudaFree(d_list_cis_start));
	checkCudaErrors(cudaFree(d_list_cis_end));
	checkCudaErrors(cudaFree(d_list_beta_cis_start));
	checkCudaErrors(cudaFree(d_beta_cis_sub));

	checkCudaErrors(cudaFree(d_beta_batch));
	checkCudaErrors(cudaFree(d_beta_batch_reshape));

	checkCudaErrors(cudaFree(d_beta_cellfactor1));
	checkCudaErrors(cudaFree(d_beta_cellfactor1_reshape));

	checkCudaErrors(cudaFree(d_beta_cellfactor2_sub));
	checkCudaErrors(cudaFree(d_beta_cellfactor2_sub_reshape));

	return;
}





// init tissue data and parameters in tissue#k
void mem_gpu_settissue(int k)
{
	//===============================================================
	//==== init and transmit tissue data when starting a new tissue
	//===============================================================
	int dimension1 = Y.get_dimension2_at(k);
	int dimension2 = J;
	int * list_indiv_pos = Y.get_list_indiv_pos_at(k);
	int dimension;

	//==== construct the genotype matrix for this tissue (with intercept term), on CPU then GPU memory
	//== X_sub, d_X_sub
	Matrix X_sub;
	dimension = X.get_dimension2();
	X_sub.init(dimension1, dimension);
	float * X_pointer = X.get_pointer();
	X_sub.fill_with_ref_list(list_indiv_pos, X_pointer);
	//
	checkCudaErrors(cudaMalloc((void **) &d_X_sub, (dimension1*dimension)*sizeof(float)));
	float * X_sub_pointer = X_sub.get_pointer();
	checkCudaErrors(cudaMemcpy(d_X_sub, X_sub_pointer, (dimension1*dimension)*sizeof(float), cudaMemcpyHostToDevice));

	//== Z_sub, d_Z_sub
	Matrix Z_sub;
	dimension = Z.get_dimension2();
	Z_sub.init(dimension1, dimension);
	float * Z_pointer = Z.get_pointer();
	Z_sub.fill_with_ref_list(list_indiv_pos, Z_pointer);
	//
	checkCudaErrors(cudaMalloc((void **) &d_Z_sub, (dimension1*dimension)*sizeof(float)));
	float * Z_sub_pointer = Z_sub.get_pointer();
	checkCudaErrors(cudaMemcpy(d_Z_sub, Z_sub_pointer, (dimension1*dimension)*sizeof(float), cudaMemcpyHostToDevice));

	//== d_Y_sub, d_Y_sub_exp
	float * Y_sub_pointer = Y.get_matrix_at(k);
	checkCudaErrors(cudaMalloc((void **) &d_Y_sub, (dimension1*dimension2)*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_Y_sub, Y_sub_pointer, (dimension1*dimension2)*sizeof(float), cudaMemcpyHostToDevice));
	//
	checkCudaErrors(cudaMalloc((void **) &d_Y_sub_exp, (dimension1*dimension2)*sizeof(float)));

	//== d_cellfactor_sub, d_cellfactor_sub_new
	checkCudaErrors(cudaMalloc((void **) &d_cellfactor_sub, (dimension1*D)*sizeof(float)));
	//
	checkCudaErrors(cudaMalloc((void **) &d_cellfactor_sub_new, (dimension1*(D+1))*sizeof(float)));


	//===============================================================
	//==== transmit tissue parameters when starting a new tissue
	//===============================================================
	//== d_beta_cis_sub
	float * beta_cis_sub = beta_cis.get_incomp_matrix_at(k);
	int amount = beta_cis.get_amount();
	checkCudaErrors(cudaMemcpy(d_beta_cis_sub, beta_cis_sub, amount*sizeof(float), cudaMemcpyHostToDevice));

	//== d_beta_batch
	float * beta_batch_pointer = beta_batch.get_pointer();
	int dimension1_beta_batch = beta_batch.get_dimension1();
	int dimension2_beta_batch = beta_batch.get_dimension2();
	checkCudaErrors(cudaMemcpy(d_beta_batch, beta_batch_pointer, (dimension1_beta_batch*dimension2_beta_batch)*sizeof(float), cudaMemcpyHostToDevice));

	//== d_beta_cellfactor1
	float * beta_cellfactor1_pointer = beta_cellfactor1.get_pointer();
	int dimension1_beta_cellfactor1 = beta_cellfactor1.get_dimension1();
	int dimension2_beta_cellfactor1 = beta_cellfactor1.get_dimension2();
	checkCudaErrors(cudaMemcpy(d_beta_cellfactor1, beta_cellfactor1_pointer, (dimension1_beta_cellfactor1*dimension2_beta_cellfactor1)*sizeof(float), cudaMemcpyHostToDevice));

	//== d_beta_cellfactor2_sub
	float * beta_cellfactor2_pointer = beta_cellfactor2.get_matrix_at(k);
	int dimension1_beta_cellfactor2 = beta_cellfactor2.get_dimension2();
	int dimension2_beta_cellfactor2 = beta_cellfactor2.get_dimension3();
	checkCudaErrors(cudaMemcpy(d_beta_cellfactor2_sub, beta_cellfactor2_pointer, (dimension1_beta_cellfactor2*dimension2_beta_cellfactor2)*sizeof(float), cudaMemcpyHostToDevice));
	


	//==##== collector ==##==
	X_sub.release();
	Z_sub.release();

	return;
}





void mem_gpu_destroytissue()
{
	checkCudaErrors(cudaFree(d_X_sub));
	checkCudaErrors(cudaFree(d_Z_sub));
	checkCudaErrors(cudaFree(d_Y_sub));
	checkCudaErrors(cudaFree(d_Y_sub_exp));

	checkCudaErrors(cudaFree(d_cellfactor_sub));
	checkCudaErrors(cudaFree(d_cellfactor_sub_new));

	return;
}


