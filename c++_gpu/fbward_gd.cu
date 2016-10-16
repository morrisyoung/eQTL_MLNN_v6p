// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <random>
#include <chrono>		/* sys time */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <math.h>       /* sqrt */

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// private
#include "global.h"
#include "library.h"
#include "fbward_gd.h"
#include "utility_gpu.cuh"





using namespace std;




// to fill: d_X_batch, d_Z_batch, d_Y_batch
void batch_fill(int k)
{
	int size_sub = Y.get_dimension2_at(k);
	srand(unsigned(time(0)));
	vector<int> indexvec;
	for(int i=0; i<size_sub; ++i)
	{
		indexvec.push_back(i);
	}
	// using built-in random generator:
	random_shuffle(indexvec.begin(), indexvec.end());


	int * list_indiv_pos = (int *)calloc( size_batch, sizeof(int));
	int * list_sample_pos = (int *)calloc( size_batch, sizeof(int));
	for(int i=0; i<size_batch; i++)
	{
		int pos = indexvec.at(i);
		list_indiv_pos[i] = Y.get_indiv_pos_at(k, pos);
		list_sample_pos[i] = pos;
	}


	//===============================================================
	//==== init and transmit tissue data when starting a new tissue
	//===============================================================
	int dimension1 = size_batch;
	int dimension2 = J;
	int dimension;

	//== X_batch, d_X_batch
	Matrix X_batch;
	dimension = X.get_dimension2();
	X_batch.init(dimension1, dimension);
	float * X_pointer = X.get_pointer();
	X_batch.fill_with_ref_list(list_indiv_pos, X_pointer);
	//
	float * X_batch_pointer = X_batch.get_pointer();
	checkCudaErrors(cudaMemcpy(d_X_batch, X_batch_pointer, (dimension1*dimension)*sizeof(float), cudaMemcpyHostToDevice));

	//== Z_batch, d_Z_batch
	Matrix Z_batch;
	dimension = Z.get_dimension2();
	Z_batch.init(dimension1, dimension);
	float * Z_pointer = Z.get_pointer();
	Z_batch.fill_with_ref_list(list_indiv_pos, Z_pointer);
	//
	float * Z_batch_pointer = Z_batch.get_pointer();
	checkCudaErrors(cudaMemcpy(d_Z_batch, Z_batch_pointer, (dimension1*dimension)*sizeof(float), cudaMemcpyHostToDevice));

	//== Y_batch, d_Y_batch
	Matrix Y_batch;
	Y_batch.init(dimension1, dimension2);
	float * Y_pointer = Y.get_matrix_at(k);
	Y_batch.fill_with_ref_list(list_sample_pos, Y_pointer);
	//
	float * Y_batch_pointer = Y_batch.get_pointer();
	checkCudaErrors(cudaMemcpy(d_Y_batch, Y_batch_pointer, (dimension1*dimension2)*sizeof(float), cudaMemcpyHostToDevice));



	//==== delete local heap
	free(list_indiv_pos);
	free(list_sample_pos);
	X_batch.release();
	Z_batch.release();
	Y_batch.release();

	return;
}





// do in the tissue k: forward/backward propogation, gradient descent
void fbward_gd(int k)
{
	// fill in the following:
	//	d_X_batch, d_Z_batch, d_Y_batch
	// and we have the following containers in GPU:
	//	d_cellfactor_batch, d_cellfactor_batch_new
	//	d_der_cis_sub, d_der_batch, d_der_cellfactor1, d_der_cellfactor2_sub
	batch_fill(k);






	int dimension1 = size_batch;
	int dimension2 = J;


	//==========================
	// from cis- (tissue k)
	//==========================
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// cis- matrux mul
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_cal_cis_matrixmul<32><<< grid, threads >>>(d_Y_batch_exp, dimension1, dimension2, d_X_batch, X.get_dimension2(), d_list_cis_start, d_list_cis_end, d_beta_cis_sub, d_list_beta_cis_start);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==




	//==========================
	// from batch
	//==========================
	int dimension1_beta_batch = beta_batch.get_dimension1();
	int dimension2_beta_batch = beta_batch.get_dimension2();
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// transpose
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2_beta_batch+threads.x-1)/threads.x, (dimension1_beta_batch+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(dimension1_beta_batch, dimension2_beta_batch, d_beta_batch, d_beta_batch_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// matrix mul
	// func: d_Y_batch_exp += d_Z_batch x d_beta_batch_reshape;
	// dimension: (dimension1, dimension2) += (dimension1, dimension2_beta_batch) x (dimension2_beta_batch, dimension2)
	{
		const float alpha = 1.0f;
		const float beta  = 1.0f;									// NOTE: we always do cis- first with customer kernel, and then add on expression from batch and cell factors
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension2, dimension1, dimension2_beta_batch, &alpha, d_beta_batch_reshape, dimension2, d_Z_batch, dimension2_beta_batch, &beta, d_Y_batch_exp, dimension2));
		checkCudaErrors(cublasDestroy(handle));
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==




	//=============
	// from cell factor (tissue k)
	//=============
	// first layer
	int dimension1_beta_cellfactor1 = beta_cellfactor1.get_dimension1();
	int dimension2_beta_cellfactor1 = beta_cellfactor1.get_dimension2();
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// transpose
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2_beta_cellfactor1+threads.x-1)/threads.x, (dimension1_beta_cellfactor1+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(dimension1_beta_cellfactor1, dimension2_beta_cellfactor1, d_beta_cellfactor1, d_beta_cellfactor1_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// matrix mul
	// func: d_cellfactor_batch = d_X_batch x d_beta_cellfactor1_reshape;
	// dimension: (dimension1, dimension1_beta_cellfactor1) = (dimension1, dimension2_beta_cellfactor1) x (dimension2_beta_cellfactor1, dimension1_beta_cellfactor1)
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;									// NOTE: add, other than add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension1_beta_cellfactor1, dimension1, dimension2_beta_cellfactor1, &alpha, d_beta_cellfactor1_reshape, dimension1_beta_cellfactor1, d_X_batch, dimension2_beta_cellfactor1, &beta, d_cellfactor_batch, dimension1_beta_cellfactor1));
		checkCudaErrors(cublasDestroy(handle));
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	// logistic twist: d_cellfactor_batch
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension1_beta_cellfactor1+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_cal_matrix_logistic<32><<< grid, threads >>>(dimension1, dimension1_beta_cellfactor1, d_cellfactor_batch);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// append one
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( ((dimension1_beta_cellfactor1+1)+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_op_matrix_appendone<32><<< grid, threads >>>(dimension1, dimension1_beta_cellfactor1+1, d_cellfactor_batch_new, d_cellfactor_batch);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	// second layer
	int dimension1_beta_cellfactor2 = beta_cellfactor2.get_dimension2();
	int dimension2_beta_cellfactor2 = beta_cellfactor2.get_dimension3();
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// transpose
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2_beta_cellfactor2+threads.x-1)/threads.x, (dimension1_beta_cellfactor2+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(dimension1_beta_cellfactor2, dimension2_beta_cellfactor2, d_beta_cellfactor2_sub, d_beta_cellfactor2_sub_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// matrix mul
	// func: d_Y_batch_exp += d_cellfactor_batch_new x d_beta_cellfactor2_sub_reshape;
	// dimension: (dimension1, dimension2) += (dimension1, dimension2_beta_cellfactor2) x (dimension2_beta_cellfactor2, dimension2)
	{
		const float alpha = 1.0f;
		const float beta  = 1.0f;									// NOTE: add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension2, dimension1, dimension2_beta_cellfactor2, &alpha, d_beta_cellfactor2_sub_reshape, dimension2, d_cellfactor_batch_new, dimension2_beta_cellfactor2, &beta, d_Y_batch_exp, dimension2));
		checkCudaErrors(cublasDestroy(handle));
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==





	//=============
	// error cal
	//=============
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (J+threads.x-1)/threads.x, (size_batch+threads.y-1)/threads.y );
		kernel_cal_subtract<32><<< grid, threads >>>(size_batch, J, d_error_batch, d_Y_batch_exp, d_Y_batch);
	}








	// above: forward-propogation
	//=============/=============/=============/=============/=============/=============/=============/=============
	//=============/=============/=============/=============/=============/=============/=============/=============
	// below: back-propagation
	//=============
	// from cis- (tissue k)
	//=============
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// cis- bp
	{
		int amount = beta_cis.get_amount();
		int block_size = 32;
		dim3 threads(block_size, block_size);
		int d_square = int(sqrt(amount)) + 1;	// re-shape to a square matrix slightly larger than the current array
		dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );
		kernel_cal_bp_cis<32><<< grid, threads >>>(size_batch, J, d_error_batch, (I+1), d_X_batch, d_list_cis_start, d_list_cis_end, amount, d_der_cis_sub, d_list_beta_cis_start, d_list_beta_cis_geneindex);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==




	//=============
	// from batch
	//=============
	//== d_error_batch_reshape
	float * d_error_batch_reshape;
	checkCudaErrors(cudaMalloc((void **) &d_error_batch_reshape, (size_batch*J)*sizeof(float)));
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (J+threads.x-1)/threads.x, (size_batch+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(size_batch, J, d_error_batch, d_error_batch_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// func: d_der_batch = (d_error_batch_reshape x d_Z_batch) / size_batch;
	// matrix mul: J x N, N x (B+1)
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;									// NOTE: non add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (B+1), J, size_batch, &alpha, d_Z_batch, (B+1), d_error_batch_reshape, size_batch, &beta, d_der_batch, (B+1)));
		checkCudaErrors(cublasDestroy(handle));

		// scale
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( ((B+1)+threads.x-1)/threads.x, (J+threads.y-1)/threads.y );
		kernel_op_matrix_scale<32><<< grid, threads >>>(J, B+1, d_der_batch, size_batch);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==





	//=============
	// from cell factor (tissue k)
	//=============
	//== last layer
	//== d_error_batch_reshape --> we have this
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// func: d_der_cellfactor2_sub = (d_error_batch_reshape x d_cellfactor_batch_new) / size_batch;
	// J x N, N x (D+1)
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;									// NOTE: non add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (D+1), J, size_batch, &alpha, d_cellfactor_batch_new, (D+1), d_error_batch_reshape, size_batch, &beta, d_der_cellfactor2_sub, (D+1)));
		checkCudaErrors(cublasDestroy(handle));

		// scale
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( ((D+1)+threads.x-1)/threads.x, (J+threads.y-1)/threads.y );
		kernel_op_matrix_scale<32><<< grid, threads >>>(J, D+1, d_der_cellfactor2_sub, size_batch);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==


	//== first layer
	float * d_temp_more;
	float * d_temp;
	float * d_temp_reshape;
	float * d_cellfactor_batch_der;
	checkCudaErrors(cudaMalloc((void **) &d_temp_more, (size_batch*(D+1))*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_temp, (size_batch*D)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_temp_reshape, (size_batch*D)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_cellfactor_batch_der, (size_batch*D)*sizeof(float)));
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// func: d_temp_more = d_error_batch x d_beta_cellfactor2_sub
	// N x J, J x (D+1) --> N x (D+1)
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;									// NOTE: non add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (D+1), size_batch, J, &alpha, d_beta_cellfactor2_sub, (D+1), d_error_batch, J, &beta, d_temp_more, (D+1)));
		checkCudaErrors(cublasDestroy(handle));
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// cut last column:
	// d_temp_more --> d_temp
	// N x (D+1) --> N x D
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (D+threads.x-1)/threads.x, (size_batch+threads.y-1)/threads.y );
		kernel_op_matrix_cutone<32><<< grid, threads >>>(size_batch, D, d_temp, d_temp_more);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// der_logit:
	// d_cellfactor_batch_der = d_cellfactor_batch * (1 - d_cellfactor_batch)
	// N x D
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (D+threads.x-1)/threads.x, (size_batch+threads.y-1)/threads.y );
		kernel_cal_logit_der<32><<< grid, threads >>>(size_batch, D, d_cellfactor_batch_der, d_cellfactor_batch);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// multiply:
	// d_temp = d_temp * d_cellfactor_batch_der
	// N x D
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (D+threads.x-1)/threads.x, (size_batch+threads.y-1)/threads.y );
		kernel_cal_matrix_multion<32><<< grid, threads >>>(size_batch, D, d_temp, d_cellfactor_batch_der);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// reshape: d_temp (size_batch, D)
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (D+threads.x-1)/threads.x, (size_batch+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(size_batch, D, d_temp, d_temp_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==



	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// dot and scale
	// func: d_der_cellfactor1 = (d_temp_reshape x d_X_batch) / size_batch;
	// matrix mul: D x size_batch, size_batch x (I+1) --> D x (I+1)
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;									// NOTE: non add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (I+1), D, size_batch, &alpha, d_X_batch, (I+1), d_temp_reshape, size_batch, &beta, d_der_cellfactor1, (I+1)));
		checkCudaErrors(cublasDestroy(handle));

		// scale
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( ((I+1)+threads.x-1)/threads.x, (D+threads.y-1)/threads.y );
		kernel_op_matrix_scale<32><<< grid, threads >>>(D, I+1, d_der_cellfactor1, size_batch);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==


	//====//====//====//====//====//
	checkCudaErrors(cudaFree(d_error_batch_reshape));
	checkCudaErrors(cudaFree(d_temp_more));
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(d_cellfactor_batch_der));
	checkCudaErrors(cudaFree(d_temp_reshape));











	//=============/=============/=============/=============/=============/=============/=============/=============
	//=============/=============/=============/=============/=============/=============/=============/=============
	// gradient descent
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// d_beta_cis_sub
	{
		int amount = beta_cis.get_amount();
		//
		int block_size = 32;
		dim3 threads(block_size, block_size);
		int d_square = int(sqrt(amount)) + 1;	// re-shape to a square matrix slightly larger than the current array
		dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );
		kernel_cal_gd_array<32><<< grid, threads >>>(amount, d_beta_cis_sub, d_der_cis_sub, rate_learn);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// d_beta_cellfactor1
	{
		int dimension1 = beta_cellfactor1.get_dimension1();
		int dimension2 = beta_cellfactor1.get_dimension2();
		//
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_cal_gd_matrix<32><<< grid, threads >>>(dimension1, dimension2, d_beta_cellfactor1, d_der_cellfactor1, rate_learn);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// d_beta_batch
	{
		int dimension1 = beta_batch.get_dimension1();
		int dimension2 = beta_batch.get_dimension2();
		//
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_cal_gd_matrix<32><<< grid, threads >>>(dimension1, dimension2, d_beta_batch, d_der_batch, rate_learn);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// d_beta_cellfactor2_sub
	{
		int dimension1 = beta_cellfactor2.get_dimension2();
		int dimension2 = beta_cellfactor2.get_dimension3();
		//
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_cal_gd_matrix<32><<< grid, threads >>>(dimension1, dimension2, d_beta_cellfactor2_sub, d_der_cellfactor2_sub, rate_learn);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==





	return;
}



