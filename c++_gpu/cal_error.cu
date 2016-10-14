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
#include "cal_error.h"
#include "global.h"
#include "library.h"
#include "utility_gpu.cuh"




using namespace std;




// calculate the total squared error for specified tissue
float cal_error(int k)
{
	float error = 0;
	int dimension1 = Y.get_dimension2_at(k);
	int dimension2 = J;


	//==========================
	// from cis- (tissue k)
	//==========================
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_cal_cis_matrixmul<32><<< grid, threads >>>(d_Y_sub_exp, dimension1, dimension2, d_X_sub, X.get_dimension2(), d_list_cis_start, d_list_cis_end, d_beta_cis_sub, d_list_beta_cis_start);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==




	//==========================
	// from batch
	//==========================
	int dimension1_beta_batch = beta_batch.get_dimension1();
	int dimension2_beta_batch = beta_batch.get_dimension2();
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2_beta_batch+threads.x-1)/threads.x, (dimension1_beta_batch+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(dimension1_beta_batch, dimension2_beta_batch, d_beta_batch, d_beta_batch_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// func: d_Y_sub_exp += d_Z_sub x d_beta_batch_reshape;
	// dimension: (dimension1, dimension2) += (dimension1, dimension2_beta_batch) x (dimension2_beta_batch, dimension2)
	{
		const float alpha = 1.0f;
		const float beta  = 1.0f;									// NOTE: we always do cis- first with customer kernel, and then add on expression from batch and cell factors
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension2, dimension1, dimension2_beta_batch, &alpha, d_beta_batch_reshape, dimension2, d_Z_sub, dimension2_beta_batch, &beta, d_Y_sub_exp, dimension2));
		checkCudaErrors(cublasDestroy(handle));
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==





	//=============
	// from cell factor (tissue k)
	//=============
	int dimension1_beta_cellfactor1 = beta_cellfactor1.get_dimension1();
	int dimension2_beta_cellfactor1 = beta_cellfactor1.get_dimension2();
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2_beta_cellfactor1+threads.x-1)/threads.x, (dimension1_beta_cellfactor1+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(dimension1_beta_cellfactor1, dimension2_beta_cellfactor1, d_beta_cellfactor1, d_beta_cellfactor1_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	// first layer
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// func: d_cellfactor_sub = d_X_sub x d_beta_cellfactor1_reshape;
	// dimension: (dimension1, dimension1_beta_cellfactor1) += (dimension1, dimension2_beta_cellfactor1) x (dimension2_beta_cellfactor1, dimension1_beta_cellfactor1)
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;									// NOTE: add, other than add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension1_beta_cellfactor1, dimension1, dimension2_beta_cellfactor1, &alpha, d_beta_cellfactor1_reshape, dimension1_beta_cellfactor1, d_X_sub, dimension2_beta_cellfactor1, &beta, d_cellfactor_sub, dimension1_beta_cellfactor1));
		checkCudaErrors(cublasDestroy(handle));
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	// logistic twist: d_cellfactor_sub
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension1_beta_cellfactor1+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_cal_matrix_logistic<32><<< grid, threads >>>(dimension1, dimension1_beta_cellfactor1, d_cellfactor_sub);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( ((dimension1_beta_cellfactor1+1)+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y );
		kernel_op_matrix_appendone<32><<< grid, threads >>>(dimension1, dimension1_beta_cellfactor1+1, d_cellfactor_sub_new, d_cellfactor_sub);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	int dimension1_beta_cellfactor2 = beta_cellfactor2.get_dimension2();
	int dimension2_beta_cellfactor2 = beta_cellfactor2.get_dimension3();
	{
		int block_size = 32;
		dim3 threads(block_size, block_size);
		dim3 grid( (dimension2_beta_cellfactor2+threads.x-1)/threads.x, (dimension1_beta_cellfactor2+threads.y-1)/threads.y );
		kernel_op_matrix_reshape<32><<< grid, threads >>>(dimension1_beta_cellfactor2, dimension2_beta_cellfactor2, d_beta_cellfactor2_sub, d_beta_cellfactor2_sub_reshape);
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==

	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==
	// func: d_Y_sub_exp += d_cellfactor_sub_new x d_beta_cellfactor2_sub_reshape;
	// dimension: (dimension1, dimension2) += (dimension1, dimension2_beta_cellfactor2) x (dimension2_beta_cellfactor2, dimension2)
	{
		const float alpha = 1.0f;
		const float beta  = 1.0f;									// NOTE: add-on
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		//note cublas is column primary! need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension2, dimension1, dimension2_beta_cellfactor2, &alpha, d_beta_cellfactor2_sub_reshape, dimension2, d_cellfactor_sub_new, dimension2_beta_cellfactor2, &beta, d_Y_sub_exp, dimension2));
		checkCudaErrors(cublasDestroy(handle));
	}
	//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==






	//=============
	// compile and error cal
	//=============
	{
		// two steps: sub sum; sum sub
		//== d_sumY_temp, d_sum
		float * d_sumY_temp;
		int sub_amount = 200;									// TODO: to tune this number
		int sub_length = (dimension1*dimension2 + sub_amount-1) / sub_amount;
		checkCudaErrors(cudaMalloc((void **) &d_sumY_temp, sub_length*sizeof(float)));
		//
		float * d_sum;
		checkCudaErrors(cudaMalloc((void **) &d_sum, 1*sizeof(float)));
		float h_sum;

		int block_size = 32;
		dim3 threads(block_size);
		dim3 grid( (sub_length+threads.x-1)/threads.x );
		//
		kernel_cal_sosod_subsum<32><<< grid, threads >>>(sub_amount, sub_length, dimension1*dimension2, d_sumY_temp, d_Y_sub_exp, d_Y_sub);
		//
		kernel_cal_sosod_sumsub<32><<< grid, threads >>>(sub_length, d_sumY_temp, d_sum);
		//
		checkCudaErrors(cudaMemcpy(h_sum, d_sum, 1*sizeof(float), cudaMemcpyDeviceToHost));
		error = h_sum;

		//==##== collector ==##==
		checkCudaErrors(cudaFree(d_sumY_temp));
		checkCudaErrors(cudaFree(d_sum));
	}





	return error;
}



