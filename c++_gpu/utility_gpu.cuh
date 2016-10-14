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






using namespace std;







template <int BLOCK_SIZE> __global__ void
kernel_cal_cis_matrixmul(float * d_Y_sub_exp, int dimension1, int dimension2, float * d_X_sub, int dimension2_X, int * d_list_cis_start, int * d_list_cis_end, float * d_beta_cis_sub, int * d_list_beta_cis_start)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)						// (i, j) is one (sample, gene) point
	{
		int cis_start = d_list_cis_start[j];
		int cis_end = d_list_cis_end[j];
		int amount = cis_end - start + 1;
		int snp_start = i*dimension2_X + start;
		int coef_start = d_list_beta_cis_start[j];
		//
		float result = 0;
		for(int k=0; k<amount; k++)
		{
			float dosage = d_X_sub[snp_start + k];
			float coef = d_beta_cis_sub[coef_start + k];
			result += dosage * coef;
		}
		result += 1*d_beta_cis_sub[coef_start+amount];		// the intercept
		//
		int pos = i*dimension2 + j;
		d_Y_sub_exp[pos] = result;
	}

	return;
}





template <int BLOCK_SIZE> __global__ void
kernel_op_matrix_reshape(int dimension1, int dimension2, float * m_input, float * m_result)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)						// (i, j) is one (sample, gene) point
	{
		int pos_old = i * dimension2 + j;
		int pos_new = j * dimension1 + i;
		m_result[pos_new] = m_input[pos_old];
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_cal_matrix_logistic(int dimension1, int dimension2, float * matrix)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)						// (i, j) is one (sample, gene) point
	{
		int pos = i * dimension2 + j;
		float x = matrix[pos];
		matrix[pos] = 1.0 / (1.0 + expf(-x));				// expf() is for single precision float point
	}

	return;
}





template <int BLOCK_SIZE> __global__ void
kernel_op_matrix_appendone(int dimension1, int dimension2, float * matrix_new, float * matrix_ref)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)						// (i, j) is one (sample, gene) point
	{
		int pos_new = i*dimension2+j;
		if(j==(dimension2-1))
		{
			matrix_new[pos_new] = 1;
		}
		else
		{
			int pos_ref = i*(dimension2-1)+j;
			float x = matrix_ref[pos_ref];
			matrix_new[pos_new] = x;
		}
	}

	return;
}





template <int BLOCK_SIZE> __global__ void
kernel_cal_sosod_subsum(int sub_amount, int sub_length, int amount, float * d_sumY_temp, float * d_Y_sub_exp, float * d_Y_sub)
{
	// Block index
	long int bx = blockIdx.x;

	// Thread index
	long int tx = threadIdx.x;

	// indexing in the matrix
	long int i = bx * blockDim.x + tx;

	// boundary check
	if(i<sub_length)
	{
		if(i==(sub_length-1))				// not sub_amount totally
		{
			float sum = 0;
			for(int pos=i*sub_amount; pos<amount; pos++)
			{
				float temp = d_Y_sub_exp[pos] - d_Y_sub[pos];
				sum += temp * temp;
			}
			d_sumY_temp[i] = sum;
		}
		else								// sub_amount totally
		{
			int pos_start = i*sub_amount;
			float sum = 0;
			for(int k=0; k<sub_amount; k++)
			{
				float temp = d_Y_sub_exp[pos_start+k] - d_Y_sub[pos_start+k];
				sum += temp * temp;
			}
			d_sumY_temp[i] = sum;
		}
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_cal_sosod_sumsub(int sub_length, float * d_sumY_temp, float * d_sum)
{
	// Block index
	long int bx = blockIdx.x;

	// Thread index
	long int tx = threadIdx.x;

	// indexing in the matrix
	long int i = bx * blockDim.x + tx;

	// boundary check
	if(i==0)
	{
		float sum = 0;
		for(int k=0; k<sub_length; k++)
		{
			sum += d_sumY_temp[k];
		}
		d_sum = sum;
	}

	return;
}





// subtract matrix2 from matrix1, and save into result
template <int BLOCK_SIZE> __global__ void
kernel_cal_subtract(int dimension1, int dimension2, float * result, float * matrix1, float * matrix2)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)						// (i, j) is one (sample, gene) point
	{
		int pos = i*dimension2+j;
		result[pos] = matrix1[pos] - matrix2[pos];
	}

	return;
}








//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
// back-propagation routines



template <int BLOCK_SIZE> __global__ void
kernel_cal_bp_cis(int size_batch, int dimension2_Y, float * d_error_batch, int dimension2_X, float * d_X_batch, int * d_list_cis_start, int * d_list_cis_end, int amount_para, float * d_der_cis_sub, int * d_list_beta_cis_start, int * d_list_beta_cis_geneindex)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;
	long int pos = i * blockDim.x * gridDim.x + j;


	// reference:
	/*
	//=============
	// from cis- (tissue k)
	//=============
	for j in range(J):
		//der_cis[k][j] = np.zeros(der_cis[k][j].shape)
		X_sub = []
		start = mapping_cis[j][0]
		end = mapping_cis[j][1]
		X_sub = X_batch[:, start:end+1]
		array_ones = (np.array([np.ones(size_batch)])).T
		X_sub = np.concatenate((X_sub, array_ones), axis=1)						// size_batch x (amount+1)

		der_cis[k][j] = np.dot(m_error[:, j].T, X_sub)
		der_cis[k][j] = der_cis[k][j] / size_batch
	*/

	// boundary check
	if(pos < amount_para)
	{
		int geneindex = d_list_beta_cis_geneindex[pos];
		int cis_start = d_list_cis_start[geneindex];
		int cis_end = d_list_cis_end[geneindex];
		int cis_amount = cis_end - cis_start + 1;
		int beta_start = d_list_beta_cis_start[geneindex];
		int beta_num = pos - beta_start + 1;
		if(beta_num == (cis_amount + 1))					// the intercept
		{
			float result = 0;
			for(int k=0; k<size_batch; k++)
			{
				float dosage = 1;
				float error = d_error_batch[geneindex + k*dimension2_Y];
				result += dosage * error;
			}
			result = result / size_batch;
			d_der_cis_sub[pos] = result;
		}
		else 												// normal snp
		{
			int snp_start = cis_start + (beta_num - 1);
			float result = 0;
			for(int k=0; k<size_batch; k++)
			{
				float dosage = d_X_batch[snp_start + k*dimension2_X];
				float error = d_error_batch[geneindex + k*dimension2_Y];
				result += dosage * error;
			}
			result = result / size_batch;
			d_der_cis_sub[pos] = result;
		}
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_op_matrix_scale(int dimension1, int dimension2, float * matrix, int size_batch)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)
	{
		int pos = i*dimension2+j;
		matrix[pos] = matrix[pos] / size_batch;
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_op_matrix_cutone(int dimension1, int dimension2, float * matrix_new, float * matrix_old)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)
	{
		int pos_new = i*dimension2+j;
		int pos_old = i*(dimension2+1)+j;
		matrix_new[pos_new] = matrix_old[pos_old];
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_cal_logit_der(int dimension1, int dimension2, float * result, float * input)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)
	{
		int pos = i*dimension2+j;
		result[pos] = input[pos]*(1-input[pos]);
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_cal_matrix_multion(int dimension1, int dimension2, float * result, float * input)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)
	{
		int pos = i*dimension2+j;
		result[pos] = result[pos] * input[pos];
	}

	return;
}





//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
// GD routines



// gd array
template <int BLOCK_SIZE> __global__ void
kernel_cal_gd_array(int amount, float * array_beta, float * array_der, float rate_learn)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;
	long int pos = i * blockDim.x * gridDim.x + j;

	// boundary check
	if(pos < amount)
	{
		array_beta[pos] = array_beta[pos] - rate_learn * array_der[pos];
	}

	return;
}




// gd matrix
template <int BLOCK_SIZE> __global__ void
kernel_cal_gd_matrix(int dimension1, int dimension2, float * matrix_beta, float * matrix_der, float rate_learn)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the matrix
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// boundary check
	if(i<dimension1 && j<dimension2)						// (i, j) is one (sample, gene) point
	{
		int pos = i*dimension2+j;
		matrix_beta[pos] = matrix_beta[pos] - rate_learn * matrix_der[pos];
	}

	return;
}





