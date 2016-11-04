// global.h
// function: global variables shared among routines

#ifndef GLOBAL_H
#define GLOBAL_H



#include <vector>
#include "library.h"





using namespace std;





//=================================================
extern int num_iter_out;
extern int num_iter_in;
extern int size_batch;
extern float rate_learn;



//=================================================
extern int I;						// num of SNPs
extern int J;						// num of genes
extern int K;						// num of tissues
extern int N;						// num of individuals
extern int D;						// num of cell factors
extern int B;						// num of batch variables



//=================================================
extern Matrix X;						// matrix of Individuals x SNPs
extern Tensor_expr Y;					// tensor of gene expression
extern Map_list mapping_cis;			// list of (index start, index end)
extern Matrix Z;						// matrix of Individuals x Batches
//
extern Tensor_beta_cis beta_cis;		// tensor of (imcomplete) matrix of Genes x cis- SNPs
extern Matrix beta_cellfactor1;		// matrix of first layer cell factor beta
extern Tensor beta_cellfactor2;		// tensor (tissue specific) of second layer cell factor beta
extern Matrix beta_batch;				// matrix of Individuals x Batches
//
/* useless in GPU update
extern Tensor_beta_cis der_cis;
extern Matrix der_cellfactor1;
extern Tensor der_cellfactor2;
extern Matrix der_batch;
*/




//=================================================
extern vector<float> list_error;






// GPU variables
//=================================================
//
extern float * d_X_batch;
extern float * d_Z_batch;
extern float * d_Y_batch;
extern float * d_Y_batch_exp;
extern float * d_error_batch;
extern float * d_cellfactor_batch;
extern float * d_cellfactor_batch_new;

extern int * d_list_cis_start;
extern int * d_list_cis_end;
extern int * d_list_beta_cis_start;
extern int * d_list_beta_cis_geneindex;
extern float * d_beta_cis_sub;

extern float * d_beta_batch;
extern float * d_beta_batch_reshape;

extern float * d_beta_cellfactor1;
extern float * d_beta_cellfactor1_reshape;

extern float * d_beta_cellfactor2_sub;
extern float * d_beta_cellfactor2_sub_reshape;

extern float * d_der_cis_sub;
extern float * d_der_batch;
extern float * d_der_cellfactor1;
extern float * d_der_cellfactor2_sub;
//
extern float * d_X_sub;
extern float * d_Z_sub;
extern float * d_Y_sub;
extern float * d_Y_sub_exp;
extern float * d_cellfactor_sub;
extern float * d_cellfactor_sub_new;






//@@@@@@@@########@@@@@@@@
// we have the testing set
extern int N_test;

extern Matrix X_test;
extern Tensor_expr Y_test;
extern Matrix Z_test;

extern vector<float> list_error_test;

extern float * d_X_subtest;
extern float * d_Z_subtest;
extern float * d_Y_subtest;
extern float * d_Y_subtest_exp;
extern float * d_cellfactor_subtest;
extern float * d_cellfactor_subtest_new;






//==== other indicators
extern int indicator_crossv;





#endif

// end of global.h

