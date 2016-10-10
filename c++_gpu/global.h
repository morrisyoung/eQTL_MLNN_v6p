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
extern Maatrix X;						// matrix of Individuals x SNPs
extern Tensor_expr Y;					// tensor of gene expression
extern Map_list mapping_cis;			// list of (index start, index end)
extern Matrix Z;						// matrix of Individuals x Batches
//
extern Tensor_beta_cis beta_cis;		// tensor of (imcomplete) matrix of Genes x cis- SNPs
extern Matrix beta_cellfactor1;		// matrix of first layer cell factor beta
extern Tensor beta_cellfactor2;		// tensor (tissue specific) of second layer cell factor beta
extern Matrix beta_batch;				// matrix of Individuals x Batches
//
extern Tensor_beta_cis der_cis;
extern Matrix der_cellfactor1;
extern Tensor der_cellfactor2;
extern Matrix der_batch;




//=================================================
extern vector<float> list_error;








#endif

// end of global.h
