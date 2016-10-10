#include <iostream>
#include <vector>
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include "global.h"
#include "library.h"



using namespace std;



//==== learning setting
// TODO: to determine some parameters
int num_iter_out = 100;
int num_iter_in = 100;
int size_batch = 20;
float rate_learn = 0.0000001;


//==== to be filled later on
int I = 0;						// num of SNPs
int J = 0;						// num of genes
int K = 0;						// num of tissues
int N = 0;						// num of individuals
int D = 0;						// num of cell factors
int B = 0;						// num of batch variables


//==== variables
// NOTE: here we assume one chromosome model (this is easily applicable for multiple-chromosome model)
Maatrix X;						// matrix of Individuals x SNPs
Tensor_expr Y;					// tensor of gene expression
Map_list mapping_cis;			// list of (index start, index end)
Matrix Z;						// matrix of Individuals x Batches
// NOTE: the following have the intercept term
Tensor_beta_cis beta_cis;		// tensor of (imcomplete) matrix of Genes x cis- SNPs
Matrix beta_cellfactor1;		// matrix of first layer cell factor beta
Tensor beta_cellfactor2;		// tensor (tissue specific) of second layer cell factor beta
Matrix beta_batch;				// matrix of Individuals x Batches
// the following corresponds to the above
Tensor_beta_cis der_cis;
Matrix der_cellfactor1;
Tensor der_cellfactor2;
Matrix der_batch;




//==== error list
vector<float> list_error;








int main()
{
	cout << "now entering the sampling program..." << endl;


	//==== data loading, and data preparation
	data_load_simu();
	//data_load_real();
	error_init();


	//============
	// train (mini-batch)
	//============
	for(int iter1=0; iter1<num_iter_out; iter1++)
	{
		cout << "[@@@]working on out iter#" << iter1 << endl;
		for(int k=0; k<K; k++)
		{
			cout << "[##]" << iter1 << ", " << k << endl;
			for(int iter2=0; iter2<num_iter_in; iter2++)
			{
				cout << "[@]" << iter1 << ", " << k << ", " << iter2 << endl;

				//==== timer starts
				struct timeval time_start;
				struct timeval time_end;
				double time_diff;
				gettimeofday(&time_start, NULL);


				//========
				fbward_gd(k);
				float error = cal_error(k);
				cout << "current (tissue) total error:" << error << endl;

				list_error.push_back(error);
				error_save_online();


				//==== timer ends
				gettimeofday(&time_end, NULL);
				time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
				printf("time used for this mini-batch is %f seconds.\n", time_diff);
				cout << "####" << endl;


				// TODO: save the model per need (save model, save error)
			}
			// end current tissue
		}
		// end current outer iter
	}



	//==== save the learned model
	//==== timer starts
	struct timeval time_start;
	struct timeval time_end;
	double time_diff;
	gettimeofday(&time_start, NULL);

	model_save();
	error_save();

	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("saving model and error done... it uses time %f seconds.\n", time_diff);









	cout << "we are done..." << endl;

	return 0;
}

