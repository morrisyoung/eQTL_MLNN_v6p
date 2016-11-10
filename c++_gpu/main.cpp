#include <iostream>
#include <vector>
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include "global.h"
#include "library.h"
#include "mem_gpu_setup.h"
#include "data_interface.h"
#include "fbward_gd.h"
#include "cal_error.h"
#include "cal_error_test.h"




using namespace std;




//==== learning setting
// TODO: to determine some parameters from commond line (the following are default)
int num_iter_out = 100;
int num_iter_in = 100;
int size_batch = 20;
float rate_learn = 0.0000001;



//==== other indicators
int indicator_crossv = 1;		// this means we will have both training set and testing set loaded and reported







//==== to be filled later on by data loading program
int I = 0;						// num of SNPs
int J = 0;						// num of genes
int K = 0;						// num of tissues
int N = 0;						// num of individuals
int D = 0;						// num of cell factors
int B = 0;						// num of batch variables


//@@@@@@@@########@@@@@@@@
// we have the testing set
int N_test = 0;					// num of individuals (for testing)




//==== variables
// NOTE: here we assume one chromosome model (this is easily applicable for multiple-chromosome model)
Matrix X;						// matrix of Individuals x SNPs
Tensor_expr Y;					// tensor of gene expression
Map_list mapping_cis;			// list of (index start, index end)
Matrix Z;						// matrix of Individuals x Batches
// NOTE: the following have the intercept term
Tensor_beta_cis beta_cis;		// tensor of (imcomplete) matrix of Genes x cis- SNPs
Matrix beta_cellfactor1;		// matrix of first layer cell factor beta
Tensor beta_cellfactor2;		// tensor (tissue specific) of second layer cell factor beta
Matrix beta_batch;				// matrix of Individuals x Batches
// the following corresponds to the above
/* following useless in GPU update
Tensor_beta_cis der_cis;
Matrix der_cellfactor1;
Tensor der_cellfactor2;
Matrix der_batch;
*/
//@@@@@@@@########@@@@@@@@
// we have the testing set
Matrix X_test;					// matrix of Individuals x SNPs
Tensor_expr Y_test;				// tensor of gene expression
Matrix Z_test;					// matrix of Individuals x Batches





//==== error list
vector<float> list_error;
//@@@@@@@@########@@@@@@@@
// we have the testing set
vector<float> list_error_test;





//==== GPU memory variables
//
float * d_X_batch;
float * d_Z_batch;
float * d_Y_batch;
float * d_Y_batch_exp;
float * d_error_batch;
float * d_cellfactor_batch;
float * d_cellfactor_batch_new;
// cis- relevant
int * d_list_cis_start;						// NOTE: cis-
int * d_list_cis_end;						// NOTE: cis-
int * d_list_beta_cis_start;				// NOTE: cis-
int * d_list_beta_cis_geneindex;			// NOTE: cis-
int * d_list_indi_cis;						// NOTE: cis-
float * d_beta_cis_sub;						// NOTE: cis-

float * d_beta_batch;
float * d_beta_batch_reshape;

float * d_beta_cellfactor1;
float * d_beta_cellfactor1_reshape;

float * d_beta_cellfactor2_sub;
float * d_beta_cellfactor2_sub_reshape;

float * d_der_cis_sub;
float * d_der_batch;
float * d_der_cellfactor1;
float * d_der_cellfactor2_sub;
// the training dataset
float * d_X_sub;
float * d_Z_sub;
float * d_Y_sub;
float * d_Y_sub_exp;
float * d_cellfactor_sub;
float * d_cellfactor_sub_new;
//@@@@@@@@########@@@@@@@@
// we have the testing set
float * d_X_subtest;
float * d_Z_subtest;
float * d_Y_subtest;
float * d_Y_subtest_exp;
float * d_cellfactor_subtest;
float * d_cellfactor_subtest_new;






int main(int argc, char *argv[])
{
	cout << "now entering the sampling program..." << endl;


	//==== commond line parameters loading
	if(argc > 1)
	{
		num_iter_out = atoi(argv[1]);
		num_iter_in = atoi(argv[2]);
		size_batch = atoi(argv[3]);
		rate_learn = atof(argv[4]);

		cout << "filled-in learning parameters:" << endl;
		cout << "num_iter_out: " << num_iter_out << endl;
		cout << "num_iter_in: " << num_iter_in << endl;
		cout << "size_batch: " << size_batch << endl;
		cout << "rate_learn: " << rate_learn << endl;
	}



	//==== data loading, and data preparation (loading data always first of all)
	//data_load_simu();
	data_load_real();
	error_init();



	//==== pre-allocate some GPU memory
	mem_gpu_init();



	//============
	// train (mini-batch)
	//============
	for(int iter1=0; iter1<num_iter_out; iter1++)
	{
		cout << "[@@@]working on out iter#" << iter1 << endl;
		for(int k=0; k<K; k++)
		{
			// start current tissue
			// init gpu memory for this tissue
			mem_gpu_settissue(k);



			cout << "[##]" << iter1 << ", " << k << endl;
			for(int iter2=0; iter2<num_iter_in; iter2++)
			{
				cout << "[@]" << iter1 << ", " << k << ", " << iter2 << endl;

				//==== timer starts
				struct timeval time_start;
				struct timeval time_end;
				double time_diff;
				gettimeofday(&time_start, NULL);



				// DEBUG
				float error_before = cal_error(k);
				cout << "[error before] current (tissue) total error (trainging): " << error_before << endl;



				//========
				fbward_gd(k);
				float error = cal_error(k);
				cout << "current (tissue) total error (trainging): " << error << endl;
				list_error.push_back(error);
				//
				if(indicator_crossv)
				{
					//float error_test = cal_error_test(k);
					// DEBUG
					float error_test = 1.0;

					cout << "current (tissue) total error (testing): " << error_test << endl;
					list_error_test.push_back(error_test);
				}
				//
				error_save_online();








				//==== timer ends
				gettimeofday(&time_end, NULL);
				time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
				printf("time used for this mini-batch is %f seconds.\n", time_diff);
				cout << "####" << endl;
			}



			// release gpu memory for this tissue, and transfer back the parameters for this tissue
			mem_gpu_destroytissue(k);
			// end current tissue
		}


		//==== save the learned model (of all tissues)
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
		printf("saving model and error (of this full iter) done... it uses time %f seconds.\n", time_diff);


		// end current outer iter
	}



	//==== pre-allocate some GPU memory
	mem_gpu_release();




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


