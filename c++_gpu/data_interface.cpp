#include <iostream>
#include <string>		/* stof, stod */
#include <stdlib.h>     /* atoi */
#include "global.h"
#include "data_interface.h"
#include "lib_io_file.h"
#include "lib_op_line.h"




using namespace std;




// loading the Matrix matrix with file name as char * filename
// for a matrix, the format is straightforward, that we only need to save the matrix as they are
void load_matrix(Matrix & matrix, char * filename)
{
	//==== load data into temporary container
	vector<vector<float>> container_temp;

	char type[10] = "r";
	filehandle file(filename, type);

	long input_length = 1000000000;
	char * line = (char *)malloc( sizeof(char) * input_length );
	while(1)
	{
		int end = file.readline(line, input_length);
		if(end)
			break;

		line_class line_obj(line);
		line_obj.split_tab();
		vector<float> vec;
		for(unsigned i=0; i<line_obj.size(); i++)
		{
			char * pointer = line_obj.at(i);
			//float value = stof(pointer);			// NOTE: there are double-range numbers
			float value = stod(pointer);
			vec.push_back(value);
		}
		line_obj.release();

		container_temp.push_back(vec);
	}
	free(line);
	file.close();


	//==== load data into Matrix from temporary container
	matrix.init(container_temp);

	return;
}



// load a tensor into Tensor tensor, with filename as char * filename
// since I want to keep the tensor as a whole, other than splitting them into sub-files, I will use meta info (first line, shape of tensor)
void load_tensor(Tensor & tensor, char * filename)
{
	char type[10] = "r";
	filehandle file(filename, type);

	long input_length = 1000000000;
	char * line = (char *)malloc( sizeof(char) * input_length );


	//==== first, get tensor shape
	int dimension1 = 0;
	int dimension2 = 0;
	int dimension3 = 0;

	file.readline(line, input_length);
	line_class line_obj(line);
	line_obj.split_tab();
	char * pointer;
	//== d1
	pointer = line_obj.at(0);
	dimension1 = atoi(pointer);
	//== d2
	pointer = line_obj.at(1);
	dimension2 = atoi(pointer);
	//== d3
	pointer = line_obj.at(2);
	dimension3 = atoi(pointer);

	line_obj.release();


	//==== then, load data into temporary container
	vector<vector<vector<float>>> container_temp;

	for(int i=0; i<dimension1; i++)
	{
		vector<vector<float>> vec;
		container_temp.push_back(vec);

		for(int j=0; j<dimension2; j++)
		{
			int end = file.readline(line, input_length);

			line_class line_obj(line);
			line_obj.split_tab();

			vector<float> vec;
			for(unsigned i=0; i<line_obj.size(); i++)
			{
				char * pointer = line_obj.at(i);
				//float value = stof(pointer);				// there are double-range numbers
				float value = stod(pointer);
				vec.push_back(value);
			}
			line_obj.release();
			(container_temp.at(i)).push_back(vec);

		}
	}

	free(line);
	file.close();


	//==== load data into Tensor from temporary container
	tensor.init(container_temp);


	return;
}






//// loading the simulated data (I'll probably work on a full tensor, which has the upper bound for the computing)
void data_load_simu()
{



"""
	##============
	## prep
	##============
	##==== load data (simu)
	X = np.load("./data_simu_data/X.npy")
	Y = np.load("./data_simu_data/Y.npy")
	mapping_cis = np.load("./data_simu_data/mapping_cis.npy")
	Z = np.load("./data_simu_data/Z.npy")

	beta_cis = np.load("./data_simu_init/beta_cis.npy")
	beta_cellfactor1 = np.load("./data_simu_init/beta_cellfactor1.npy")
	beta_cellfactor2 = np.load("./data_simu_init/beta_cellfactor2.npy")
	beta_batch = np.load("./data_simu_init/beta_batch.npy")
	##==== fill dimension
	I = len(X[0])
	J = len(Y[0][0])
	K = len(Y)
	N = len(X)
	D = len(beta_cellfactor1)
	B = len(Z[0])


	# make incomplete tensor numpy array at all levels, in order to supprt numpy array computing
	der_cis = []
	for k in range(K):
		der_cis.append([])
		for j in range(J):
			temp = np.zeros(beta_cis[k][j].shape)
			der_cis[k].append(temp)
		der_cis[k] = np.array(der_cis[k])
	der_cis = np.array(der_cis)

	der_cellfactor1 = np.zeros(beta_cellfactor1.shape)
	der_cellfactor2 = np.zeros(beta_cellfactor2.shape)
	der_batch = np.zeros(beta_batch.shape)

	##==== append intercept to X, and Z (for convenience of cell factor pathway, and batch pathway)
	## X
	array_ones = (np.array([np.ones(N)])).T
	X = np.concatenate((X, array_ones), axis=1)									# N x (I+1)
	## Z
	array_ones = (np.array([np.ones(N)])).T
	Z = np.concatenate((Z, array_ones), axis=1)									# N x (B+1)
"""






	// global:
	//	X, Y, markerset
	//	Y1, U1, Beta, V1, T1, Y2, U2, V2, T2
	//	K, I, J, S, D1, D2
	//	alpha, N_element
	cout << "loading the simu data..." << endl;



	// NOTE: use fake data (to test) or not?
	int indicator_real = 1;
	if(indicator_real)
	{
		cout << "--> loading the real simu data..." << endl;

		//==========================================
		//==== load parameters, init
		char filename[100];




		//==== matrix
		//== U1
		sprintf(filename, "../data_simu_init/U1.txt");
		load_matrix(U1, filename);
		//== V1
		sprintf(filename, "../data_simu_init/V1.txt");
		load_matrix(V1, filename);
		//== T1
		sprintf(filename, "../data_simu_init/T1.txt");
		load_matrix(T1, filename);
		//== U2
		sprintf(filename, "../data_simu_init/U2.txt");
		load_matrix(U2, filename);
		//== V2
		sprintf(filename, "../data_simu_init/V2.txt");
		load_matrix(V2, filename);
		//== T2
		sprintf(filename, "../data_simu_init/T2.txt");
		load_matrix(T2, filename);
		//== Beta
		sprintf(filename, "../data_simu_init/Beta.txt");
		load_matrix(Beta, filename);

		//==== tensor
		//== Y1
		sprintf(filename, "../data_simu_init/Y1.txt");
		load_tensor(Y1, filename);
		//== Y2
		sprintf(filename, "../data_simu_init/Y2.txt");
		load_tensor(Y2, filename);




		//==========================================
		//==== fill in the dimensions
		K = Y1.get_dimension1();
		I = Y1.get_dimension2();
		J = Y1.get_dimension3();
		//
		S = Beta.get_dimension2();
		D1 = Beta.get_dimension1();
		//
		D2 = U2.get_dimension2();


		//==========================================
		//==== load data
		//== X
		sprintf(filename, "../data_simu/X.txt");
		load_matrix(X, filename);



		//== Y
		int indicator_comp = 0;
		if(indicator_comp)				// load complete Y
		{
			cout << "loading complete Y tensor..." << endl;
			sprintf(filename, "../data_simu/Y.txt");
			load_tensor(Y, filename);


			//==========================================
			//==== the others
			int temp = 1;
			markerset.init(K, I, J, temp);
			N_element = int(markerset.sum());
			alpha = 1.0;				// NOTE: need to manually set this

		}
		else 							// load incomplete Y
		{
			cout << "loading incomplete Y tensor..." << endl;
			// Y (dataset), markerset
			Y.init(K, I, J);
			/*
			float temp = 0;
			markerset.init(K, I, J, temp);
			*/
			markerset.init(K, I, J);							// NOTE: this already init all elements as 0
			for(int k=0; k<K; k++)
			{
				char filename[100];
				filename[0] = '\0';
				strcat(filename, "../data_simu_init/Tensor_tissue_");
				char tissue[10];
				sprintf(tissue, "%d", k);
				strcat(filename, tissue);
				strcat(filename, ".txt");

				char type[10] = "r";
				filehandle file(filename, type);

				long input_length = 1000000000;
				char * line = (char *)malloc( sizeof(char) * input_length );
				while(1)
				{
					int end = file.readline(line, input_length);
					if(end)
						break;

					line_class line_obj(line);
					line_obj.split_tab();

					int index = atoi(line_obj.at(0));
					vector<float> vec;
					for(unsigned i=1; i<line_obj.size(); i++)		// NOTE: here we start from pos#1
					{
						char * pointer = line_obj.at(i);
						//float value = stof(pointer);				// NOTE: there are double-range numbers
						float value = stod(pointer);
						vec.push_back(value);
					}
					line_obj.release();
					for(int i=0; i<vec.size(); i++)
					{
						Y.set_element(k, index, i, vec.at(i));
						markerset.set_element(k, index, i, 1);
					}

				}
				free(line);
				file.close();
			}

			cout << "Y and markerset shape:" << endl;
			cout << "(" << Y.get_dimension1() << ", " << Y.get_dimension2() << ", " << Y.get_dimension3() << ")" << endl;
			cout << "(" << markerset.get_dimension1() << ", " << markerset.get_dimension2() << ", " << markerset.get_dimension3() << ")" << endl;


			//==========================================
			//==== init others
			alpha = 1.0;		// just random
			N_element = int(markerset.sum());
		}

	}
	else
	{
		cout << "--> loading the fake simu data..." << endl;

		K = 33;
		I = 450;
		J = 21150;
		S = 824113;
		D1 = 400;
		D2 = 400;

		//==== matrix
		U1.init(I, D1, 1.1);
		V1.init(J, D1, 1.1);
		T1.init(K, D1, 1.1);
		U2.init(I, D2, 1.1);
		V2.init(J, D2, 1.1);
		T2.init(K, D2, 1.1);
		Beta.init(D1, S, 1.1);
		Y1.init(K, I, J, 1.1);
		Y2.init(K, I, J, 1.1);
		X.init(I, S, 1.1);
		Y.init(K, I, J, 1.1);



		//==========================================
		//==== the others
		int temp = 1;
		markerset.init(K, I, J, temp);
		N_element = int(markerset.sum());
		alpha = 1.0;				// NOTE: need to manually set this
	}


	return;
}




/*
//// loading the real data that have been preprocessed
void data_load_real()
{






	return;
}
*/




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============





// save the learned model
// where: "../result/"
void model_save()
{
	cout << "now saving the learned models... (beta_cis, beta_cellfactor1, beta_cellfactor2, beta_batch)" << endl;

	char filename[100];

	//==== matrix
	sprintf(filename, "../result/beta_cellfactor1.txt");
	beta_cellfactor1.save(filename);
	sprintf(filename, "../result/beta_batch.txt");
	beta_batch.save(filename);

	//==== tensor
	sprintf(filename, "../result/beta_cellfactor2.txt");
	beta_cellfactor2.save(filename);

	//==== irregular tensor
	sprintf(filename, "../result/beta_cis.txt");
	beta_cis.save(filename);

	return;
}




// init error list container
void error_init()
{
	list_error.clear();

	return;
}




void save_vector(vector<float> & vec, char * filename)
{
	FILE * file_out = fopen(filename, "w+");
	if(file_out == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}

	for(int i=0; i<vec.size(); i++)
	{
		float value = vec.at(i);
		char buf[1024];
		sprintf(buf, "%f\n", value);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
	}
	fclose(file_out);

	return;
}



// save loglike per need
// where: "../result/"
void error_save()
{
	char filename[] = "../result/error_total.txt";
	save_vector(list_error, filename);

	return;
}



void save_vector_online(vector<float> & vec, char * filename)
{
	int count = vec.size();
	float error = vec.back();

	FILE * file_out;
	if(count == 1)
	{
		file_out = fopen(filename, "w+");
		if(file_out == NULL)
		{
		    fputs("File error\n", stderr); exit(1);
		}
	}
	else
	{
		file_out = fopen(filename, "a+");
		if(file_out == NULL)
		{
		    fputs("File error\n", stderr); exit(1);
		}
	}

	char buf[1024];
	sprintf(buf, "%f\n", error);
	fwrite(buf, sizeof(char), strlen(buf), file_out);

	fclose(file_out);

	return;
}



// save loglike in an online fashion
void error_save_online()
{
	char filename[] = "../result/error_total_online.txt";
	save_vector_online(list_error, filename);

	return;
}



