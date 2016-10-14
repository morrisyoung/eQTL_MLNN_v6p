// utility.h
// function:

#ifndef UTILITY_H
#define UTILITY_H



#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       /* pow */
#include <string>
#include <string.h>
#include <vector>




using namespace std;



class Map_list
{
	int dimension;									// number of genes
	//vector<vector<int>> list_pair;
	int * list_start;								// list of start pos
	int * list_end;									// list of end pos

public:

	int get_start_at(int pos)
	{
		return list_start[pos];
	}

	int get_end_at(int pos)
	{
		return list_end[pos];
	}

	int * get_list_start()
	{
		return list_start;
	}

	int * get_list_end()
	{
		return list_end;
	}

};





//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============

class Matrix
{
	int dimension1;
	int dimension2;
	float * matrix;

public:

	void init(int length1, int length2)
	{
		dimension1 = length1;
		dimension2 = length2;
		matrix = (float *)calloc( length1 * length2, sizeof(float) );
		return;
	}

	float * get_array_at(int pos)
	{
		return matrix + (pos-1)*dimension2;
	}

	void fill_with_ref_list(int * list_indiv_pos, float * matrix_ref)
	{
		for(int i=0; i<dimension1; i++)
		{
			int indiv_pos = list_indiv_pos[i];
			float * pointer_ref = matrix_ref + indiv_pos * dimension2;
			float * pointer = matrix + i * dimension2;
			memcpy( pointer, pointer_ref, dimension2*sizeof(float) );
		}
		return;
	}

	int get_dimension1()
	{
		return dimension1;
	}

	int get_dimension2()
	{
		return dimension2;
	}

	float * get_pointer()
	{
		return matrix;
	}

	// delete object
	void release()
	{
		free(matrix);
		return;
	}


	/*
	void init(int length1, int length2, float value)
	{
		dimension1 = length1;
		dimension2 = length2;
		matrix = (float *)calloc( length1 * length2, sizeof(float) );

		for(int i=0; i<length1*length2; i++)
		{
			matrix[i] = value;
		}
		return;
	}

	void init(int length1, int length2)
	{
		dimension1 = length1;
		dimension2 = length2;
		matrix = (float *)calloc( length1 * length2, sizeof(float) );
		return;
	}

	void init(int length1, int length2, float * data)
	{
		dimension1 = length1;
		dimension2 = length2;
		matrix = (float *)calloc( length1 * length2, sizeof(float) );

		int amount = dimension1*dimension2;
		memcpy( matrix, data, amount*sizeof(float) );

		return;
	}

	// used for loading data, since we might use other containers to load data first of all
	void init(vector<vector<float>> & container)
	{
		dimension1 = container.size();
		dimension2 = (container.at(0)).size();
		matrix = (float *)calloc( dimension1 * dimension2, sizeof(float) );

		int count = 0;
		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				float value = (container.at(i)).at(j);
				matrix[count] = value;
				count += 1;
			}
		}

		return;
	}

	void print_shape()
	{
		cout << "this matrix has shape: " << dimension1 << ", " << dimension2 << endl;
		return;
	}

	int get_dimension1()
	{
		return dimension1;
	}

	int get_dimension2()
	{
		return dimension2;
	}

	// NOTE: use the following function carefully -- do not modify elements from outside
	float * get_matrix()
	{
		return matrix;
	}

	float get_element(int ind1, int ind2)
	{
		int index = ind1 * dimension2 + ind2;
		return matrix[index];
	}

	void set_element(int ind1, int ind2, float value)
	{
		int index = ind1 * dimension2 + ind2;
		matrix[index] = value;
		return;
	}

	// add on a value to the matrix entry, in place
	void addon(int ind1, int ind2, float value)
	{
		int index = ind1 * dimension2 + ind2;
		matrix[index] += value;
		return;
	}

	// sum all elements
	float sum()
	{
		float result = 0;
		for(int i=0; i<dimension1*dimension2; i++)
		{
			result += matrix[i];
		}
		return result;
	}

	// SOS for all elements
	float sum_of_square()
	{
		float result = 0;
		for(int i=0; i<dimension1*dimension2; i++)
		{
			result += pow(matrix[i], 2.0);
		}
		return result;
	}

	// SOS for on array
	float sum_of_square(int ind)
	{
		float result = 0;
		int start = ind * dimension2;
		for(int i=0; i<dimension2; i++)
		{
			result += pow(matrix[start+i], 2.0);
		}
		return result;
	}

	// raise all elements to the power of float exp
	void power(float exp)
	{
		for(int i=0; i<dimension1*dimension2; i++)
		{
			matrix[i] = pow(matrix[i], exp);
		}
		return;
	}

	// times all elements with coef
	void multiply(float coef)
	{
		for(int i=0; i<dimension1*dimension2; i++)
		{
			matrix[i] = matrix[i] * coef;
		}
		return;
	}

	// given a filename, try to save this matrix into a file
	void save(char * filename)
	{
		FILE * file_out = fopen(filename, "w+");
		if(file_out == NULL)
		{
		    fputs("File error\n", stderr); exit(1);
		}

		for(int i=0; i<dimension1; i++)
		{
			int start = i * dimension2;
			for(int j=0; j<dimension2; j++)
			{
				int index = start + j;
				float value = matrix[index];
				char buf[1024];
				sprintf(buf, "%f\t", value);
				fwrite(buf, sizeof(char), strlen(buf), file_out);
			}
			fwrite("\n", sizeof(char), 1, file_out);
		}
		fclose(file_out);
		return;
	}


	// delete object
	void release()
	{
		free(matrix);
		return;
	}
	*/

};



//================
class Tensor
{
	long int dimension1;
	long int dimension2;
	long int dimension3;
	float * tensor;

public:


	float * get_matrix_at(int pos)
	{
		float * pointer = tensor + pos*(dimension2*dimension3);
		return pointer;
	}


	/*
	void init(long int length1, long int length2, long int length3)
	{
		dimension1 = length1;
		dimension2 = length2;
		dimension3 = length3;
		long int dimension = length1 * length2 * length3;
		tensor = (float *)calloc( dimension, sizeof(float) );
		return;
	}

	void init(long int length1, long int length2, long int length3, float value)
	{
		dimension1 = length1;
		dimension2 = length2;
		dimension3 = length3;
		long int dimension = length1 * length2 * length3;
		tensor = (float *)calloc( dimension, sizeof(float) );

		long int amount = dimension1*dimension2*dimension3;
		for(long int i=0; i<amount; i++)
		{
			tensor[i] = value;
		}

		return;
	}

	void init(long int length1, long int length2, long int length3, float * data)
	{
		dimension1 = length1;
		dimension2 = length2;
		dimension3 = length3;
		long int dimension = length1 * length2 * length3;
		tensor = (float *)calloc( dimension, sizeof(float) );

		long int amount = dimension1*dimension2*dimension3;
		memcpy( tensor, data, amount*sizeof(float) );

		return;
	}

	// used for loading data, since we might use other containers to load data first of all
	void init(vector<vector<vector<float>>> & container)
	{
		dimension1 = container.size();
		dimension2 = (container.at(0)).size();
		dimension3 = ((container.at(0)).at(0)).size();
		long int dimension = dimension1 * dimension2 * dimension3;
		tensor = (float *)calloc( dimension, sizeof(float) );

		long int count = 0;
		for(long int i=0; i<dimension1; i++)
		{
			for(long int j=0; j<dimension2; j++)
			{
				for(long int d=0; d<dimension3; d++)
				{
					float value = ((container.at(i)).at(j)).at(d);
					tensor[count] = value;
					count += 1;
				}
			}
		}

		return;
	}

	void print_shape()
	{
		cout << "this tensor has shape: " << dimension1 << ", " << dimension2 << ", " << dimension3 << endl;
		return;
	}

	long int get_dimension1()
	{
		return dimension1;
	}

	long int get_dimension2()
	{
		return dimension2;
	}

	long int get_dimension3()
	{
		return dimension3;
	}

	// NOTE: use the following function carefully -- do not modify elements from outside
	float * get_tensor()
	{
		return tensor;
	}

	float * get_tensor_at(long int i)
	{
		long int shift = i * dimension2 * dimension3;
		return (tensor+shift);
	}

	float get_element(long int ind1, long int ind2, long int ind3)
	{
		long int index = ind1 * dimension2 * dimension3 + ind2 * dimension3 + ind3;
		return tensor[index];
	}

	void set_element(long int ind1, long int ind2, long int ind3, float value)
	{
		long int index = ind1 * dimension2 * dimension3 + ind2 * dimension3 + ind3;
		tensor[index] = value;
		return;
	}

	float sum()
	{
		float result = 0;
		for(long int i=0; i<dimension1*dimension2*dimension3; i++)
		{
			result += tensor[i];
		}
		return result;
	}

	// SOS for one layer
	float sum_of_square(long int ind)
	{
		float result = 0;
		long int start = ind * dimension2 * dimension3;
		for(long int i=0; i<dimension2*dimension3; i++)
		{
			result += pow(tensor[start + i], 2.0);
		}
		return result;
	}

	// raise all elements to the power of float exp
	void power(float exp)
	{
		for(long int i=0; i<dimension1*dimension2*dimension3; i++)
		{
			tensor[i] = pow(tensor[i], exp);
		}
		return;
	}

	// given a filename, try to save this tensor into a file
	void save(char * filename)
	{
		FILE * file_out = fopen(filename, "w+");
		if(file_out == NULL)
		{
		    fputs("File error\n", stderr); exit(1);
		}

		char buf[1024];
		sprintf(buf, "%d\t", dimension1);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
		sprintf(buf, "%d\t", dimension2);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
		sprintf(buf, "%d\t", dimension3);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
		fwrite("\n", sizeof(char), 1, file_out);

		for(long int k=0; k<dimension1; k++)
		{
			for(long int i=0; i<dimension2; i++)
			{
				long int start = k * dimension2 * dimension3 + i * dimension3;
				for(long int j=0; j<dimension3; j++)
				{
					long int index = start + j;
					float value = tensor[index];
					char buf[1024];
					sprintf(buf, "%f\t", value);
					fwrite(buf, sizeof(char), strlen(buf), file_out);
				}
				fwrite("\n", sizeof(char), 1, file_out);
			}
		}
		fclose(file_out);
		return;
	}


	// delete object
	void release()
	{
		free(tensor);
		return;
	}
	*/


};





class Tensor_expr
{
	int dimension1;										// number of tissues
	//vector<vector<int>> list_list_indiv_pos;			// contains the mappings to pos of individuals
	//int ** list_list_indiv_pos;							// same as above
	vector<int *> list_list_indiv_pos;					// same as above
	vector<int> list_dimension2;						// number of genes for different tissues
	int dimension3;										// number of genes
	vector<float *> list_matrix;						// expression matrix for different tissues

public:

	void load()
	{
		return;
	}

	int get_dimension2_at(int pos)
	{
		return list_dimension2.at(pos);
	}

	float * get_matrix_at(int pos)
	{
		return list_matrix.at(pos);
	}

	int * get_list_indiv_pos_at(int pos)
	{
		return list_list_indiv_pos.at(pos);
	}

	int get_indiv_pos_at(int index_tissue, int pos)
	{
		return (list_list_indiv_pos.at(index_tissue))[pos];
	}



};



class Tensor_beta_cis
{
	int dimension1;										// number of tissues
	int dimension2;										// number of genes
	//vector<int> list_dimension3;						// number of dimension3 for different dimension2 (gene)
	//int * list_dimension3;							// same as above
	int * list_start;									// kind of same as above
	vector<float *>	list_incomp_matrix;					// incomplete beta matrix for different tissues
	int amount;											// total amount of cis- parameters (for one tissue)
	int * list_beta_cis_geneindex;						// mapping each parameter to their gene index


public:


	void load()
	{
		//
		return;
	}

	int * get_list_start()
	{
		return list_start;
	}

	float * get_incomp_matrix_at(int pos)
	{
		return list_incomp_matrix.at(pos);
	}

	int get_amount()
	{
		return amount;
	}

	int * get_list_beta_cis_geneindex()
	{
		return list_beta_cis_geneindex;
	}


};






// other function declearations





#endif

// end of utility.h


