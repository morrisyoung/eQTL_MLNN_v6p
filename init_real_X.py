## prepare the concatenated and text format genotype

import numpy as np


def reformat_matrix(matrix, filename):
	shape = matrix.shape
	dimension1 = shape[0]
	dimension2 = shape[1]

	file = open(filename, 'w')
	for i in range(dimension1):
		for j in range(dimension2):
			value = matrix[i][j]
			if j != (dimension2-1):
				file.write(str(value) + '\t')
			else:
				file.write(str(value))
		file.write('\n')
	file.close()
	return



if __name__ == "__main__":

	##==== X (genotype)
	list_individual = np.load("../../GTEx_tensor_genetics/Individual_list.npy")
	X = []
	for individual in list_individual:
		X.append([])

		for chr_index in range(22):
			chr = chr_index + 1
			file = open("../../GTEx_tensor_genetics/genotype_450_dosage_matrix_qc/chr" + str(chr) + "/SNP_dosage_" + individual + ".txt", 'r')
			while 1:
				line = (file.readline()).strip()
				if not line:
					break

				X[-1].append(float(line))
			file.close()

	X = np.array(X)
	print "X (dosage) shape:", X.shape
	np.save("./data_real_data/X", X)

	##== reformat data
	reformat_matrix(X, "./data_real_data/X.txt")



