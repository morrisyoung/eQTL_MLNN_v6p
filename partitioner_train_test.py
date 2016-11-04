## partition all samples into training set and testing set
## input:
##	1. data_real_data
## output:
##	1. data_real_data_train
##	2. data_real_data_test


## NOTE: only save .txt for the training and testing sets



import numpy as np



##==== scale of the input data
I = 0						# num of SNPs
J = 0						# num of genes
K = 0						# num of tissues
N = 0						# num of individuals
D = 0						# num of cell factors
B = 0						# num of batch variables



##==== variables
## NOTE: here we assume one chromosome model
X = []						# matrix of Individuals x SNPs
Y = []						# tensor of gene expression (incomplete)
Y_pos = []					# list of pos
Z = []						# matrix of Individuals x Batches

## new data
X_train = []
X_test = []
Z_train = []
Z_test = []

Y_train = []
Y_train_pos = []
Y_test = []
Y_test_pos = []


percent = 0.75				# the portion of training samples




##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============


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


def reformat_mapping_cis(matrix, filename):
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


##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============



if __name__ == "__main__":


	##=====================================================================================================================
	##==== load data (real)
	##=====================================================================================================================
	#
	X = np.load("./data_real_data/X.npy")
	# Y and Y_pos
	K = 28										## TODO: specify the number of tissues
	Y = []
	Y_pos = []
	for k in range(K):
		data = []
		list_pos = []

		file = open("./data_real_data/Tensor_tissue_" + str(k) + ".txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break

			line = line.split('\t')
			pos = int(line[0])
			expr_list = map(lambda x: float(x), line[1:])

			data.append(expr_list)
			list_pos.append(pos)
		file.close()
		data = np.array(data)
		list_pos = np.array(list_pos)
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)
	#
	Z = np.load("./data_real_data/Z.npy")

	##==== fill dimension
	I = len(X[0])
	J = len(Y[0][0])
	K = len(Y)
	N = len(X)
	D = 400										## TODO: manually set this
	B = len(Z[0])
	print "shape:"
	print "I:", I
	print "J:", J
	print "K:", K
	print "N:", N
	print "D:", D
	print "B:", B



	##=====================================================================================================================
	##==== random
	##=====================================================================================================================
	list_random = np.random.permutation(N)
	upper = int(percent * N)
	list_train = list_random[:upper]
	list_test = list_random[upper:]
	rep_train = {}
	for pos in list_train:
		rep_train[pos] = 1
	map_pos_train = {}
	for i in range(len(list_train)):
		pos = list_train[i]
		map_pos_train[pos] = i
	map_pos_test = {}
	for i in range(len(list_test)):
		pos = list_test[i]
		map_pos_test[pos] = i


	############
	#### X_train and X_test
	############
	X_train = []
	X_test = []
	for i in range(len(X)):
		if i in rep_train:
			X_train.append(X[i])
		else:
			X_test.append(X[i])
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	print "genotype train and test shapes:",
	print X_train.shape,
	print X_test.shape

	np.save("./data_real_data_train/X", X_train)
	reformat_matrix(X_train, "./data_real_data_train/X.txt")
	reformat_matrix(X_test, "./data_real_data_test/X.txt")




	############
	#### Z_train, Z_test
	############
	Z_train = []
	Z_test = []
	for i in range(len(Z)):
		if i in rep_train:
			Z_train.append(Z[i])
		else:
			Z_test.append(Z[i])
	Z_train = np.array(Z_train)
	Z_test = np.array(Z_test)
	print "batch train and test shapes:",
	print Z_train.shape,
	print Z_test.shape

	np.save("./data_real_data_train/Z", Z_train)
	reformat_matrix(Z_train, "./data_real_data_train/Z.txt")
	reformat_matrix(Z_test, "./data_real_data_test/Z.txt")





	############
	#### Y_train, Y_train_pos, Y_test, Y_test_pos
	############
	Y_train = []
	Y_train_pos = []
	Y_test = []
	Y_test_pos = []

	for k in range(K):
		Y_train.append([])
		Y_train_pos.append([])
		Y_test.append([])
		Y_test_pos.append([])

		for i in range(len(Y_pos[k])):
			pos = Y_pos[k][i]
			list_expr = Y[k][i]
			if pos in rep_train:
				Y_train[-1].append(list_expr)
				Y_train_pos[-1].append(pos)
			else:
				Y_test[-1].append(list_expr)
				Y_test_pos[-1].append(pos)

		## TEST
		print "tissue# and their training and testing samples:",
		print k,
		print len(Y_train_pos[-1]),
		print len(Y_test_pos[-1])

		## mapping to the new pos, for both training set and testing set
		for i in range(len(Y_train_pos[-1])):
			pos_old = Y_train_pos[-1][i]
			pos_new = map_pos_train[pos_old]
			Y_train_pos[-1][i] = pos_new
		for i in range(len(Y_test_pos[-1])):
			pos_old = Y_test_pos[-1][i]
			pos_new = map_pos_test[pos_old]
			Y_test_pos[-1][i] = pos_new

	## save training and testing tensor
	## training
	for k in range(K):
		file = open("./data_real_data_train/Tensor_tissue_" + str(k) + ".txt", 'w')
		for i in range(len(Y_train[k])):
			pos = Y_train_pos[k][i]
			file.write(str(pos) + '\t')
			for j in range(len(Y_train[k][i])):
				value = Y_train[k][i][j]
				if j != (len(Y_train[k][i])-1):
					file.write(str(value) + '\t')
				else:
					file.write(str(value))
			file.write('\n')
		file.close()
	## testing
	for k in range(K):
		file = open("./data_real_data_test/Tensor_tissue_" + str(k) + ".txt", 'w')
		for i in range(len(Y_test[k])):
			pos = Y_test_pos[k][i]
			file.write(str(pos) + '\t')
			for j in range(len(Y_test[k][i])):
				value = Y_test[k][i][j]
				if j != (len(Y_test[k][i])-1):
					file.write(str(value) + '\t')
				else:
					file.write(str(value))
			file.write('\n')
		file.close()






	##==== mapping_cis
	mapping_cis = np.load("./data_real_data/mapping_cis.npy")

	reformat_mapping_cis(mapping_cis, "./data_real_data_train/mapping_cis.txt")
	reformat_mapping_cis(mapping_cis, "./data_real_data_test/mapping_cis.txt")

	np.save("./data_real_data_train/mapping_cis", mapping_cis)
	np.save("./data_real_data_test/mapping_cis", mapping_cis)




