## init the model from real data


## we will do the following initialization:
##	1. we assume the cis- effect and trans- effect are equal, so we split the expression tensor and use half to do each
##	2. for the cell factors, we will do PCA for each tissue separately, and average to get the factors; we then re-do the linear system to get the tissue specific parameters (after we twist the cell factors to (0, 1))
##	3. we twist the cell factors into range(0, 1), and map back them for logistic function
##	4. we then solve the linear system to get the genome-wide factor regulation
##	5. we solve multiple linear systems to get the cis- part
##	6. we then forward propagation and get the residual, and use the averaged residual to initialize the batch effect
##	7. we will always append a random constant effect (the intercept) if necessary

## for the factors, second thought:
##	1. use all the samples to get the 400 factors (per tissue sample size might be smaller than this number)
##	2. then to disentangle the tissue-specific effects



## NOTE: we are working on an incomplete tensor



import numpy as np
from sklearn.decomposition import PCA
import math



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
mapping_cis = []			# list of (index start, index end)
Z = []						# matrix of Individuals x Batches

init_beta_cis = []				# tensor of (imcomplete) matrix of Genes x cis- SNPs
init_beta_cellfactor1 = []		# matrix of first layer cell factor beta
init_beta_cellfactor2 = []		# tensor (tissue specific) of second layer cell factor beta
init_beta_batch = []			# matrix of Individuals x Batches





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


def reformat_tensor(tensor, filename):
	shape = tensor.shape
	dimension1 = shape[0]
	dimension2 = shape[1]
	dimension3 = shape[2]

	file = open(filename, 'w')
	file.write(str(dimension1) + '\t' + str(dimension2) + '\t' + str(dimension3) + '\n')
	for i in range(dimension1):
		for j in range(dimension2):

			for count in range(dimension3):
				value = tensor[i][j][count]
				if count != (dimension3-1):
					file.write(str(value) + '\t')
				else:
					file.write(str(value))
			file.write('\n')
	file.close()
	return


def reformat_beta_cis(tensor, filename):
	shape = tensor.shape
	dimension1 = shape[0]
	dimension2 = shape[1]

	file = open(filename, 'w')
	file.write(str(dimension1) + '\t' + str(dimension2) + '\n')
	for i in range(dimension1):
		for j in range(dimension2):

			dimension3 = len(tensor[i][j])
			for count in range(dimension3):
				value = tensor[i][j][count]
				if count != (dimension3-1):
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
	X = np.load("./data_real_data_train/X.npy")
	# Y and Y_pos
	K = 28										## TODO: specify the number of tissues
	Y = []
	Y_pos = []
	for k in range(K):
		data = []
		list_pos = []

		file = open("./data_real_data_train/Tensor_tissue_" + str(k) + ".txt", 'r')
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
	mapping_cis = np.load("./data_real_data_train/mapping_cis.npy")
	#
	Z = np.load("./data_real_data_train/Z.npy")

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


	# make incomplete tensor numpy array at all levels, in order to supprt numpy array computing
	init_beta_cis = []
	for k in range(K):
		init_beta_cis.append([])
		for j in range(J):
			#temp = np.zeros(beta_cis[k][j].shape)
			amount = mapping_cis[j][1] - mapping_cis[j][0] + 1 + 1			## NOTE: the intercept
			temp = np.zeros(amount)
			init_beta_cis[k].append(temp)
		init_beta_cis[k] = np.array(init_beta_cis[k])
	init_beta_cis = np.array(init_beta_cis)

	#init_beta_cellfactor1 = np.zeros(beta_cellfactor1.shape)
	init_beta_cellfactor1 = np.zeros((D, I+1))
	#init_beta_cellfactor2 = np.zeros(beta_cellfactor2.shape)
	init_beta_cellfactor2 = np.zeros((K, J, D+1))
	#init_beta_batch = np.zeros(beta_batch.shape)
	init_beta_batch = np.zeros((J, B+1))


	##==== append intercept to X, and Z (for convenience of cell factor pathway, and batch pathway)
	## X
	array_ones = (np.array([np.ones(N)])).T
	X = np.concatenate((X, array_ones), axis=1)									# N x (I+1)
	## Z
	array_ones = (np.array([np.ones(N)])).T
	Z = np.concatenate((Z, array_ones), axis=1)									# N x (B+1)






	##=====================================================================================================================
	##==== cis-
	##=====================================================================================================================
	Y_cis = 0.5 * Y
	for j in range(J):
		#X_sub = []
		#start = mapping_cis[j][0]
		#end = mapping_cis[j][1]
		#X_sub = X[:, start:end+1]
		#array_ones = (np.array([np.ones(N)])).T
		#X_sub = np.concatenate((X_sub, array_ones), axis=1)					# N x (amount+1)

		for k in range(K):
			#get the X_sub for the avaliable samples for this tissue
			X_sub = []
			start = mapping_cis[j][0]
			end = mapping_cis[j][1]
			for n in range(len(Y_pos[k])):
				pos = Y_pos[k][n]
				X_sub.append(X[pos][start:end+1])
			array_ones = (np.array([np.ones(len(Y_pos[k]))])).T
			X_sub = np.concatenate((X_sub, array_ones), axis=1)					# N x (amount+1)

			# the linear system: X_sub x beta = Y_sub
			Y_sub = Y_cis[k][:, j]												# N x 1
			init_beta_sub = np.linalg.lstsq(X_sub, Y_sub)[0]
			init_beta_cis[k][j] = init_beta_sub
	init_beta_cis = np.array(init_beta_cis)
	print "init_beta_cis shape:",
	print init_beta_cis.shape
	print "and data types of three levels:",
	print type(init_beta_cis),
	print type(init_beta_cis[0]),
	print type(init_beta_cis[0][0])






	##=====================================================================================================================
	##==== cell factor
	##=====================================================================================================================
	Y_cellfactor = 0.5 * Y
	##
	## init_beta_cellfactor2
	##
	## to get the individual factor matrix, we have two approaches:
	"""
	####=============================== Scheme#1 ===============================
	## tissue pca, which limits the number of factors that could be used
	t_factor = []
	for k in range(K):
		Data = Y_cellfactor[k]
		n_factor = D
		##==== do PCA for Individual x Gene, with number of factors as D
		pca = PCA(n_components=n_factor)
		pca.fit(Data)
		Y2 = (pca.components_).T 					# Gene x Factor
		Y1 = pca.transform(Data)					# Individual x Factor
		variance = pca.explained_variance_ratio_

		Y1 = np.array(Y1)
		t_factor.append(Y1)
	t_factor = np.array(t_factor)
	#m_factor = np.average(t_factor, axis=0)
	m_factor = np.zeros((N, D))
	list_count = np.zeros(N)
	for k in range(K):
		for n in range(len(Y_pos[k])):
			pos = Y_pos[k][n]
			m_factor[pos] += t_factor[k][n]
			list_count[pos] += 1
	for n in range(N):
		m_factor[n] = m_factor[n] / list_count[n]
	####========================================================================
	"""


	####=============================== Scheme#2 ===============================
	##	1. do PCA on sample matrix
	##	2. averaging the (Individual x Factor) matrix in order to eliminate the tissue effects, thus only individual effects left
	##	3. use these individual effects to retrieve their SNP causality
	##	4. use these individual effects to separately associate tissue effects of these factors
	##==== sample matrix
	Y_matrix = []
	Y_matrix_pos = []
	for i in range(len(Y_cellfactor)):
		for j in range(len(Y_cellfactor[i])):
			Y_matrix.append(Y_cellfactor[i][j])
			Y_matrix_pos.append(Y_pos[i][j])
	Y_matrix = np.array(Y_matrix)
	print "sample matrix shape:", Y_matrix.shape

	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = D
	pca = PCA(n_components=n_factor)
	pca.fit(Y_matrix)
	Y2 = (pca.components_).T 					# Gene x Factor
	Y1 = pca.transform(Y_matrix)				# Sample x Factor
	variance = pca.explained_variance_ratio_

	##==== individual factors
	m_factor = np.zeros((N, D))
	list_count = np.zeros(N)
	for i in range(len(Y1)):
		pos = Y_matrix_pos[i]
		m_factor[pos] += Y1[i]
		list_count[pos] += 1
	for n in range(N):
		m_factor[n] = m_factor[n] / list_count[n]
	####========================================================================





	## tune factor matrix into [0.1, 0.9]
	value_max = np.amax(m_factor)
	value_min = np.amin(m_factor)
	m_factor_tune = (m_factor - value_min) * (1 / (value_max - value_min))
	m_factor_tune = 0.5 + 0.8 * (m_factor_tune - 0.5)

	array_ones = (np.array([np.ones(N)])).T
	m_factor_tune = np.concatenate((m_factor_tune, array_ones), axis=1)					# N x (D+1)


	for k in range(K):
		#reshape to t_factor, in which each tissue has different number of samples
		m_factor_tune_sub = []
		for n in range(len(Y_pos[k])):
			pos = Y_pos[k][n]
			m_factor_tune_sub.append(m_factor_tune[pos])
		m_factor_tune_sub = np.array(m_factor_tune_sub)
		#
		Y_sub = Y_cellfactor[k]
		# the linear system: m_factor_tune_sub x beta = Y_sub
		init_beta = np.linalg.lstsq(m_factor_tune_sub, Y_sub)[0].T
		init_beta_cellfactor2[k] = init_beta
	init_beta_cellfactor2 = np.array(init_beta_cellfactor2)
	print "init_beta_cellfactor2 shape:",
	print init_beta_cellfactor2.shape





	##
	## init_beta_cellfactor1
	##
	m_factor_before = np.zeros(m_factor_tune.shape)[:, :-1]
	for n in range(N):
		for d in range(D):
			x = m_factor_tune[n][d]
			m_factor_before[n][d] = np.log( x / (1-x) )
	# the linear system: X x beta = m_factor_before
	init_beta_cellfactor1 = np.linalg.lstsq(X, m_factor_before)[0].T
	print "init_beta_cellfactor1 shape:",
	print init_beta_cellfactor1.shape







	##=====================================================================================================================
	##==== residual for batch
	##=====================================================================================================================
	Y_batch = np.zeros((N, J))
	list_count = np.zeros(N)

	## extra cache for cell factor 1
	# first layer
	init_beta_cellfactor1_reshape = init_beta_cellfactor1.T 				# (I+1) x D
	m_factor = np.dot(X, init_beta_cellfactor1_reshape)						# N x D
	# logistic twist
	for n in range(N):
		for d in range(D):
			x = m_factor[n][d]
			m_factor[n][d] = 1.0 / (1.0 + math.exp(-x))
	# second layer input
	array_ones = (np.array([np.ones(N)])).T
	m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)

	for k in range(K):
		##=============
		## from cis-
		##=============
		Y_cis = []
		for j in range(J):
			X_sub = []
			start = mapping_cis[j][0]
			end = mapping_cis[j][1]
			X_sub = X[:, start:end+1]
			array_ones = (np.array([np.ones(N)])).T
			X_sub = np.concatenate((X_sub, array_ones), axis=1)						# N x (amount+1)
			init_beta_sub = init_beta_cis[k][j]										# 1 x (amount+1)
			Y_sub = np.dot(X_sub, init_beta_sub)									# 1 x N
			Y_cis.append(Y_sub)
		Y_cis = np.array(Y_cis)
		Y_cis = Y_cis.T
		print "Y_cis shape:",
		print Y_cis.shape

		##=============
		## from cell factor
		##=============
		Y_cellfactor = []

		# NOTE: cell factor 1 cal in advance

		## second layer
		#array_ones = (np.array([np.ones(N)])).T
		#m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)
		init_beta_cellfactor2_reshape = init_beta_cellfactor2[k].T 				# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new, init_beta_cellfactor2_reshape)		# N x J
		print "Y_cellfactor shape:",
		print Y_cellfactor.shape

		##=============
		## compile to get residual
		##=============
		#only record some samples
		Y_cis_cellfactor = Y_cis + Y_cellfactor
		Y_sub = Y[k]
		for n in range(len(Y_pos[k])):
			pos = Y_pos[k][n]
			delta = Y_sub[n] - Y_cis_cellfactor[pos]
			Y_batch[pos] += delta
			list_count[pos] += 1

	## averaging:
	for n in range(N):
		Y_batch[n] = Y_batch[n] / list_count[n]

	## the linear system: Z x beta = Y_batch
	init_beta_batch = np.linalg.lstsq(Z, Y_batch)[0].T
	print "init_beta_batch shape:",
	print init_beta_batch.shape






	##=====================================================================================================================
	##==== save the data
	##=====================================================================================================================
	##==== save data (NOTE: not do, as too much space to cost)
	#np.save("./data_real_init/beta_cis", init_beta_cis)
	#np.save("./data_real_init/beta_cellfactor1", init_beta_cellfactor1)
	#np.save("./data_real_init/beta_cellfactor2", init_beta_cellfactor2)
	#np.save("./data_real_init/beta_batch", init_beta_batch)

	##==== reformat data
	reformat_beta_cis(init_beta_cis, "./data_real_init/beta_cis.txt")
	reformat_matrix(init_beta_cellfactor1, "./data_real_init/beta_cellfactor1.txt")
	reformat_tensor(init_beta_cellfactor2, "./data_real_init/beta_cellfactor2.txt")
	reformat_matrix(init_beta_batch, "./data_real_init/beta_batch.txt")





	print "done..."








