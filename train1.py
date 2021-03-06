## NOTE:
##	we have (k x n) shape for data
## 	we don't reshape the simulator, but we reshape the data after loading them into the training program





import numpy as np
import math
import timeit





##==== learning setting
## TODO: to determine some parameters
num_iter_out = 100
num_iter_in = 100
size_batch = 20
#rate_learn = 0.0001					# for brain and chr22
#rate_learn = 0.00001					# for 10% of real scale

#rate_learn = 0.0000001					# for 10% of real scale, init (as para from init is too weird)

rate_learn = 0.0000001					# for real scale data







##==== to be filled later on
I = 0						# num of SNPs
J = 0						# num of genes
K = 0						# num of tissues
#L = 0						# length of chromosome
N = 0						# num of individuals
D = 0						# num of cell factors
B = 0						# num of batch variables






##==== variables
## NOTE: here we assume one chromosome model
X = []						# matrix of Individuals x SNPs
Y = []						# tensor of gene expression
mapping_cis = []			# list of (index start, index end)
Z = []						# matrix of Individuals x Batches
## NOTE: the following have the intercept term
beta_cis = []				# tensor of (imcomplete) matrix of Genes x cis- SNPs
beta_cellfactor1 = []		# matrix of first layer cell factor beta
beta_cellfactor2 = []		# tensor (tissue specific) of second layer cell factor beta
beta_batch = []				# matrix of Individuals x Batches
# the following corresponds to the above
der_cis = []
der_cellfactor1 = []
der_cellfactor2 = []
der_batch = []






##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============


##==== forward/backward propogation, and gradient descent
def forward_backward_gd(k):
	global X, Y, Z
	global mapping_cis
	global beta_cis, beta_cellfactor1, beta_cellfactor2, beta_batch
	global der_cis, der_cellfactor1, der_cellfactor2, der_batch
	global I, J, K, D, B, N

	global size_batch, rate_learn


	##==========================================================================================
	## prep
	##==========================================================================================
	X_batch = []
	Z_batch = []

	arr = np.arange(N)
	arr_permute = np.random.permutation(arr)
	list_individual_batch = arr_permute[:size_batch]
	X_batch = X[:, list_individual_batch]
	Z_batch = Z[:, list_individual_batch]


	##==========================================================================================
	## forward prop
	##==========================================================================================
	##=============
	## from cis- (tissue k)
	##=============
	Y_cis = []
	for j in range(J):
		X_sub = []
		start = mapping_cis[j][0]
		end = mapping_cis[j][1]
		X_sub = X_batch[start:end+1,:]
		array_ones = np.array([np.ones(size_batch)])
		X_sub = np.concatenate((X_sub, array_ones), axis=0)						# (amount+1) x size_batch
		beta_sub = beta_cis[k][j]												# 1 x (amount+1)
		Y_sub = np.dot(beta_sub, X_sub)											# 1 x size_batch
		Y_cis.append(Y_sub)
	Y_cis = np.array(Y_cis)														# J x size_batch

	##=============
	## from batch
	##=============
	Y_batch = np.dot(beta_batch, Z_batch)										# J x size_batch

	##=============
	## from cell factor (tissue k)
	##=============
	Y_cellfactor = []

	# first layer
	m_factor_before = np.dot(beta_cellfactor1, X_batch)							# D x size_batch

	# logistic twist
	m_factor_after = np.zeros(m_factor_before.shape)
	for d in range(D):
		for n in range(size_batch):
			x = m_factor_before[d][n]
			m_factor_after[d][n] = 1.0 / (1.0 + math.exp(-x))

	# second layer
	array_ones = np.array([np.ones(size_batch)])
	m_factor_new = np.concatenate((m_factor_after, array_ones), axis=0)			# (D+1) x size_batch
	Y_cellfactor = np.dot(beta_cellfactor2[k], m_factor_new)					# J x size_batch



	##=============
	## compile and error cal
	##=============
	Y_final = Y_cis + Y_batch + Y_cellfactor
	m_error = Y_final - Y[k][:,list_individual_batch]


	##==========================================================================================
	## backward prop
	##==========================================================================================
	##=============
	## from batch
	##=============
	der_batch = np.zeros(beta_batch.shape)		# J x (B+1)
	''' per individual
	for n in range(size_batch):
		der_batch += np.outer(m_error[n], Z_batch[n])
	'''
	# matrix mul: J x N, N x (B+1)
	der_batch = np.dot(m_error, Z_batch.T)		# J x (B+1)
	der_batch = der_batch / size_batch

	##=============
	## from cis- (tissue k)
	##=============
	for j in range(J):
		der_cis[k][j] = np.zeros(der_cis[k][j].shape)
		X_sub = []
		start = mapping_cis[j][0]
		end = mapping_cis[j][1]
		X_sub = X_batch[start:end+1, :]
		array_ones = np.array([np.ones(size_batch)])
		X_sub = np.concatenate((X_sub, array_ones), axis=0)						# (amount+1) x size_batch

		## per individual fashion
		#for n in range(size_batch):
		#	der_cis[k][j] += m_error[n][j] * X_sub[n]
		#der_cis[k][j] = np.dot(m_error[:, j].T, X_sub)
		der_cis[k][j] = np.dot(m_error[j,:], X_sub.T)
		der_cis[k][j] = der_cis[k][j] / size_batch

	##=============
	## from cell factor (tissue k)
	##=============
	##== last layer
	der_cellfactor2[k] = np.zeros(beta_cellfactor2[k].shape)			# J x (D+1)
	## per individual fashion
	#for n in range(size_batch):
	#	der_cellfactor2[k] += np.outer(m_error[n], m_factor_new[n])
	# J x N, N x (D+1)
	der_cellfactor2[k] = np.dot(m_error, m_factor_new.T)
	der_cellfactor2[k] = der_cellfactor2[k] / size_batch

	##== first layer
	der_cellfactor1 = np.zeros(der_cellfactor1.shape)
	## per individual fashion
	'''
	for n in range(size_batch):
		# one individual case
		for d in range(D):
			temp = 0
			for j in range(J):
				par = beta_cellfactor2[k][j][d]
				temp += par * m_error[n][j]

			temp *= m_factor_after[n][d] * (1 - m_factor_after[n][d])

			for i in range(I+1):						## NOTE: we have the intercept variable
				dosage = X_batch[n][i]
				der_cellfactor1[d][i] += dosage * temp
				## eliminate n at the very end, if multiple individual appear
	der_cellfactor1 = der_cellfactor1 / size_batch
	'''
	"""
	## all individual fashion
	# N x J, J x D --> N x D
	m_temp = np.dot(m_error, beta_cellfactor2[k][:, :-1])
	# N x D
	m_factor_der = np.multiply(m_factor_after, 1 - m_factor_after)
	# N x D, N x D --> N x D
	m_temp = np.multiply(m_temp, m_factor_der)
	# D x N, N x (I+1)
	der_cellfactor1 = np.dot(m_temp.T, X_batch)
	der_cellfactor1 = der_cellfactor1 / size_batch
	"""
	## all individual fashion
	# D x J, J x N --> D x N
	m_temp = np.dot(beta_cellfactor2[k][:, :-1].T, m_error)
	# D x N
	m_factor_der = np.multiply(m_factor_after, 1 - m_factor_after)
	# D x N, D x N --> D x N
	m_temp = np.multiply(m_temp, m_factor_der)
	# D x N, N x (I+1)
	der_cellfactor1 = np.dot(m_temp, X_batch.T)
	der_cellfactor1 = der_cellfactor1 / size_batch


	##==========================================================================================
	## gradient descent
	##==========================================================================================
	beta_cis[k] = beta_cis[k] - rate_learn * der_cis[k]
	beta_cellfactor1 = beta_cellfactor1 - rate_learn * der_cellfactor1
	beta_cellfactor2[k] = beta_cellfactor2[k] - rate_learn * der_cellfactor2[k]
	beta_batch = beta_batch - rate_learn * der_batch

	return




##==== calculate the total squared error for specified tissue
def cal_error(k):
	global X, Y, Z
	global mapping_cis
	global beta_cis, beta_cellfactor1, beta_cellfactor2, beta_batch
	global I, J, K, D, B, N


	##=============
	## from cis- (tissue k)
	##=============
	Y_cis = []
	for j in range(J):
		X_sub = []
		start = mapping_cis[j][0]
		end = mapping_cis[j][1]
		X_sub = X[start:end+1,:]
		array_ones = np.array([np.ones(N)])
		X_sub = np.concatenate((X_sub, array_ones), axis=0)						# (amount+1) x N
		beta_sub = beta_cis[k][j]												# 1 x (amount+1)
		Y_sub = np.dot(beta_sub, X_sub)											# 1 x N
		Y_cis.append(Y_sub)
	Y_cis = np.array(Y_cis)


	##=============
	## from batch
	##=============
	Y_batch = np.dot(beta_batch, Z)												# J x N


	##=============
	## from cell factor (tissue k)
	##=============
	Y_cellfactor = []

	# first layer
	m_factor = np.dot(beta_cellfactor1, X)										# D x N

	# logistic twist
	for d in range(D):
		for n in range(N):
			x = m_factor[d][n]
			m_factor[d][n] = 1.0 / (1.0 + math.exp(-x))

	# second layer
	array_ones = np.array([np.ones(N)])
	m_factor_new = np.concatenate((m_factor, array_ones), axis=0)				# (D+1) x N
	Y_cellfactor = np.dot(beta_cellfactor2[k], m_factor_new)					# J x N


	##=============
	## compile and error cal
	##=============
	Y_final = Y_cis + Y_batch + Y_cellfactor
	error = np.sum(np.square(Y[k] - Y_final))

	return error





##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============





if __name__ == "__main__":


	print "now training..."

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



	####=====================
	#### NOTE: reshape data
	####=====================
	X = X.T
	Y_new = []
	for k in range(K):
		matrix = np.array(Y[k]).T
		Y_new.append(matrix)
	Y = np.array(Y_new)
	Z = Z.T
	print "reshaped data:",
	print X.shape,
	print Y.shape,
	print Z.shape




	##============
	## train
	##============
	##==== train (mini-batch)
	list_error = []
	for iter1 in range(num_iter_out):
		print "[@@@]working on out iter#",
		print iter1

		for k in range(K):
			print "[##]",
			print iter1,
			print k

			for iter2 in range(num_iter_in):
				print "[@]",
				print iter1,
				print k,
				print iter2

				##==== timer
				start_time = timeit.default_timer()

				forward_backward_gd(k)
				error = cal_error(k)
				print "current total error:",
				print error

				list_error.append(error)
				np.save("./result/list_error", np.array(list_error))

				##==== timer
				elapsed = timeit.default_timer() - start_time
				print "time spent on this batch:", elapsed

	print "done!"








