## do the mini-batch gradient descent

## NOTE:
##	1. dimension indicators should be used whenever needed, rather than the len(Var) (as input will be appended to the intercept term)
##	2. batch has consistent effects across different tissues (so we don't have tissue-specific parameters)

## NOTE:
##	1. in this script, I use (n x k) to index the data, so I need to reshape beta everytime (from (d x k) to (k x d)); data should normally have (k x n) shape


## NOTE: the new GD algorithm, rounding over all samples to update all parameters





import numpy as np
import math
import timeit
import sys





##==== learning setting
## TODO: to determine some parameters
num_iter = 1

#rate_learn = 0.0001					# for brain and chr22
#rate_learn = 0.00001					# for 10% of real scale
rate_learn = 0.0000001					# for 10% of real scale, init (as para from init is too weird)
#rate_learn = 0.0000001					# for real scale data







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
Y = []
Y_pos = []
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
def forward_backward_gd():
	global X, Y, Y_pos, Z
	global mapping_cis
	global beta_cis, beta_cellfactor1, beta_cellfactor2, beta_batch
	global der_cis, der_cellfactor1, der_cellfactor2, der_batch
	global I, J, K, D, B, N

	global rate_learn


	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##==========================================================================================
	## forward prop
	##==========================================================================================
	##=========##=========##=========##=========##=========##=========##=========##=========##==
	Y_exp = []

	##==========================================================================================
	# first layer
	beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
	m_factor_before = np.dot(X, beta_cellfactor1_reshape)					# N x D

	# logistic twist
	m_factor_after = np.zeros(m_factor_before.shape)
	for n in range(N):
		for d in range(D):
			x = m_factor_before[n][d]
			m_factor_after[n][d] = 1.0 / (1.0 + math.exp(-x))

	# second layer
	array_ones = (np.array([np.ones(N)])).T
	m_factor_new = np.concatenate((m_factor_after, array_ones), axis=1)		# N x (D+1)


	##=============
	## from batch
	##=============
	beta_batch_reshape = beta_batch.T 										# (B+1) x J
	Y_batch = np.dot(Z, beta_batch_reshape)									# N x J
	##==========================================================================================


	for k in range(K):

		##=============
		## from cis- (tissue k)
		##=============
		Y_cis = []
		for j in range(J):
			start = mapping_cis[j][0]
			end = mapping_cis[j][1]

			## for non-cis- and cis- genes
			if (end - start + 1) == 0:
				temp = np.zeros(N)
				Y_cis.append(temp)
			else:
				X_sub = X[:, start:end+1]
				array_ones = (np.array([np.ones(N)])).T
				X_sub = np.concatenate((X_sub, array_ones), axis=1)						# N x (amount+1)
				beta_sub = beta_cis[k][j]												# 1 x (amount+1)
				Y_sub = np.dot(X_sub, beta_sub)											# 1 x N
				Y_cis.append(Y_sub)
		Y_cis = np.array(Y_cis)														# J x N
		Y_cis = Y_cis.T 															# N x J

		'''
		##=============
		## from batch
		##=============
		beta_batch_reshape = beta_batch.T 											# (B+1) x J
		Y_batch = np.dot(Z, beta_batch_reshape)										# N x J
		'''

		##=============
		## from cell factor (tissue k)
		##=============
		Y_cellfactor = []

		'''
		# first layer
		beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
		m_factor_before = np.dot(X, beta_cellfactor1_reshape)					# N x D

		# logistic twist
		m_factor_after = np.zeros(m_factor_before.shape)
		for n in range(N):
			for d in range(D):
				x = m_factor_before[n][d]
				m_factor_after[n][d] = 1.0 / (1.0 + math.exp(-x))

		# second layer
		array_ones = (np.array([np.ones(N)])).T
		m_factor_new = np.concatenate((m_factor_after, array_ones), axis=1)		# N x (D+1)
		'''

		beta_cellfactor2_reshape = beta_cellfactor2[k].T 						# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new, beta_cellfactor2_reshape)			# N x J

		##=============
		## compile and error cal
		##=============
		Y_final = Y_cis + Y_batch + Y_cellfactor
		list_pos = Y_pos[k]
		Y_final_sub = Y_final[list_pos]
		Y_exp.append(Y_final_sub)


	Y_exp = np.array(Y_exp)
	Tensor_error = Y_exp - Y 				# m_error







	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##==========================================================================================
	## backward prop
	##==========================================================================================
	##=========##=========##=========##=========##=========##=========##=========##=========##==

	N_sample = 0			## total number of samples

	##=============
	## from batch
	##=============
	der_batch = np.zeros(beta_batch.shape)		# J x (B+1)
	''' per individual
	for n in range(N):
		der_batch += np.outer(m_error[n], Z_batch[n])
	'''
	# matrix mul: J x N, N x (B+1)
	#
	Z_full = []
	for k in range(K):
		for pos in Y_pos[k]:
			Z_full.append(Z[pos])
	Z_full = np.array(Z_full)
	#
	m_error = []
	for i in range(len(Tensor_error)):
		for j in range(len(Tensor_error[i])):
			m_error.append(Tensor_error[i][j])
	m_error = np.array(m_error)
	N_sample = len(m_error)
	#
	der_batch = np.dot(m_error.T, Z_full)		# J x (B+1)
	der_batch = der_batch / N_sample


	##=============
	## from cis- (for k tissues)
	##=============
	for k in range(K):
		for j in range(J):
			start = mapping_cis[j][0]
			end = mapping_cis[j][1]

			## for non-cis- and cis- genes
			if (end - start + 1) == 0:
				continue


			der_cis[k][j] = np.zeros(der_cis[k][j].shape)
			#start = mapping_cis[j][0]
			#end = mapping_cis[j][1]
			X_sub = X[Y_pos[k]]
			X_sub = X_sub[:, start:end+1]
			array_ones = (np.array([np.ones(len(X_sub))])).T
			X_sub = np.concatenate((X_sub, array_ones), axis=1)						# N x (amount+1)

			## per individual fashion
			#for n in range(N):
			#	der_cis[k][j] += m_error[n][j] * X_sub[n]
			'''
			der_cis[k][j] = np.dot(m_error[:, j].T, X_sub)
			der_cis[k][j] = der_cis[k][j] / N
			'''
			der_cis[k][j] = np.dot(Tensor_error[k][:, j].T, X_sub)
			der_cis[k][j] = der_cis[k][j] / N_sample



	##=============
	## from cell factor (for all tissues)
	##=============
	##== last layer
	for k in range(K):
		der_cellfactor2[k] = np.zeros(beta_cellfactor2[k].shape)			# J x (D+1)
		## per individual fashion
		#for n in range(size_batch):
		#	der_cellfactor2[k] += np.outer(m_error[n], m_factor_new[n])
		# J x N, N x (D+1)
		#der_cellfactor2[k] = np.dot(m_error.T, m_factor_new)
		#der_cellfactor2[k] = der_cellfactor2[k] / size_batch
		#
		m_factor_new_sub = m_factor_new[Y_pos[k]]
		#
		der_cellfactor2[k] = np.dot(Tensor_error[k].T, m_factor_new_sub)
		der_cellfactor2[k] = der_cellfactor2[k] / N_sample

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
	'''
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
	'''
	## all individual fashion, per tissue additive
	m_factor_der = np.multiply(m_factor_after, 1 - m_factor_after)
	for k in range(K):
		# N x J, J x D --> N x D
		m_temp = np.dot(Tensor_error[k], beta_cellfactor2[k][:, :-1])
		# N x D
		#m_factor_der = np.multiply(m_factor_after, 1 - m_factor_after)
		# N x D, N x D --> N x D
		m_temp = np.multiply(m_temp, m_factor_der[Y_pos[k]])
		# D x N, N x (I+1)
		der_cellfactor1 += np.dot(m_temp.T, X[Y_pos[k]])
	der_cellfactor1 = der_cellfactor1 / N_sample









	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##==========================================================================================
	## gradient descent
	##==========================================================================================
	##=========##=========##=========##=========##=========##=========##=========##=========##==
	beta_cis = beta_cis - rate_learn * der_cis
	beta_cellfactor1 = beta_cellfactor1 - rate_learn * der_cellfactor1
	beta_cellfactor2 = beta_cellfactor2 - rate_learn * der_cellfactor2
	beta_batch = beta_batch - rate_learn * der_batch

	return







##==== calculate the total squared error for all tissues
def cal_error():
	global X, Y, Y_pos, Z
	global mapping_cis
	global beta_cis, beta_cellfactor1, beta_cellfactor2, beta_batch
	global I, J, K, D, B, N

	error_total = 0

	##================================================================================================================
	## cell factor first layer
	# first layer
	beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
	m_factor = np.dot(X, beta_cellfactor1_reshape)							# N x D
	# logistic twist
	for n in range(N):
		for d in range(D):
			x = m_factor[n][d]
			m_factor[n][d] = 1.0 / (1.0 + math.exp(-x))
	# second layer input
	array_ones = (np.array([np.ones(N)])).T
	m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)


	##=============
	## from batch
	##=============
	beta_batch_reshape = beta_batch.T 											# (B+1) x J
	Y_batch = np.dot(Z, beta_batch_reshape)										# N x J
	##================================================================================================================

	for k in range(K):

		##=============
		## from cis- (tissue k)
		##=============
		Y_cis = []
		for j in range(J):
			start = mapping_cis[j][0]
			end = mapping_cis[j][1]

			## for non-cis- and cis- genes
			if (end - start + 1) == 0:
				temp = np.zeros(N)
				Y_cis.append(temp)
			else:
				X_sub = X[:, start:end+1]
				array_ones = (np.array([np.ones(N)])).T
				X_sub = np.concatenate((X_sub, array_ones), axis=1)						# N x (amount+1)
				beta_sub = beta_cis[k][j]												# 1 x (amount+1)
				Y_sub = np.dot(X_sub, beta_sub)											# 1 x N
				Y_cis.append(Y_sub)
		Y_cis = np.array(Y_cis)
		Y_cis = Y_cis.T

		'''
		##=============
		## from batch
		##=============
		beta_batch_reshape = beta_batch.T 											# (B+1) x J
		Y_batch = np.dot(Z, beta_batch_reshape)										# N x J
		'''

		##=============
		## from cell factor (tissue k)
		##=============
		Y_cellfactor = []

		'''
		# first layer
		beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
		m_factor = np.dot(X, beta_cellfactor1_reshape)							# N x D

		# logistic twist
		for n in range(N):
			for d in range(D):
				x = m_factor[n][d]
				m_factor[n][d] = 1.0 / (1.0 + math.exp(-x))

		# second layer
		array_ones = (np.array([np.ones(N)])).T
		m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)
		'''

		beta_cellfactor2_reshape = beta_cellfactor2[k].T 						# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new, beta_cellfactor2_reshape)			# N x J

		##=============
		## compile and error cal
		##=============
		Y_final = Y_cis + Y_batch + Y_cellfactor
		list_pos = Y_pos[k]
		Y_final_sub = Y_final[list_pos]
		error = np.sum(np.square(Y[k] - Y_final_sub))


		##=============
		##=============
		error_total += error
		##=============
		##=============

	return error_total




##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============




if __name__ == "__main__":




	##==== get the training rate
	#rate_learn = float(sys.argv[1])
	#print "rate_learn is:", rate_learn



	print "now training..."



	##========================================================================
	## loading the data, simu
	##========================================================================
	##==== load data (simu)
	#
	X = np.load("./data_simu_data/X.npy")
	# Y and Y_pos
	K = 13										## TODO: specify the number of tissues
	Y = []
	Y_pos = []
	for k in range(K):
		data = np.load("./data_simu_data/Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load("./data_simu_data/Tensor_tissue_" + str(k) + "_pos.npy")
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)
	#
	mapping_cis = np.load("./data_simu_data/mapping_cis.npy")
	#
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
	##========================================================================
	## loading the data, real
	##========================================================================
	##==== load data (real)
	#
	X = np.load("./data_real_data/X.npy")
	# Y and Y_pos
	K = 28										## TODO: specify the number of tissues
	Y = []
	Y_pos = []
	for k in range(K):
		print "tissue#", k
		data = []
		list_pos = []
		file = open("./data_real_data/Tensor_tissue_" + str(k) + ".txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break

			line = line.split('\t')
			pos = int(line[0])
			list_expr = map(lambda x: float(x), line[1:])
			list_pos.append(pos)
			data.append(list_expr)
		file.close()

		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)
	#
	mapping_cis = np.load("./data_real_data/mapping_cis.npy")
	#
	Z = np.load("./data_real_data/Z.npy")


	beta_cis = np.load("./data_real_init/beta_cis.npy")
	beta_cellfactor1 = np.load("./data_real_init/beta_cellfactor1.npy")
	beta_cellfactor2 = np.load("./data_real_init/beta_cellfactor2.npy")
	beta_batch = np.load("./data_real_init/beta_batch.npy")
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






	##==== timer, for speed test
	start_time_total = timeit.default_timer()







	##============
	## train
	##============
	list_error = []
	for iter1 in range(num_iter):
		print "[@@@]working on out iter#", iter1


		##==== timer
		start_time = timeit.default_timer()


		"""
		##============================================
		## DEBUG
		## error before
		error = cal_error()
		print "[error_before] current total error:", error
		##============================================




		forward_backward_gd()
		"""




		##============================================
		error = cal_error()
		print "[error_after] current total error:", error
		##============================================

		list_error.append(error)
		np.save("./result/list_error", np.array(list_error))

		##==== timer
		elapsed = timeit.default_timer() - start_time
		print "time spent on this batch:", elapsed

	print "done!"





	##==== timer, for speed test
	print "speed:", (timeit.default_timer() - start_time_total) / num_iter









