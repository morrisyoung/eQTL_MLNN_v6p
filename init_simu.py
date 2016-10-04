## initialize from full simulated tensor


## we will do the following initialization:
##	1. we assume the cis- effect and trans- effect are equal, so we split the expression tensor and use half to do each
##	2. for the cell factors, we will do PCA for each tissue separately, and average to get the factors; we then re-do the linear system to get the tissue specific parameters
##	3. we twist the cell factors into range(0, 1), and map back them for logistic function
##	4. we then solve the linear system to get the genome-wide factor regulation
##	5. we solve multiple linear systems to get the cis- part
##	6. we then forward propagation and get the residual, and use the averaged residual to initialize the batch effect
##	7. we will always append a random constant effect (the intercept)


## NOTE: now we have a simulated full tensor





import numpy as np
from sklearn.decomposition import PCA





##==== scale of the input data (chr22 and brain sample setting; appropriate for simu test)
I = 14056					# num of SNPs
J = 585						# num of genes
K = 13						# num of tissues
#L = 50000000				# length of chromosome
N = 159						# num of individuals
D = 40						# num of cell factors
B = 10						# num of batch variables



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
init_beta_cis = []				# tensor of (imcomplete) matrix of Genes x cis- SNPs
init_beta_cellfactor1 = []		# matrix of first layer cell factor beta
init_beta_cellfactor2 = []		# tensor (tissue specific) of second layer cell factor beta
init_beta_batch = []			# matrix of Individuals x Batches







if __name__ == "__main__":




	##=====================================================================================================================
	##==== load data (simu)
	##=====================================================================================================================
	X = np.load("./data_simu_data/X.npy")
	Y = np.load("./data_simu_data/Y.npy")
	mapping_cis = np.load("./data_simu_data/mapping_cis.npy")
	Z = np.load("./data_simu_data/Z.npy")

	beta_cis = np.load("./data_simu_data/beta_cis.npy")
	beta_cellfactor1 = np.load("./data_simu_data/beta_cellfactor1.npy")
	beta_cellfactor2 = np.load("./data_simu_data/beta_cellfactor2.npy")
	beta_batch = np.load("./data_simu_data/beta_batch.npy")
	##==== fill dimension
	I = len(X[0])
	J = len(Y[0][0])
	K = len(Y)
	N = len(X)
	D = len(beta_cellfactor1)
	B = len(Z[0])
	print "shape:"
	print I
	print J
	print K
	print N
	print D
	print B

	# make incomplete tensor numpy array at all levels, in order to supprt numpy array computing
	init_beta_cis = []
	for k in range(K):
		init_beta_cis.append([])
		for j in range(J):
			temp = np.zeros(beta_cis[k][j].shape)
			init_beta_cis[k].append(temp)
		init_beta_cis[k] = np.array(init_beta_cis[k])
	init_beta_cis = np.array(init_beta_cis)

	init_beta_cellfactor1 = np.zeros(beta_cellfactor1.shape)
	init_beta_cellfactor2 = np.zeros(beta_cellfactor2.shape)
	init_beta_batch = np.zeros(beta_batch.shape)

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
		X_sub = []
		start = mapping_cis[j][0]
		end = mapping_cis[j][1]
		X_sub = X[:, start:end+1]
		array_ones = (np.array([np.ones(N)])).T
		X_sub = np.concatenate((X_sub, array_ones), axis=1)						# N x (amount+1)

		for k in range(K):
			# the linear system: X_sub x beta = Y_sub
			Y_sub = Y_cis[k][:, j]												# N x 1
			init_beta_sub = np.linalg.lstsq(X_sub, Y_sub)[0]
			init_beta_cis[k][j] = init_beta_sub
	init_beta_cis = np.array(init_beta_cis)
	print "init_beta_cis shape:",
	print init_beta_cis.shape
	print "and three level data type:",
	print type(init_beta_cis)
	print type(init_beta_cis[0])
	print type(init_beta_cis[0][0])






	##=====================================================================================================================
	##==== cell factor
	##=====================================================================================================================
	Y_cellfactor = 0.5 * Y
	##
	## init_beta_cellfactor2
	##
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

		t_factor.append(Y1)
	t_factor = np.array(t_factor)
	m_factor = np.average(t_factor, axis=0)

	## tune factor matrix into [0.1, 0.9]
	value_max = np.amax(m_factor)
	value_min = np.amin(m_factor)
	m_factor_tune = (m_factor - value_min) * (1 / (value_max - value_min))
	m_factor_tune = 0.5 + 0.8 * (m_factor_tune - 0.5)
	array_ones = (np.array([np.ones(N)])).T
	m_factor_tune = np.concatenate((m_factor_tune, array_ones), axis=1)					# N x (D+1)

	for k in range(K):
		Y_sub = Y_cellfactor[k]
		# the linear system: m_factor_tune x beta = Y_sub
		init_beta = np.linalg.lstsq(m_factor_tune, Y_sub)[0].T
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
	Y_batch = []
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

		# first layer NOTE: this is redundent
		init_beta_cellfactor1_reshape = init_beta_cellfactor1.T 				# (I+1) x D
		m_factor = np.dot(X, init_beta_cellfactor1_reshape)						# N x D

		# logistic twist
		for n in range(N):
			for d in range(D):
				x = m_factor[n][d]
				m_factor[n][d] = 1.0 / (1.0 + math.exp(-x))

		# second layer
		array_ones = (np.array([np.ones(N)])).T
		m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)
		init_beta_cellfactor2_reshape = init_beta_cellfactor2[k].T 				# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new, init_beta_cellfactor2_reshape)		# N x J
		print "Y_cellfactor shape:",
		print Y_cellfactor.shape

		##== compile
		temp = Y[k] - (Y_cis + Y_cellfactor)
		Y_batch.append(temp)

	Y_batch = np.array(Y_batch)
	print "the shape of residual expression tensor:",
	print Y_batch.shape

	## averaging:
	Y_batch_ave = np.average(Y_batch, axis=0)

	## the linear system: Z x beta = Y_batch
	init_beta_batch = np.linalg.lstsq(Z, Y_batch_ave)[0].T
	print "init_beta_batch shape:",
	print init_beta_batch.shape









	print "done..."







