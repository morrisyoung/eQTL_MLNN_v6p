## simulate the model:
##	1. the cell factor pathway, with logit func for nonlinear effects
##	2. the cis- refulation
##	3. the batch effects (for now, individual specific)
## all pathway will all have intercept term (the constant effects)



import numpy as np
import math
import timeit




##====================================================================================================
##==== scale of data reference
##==== scale of chr22 and brain samples
'''
K = 13
I = 159
J = 585
S = 14056
D1 = 40
D2 = 40
'''
'''
# NOTE: the following are for the whole-genome and all samples (old)
K = 33
I = 450
J = 21150
S = 824113
D1 = 400
D2 = 400
'''
##====================================================================================================



##==== scale of the input data (real data)
I = 824113					# num of SNPs
J = 21150					# num of genes
K = 33						# num of tissues
L = 3000000000				# length of chromosome
N = 450						# num of individuals
D = 400						# num of cell factors
B = 20						# num of batch variables
'''
##==== scale of the input data (10% of real data)
I = 100000					# num of SNPs
J = 2000					# num of genes
K = 13						# num of tissues
L = 355000000				# length of chromosome
N = 159						# num of individuals
D = 40						# num of cell factors
B = 10						# num of batch variables
'''
'''
##==== scale of the input data (chr22 and brain sample setting; appropriate for simu test)
I = 14056					# num of SNPs
J = 585						# num of genes
K = 13						# num of tissues
L = 50000000				# length of chromosome
N = 159						# num of individuals
D = 40						# num of cell factors
B = 10						# num of batch variables
'''






##==== variables
## NOTE: here we assume one chromosome model
X = []						# matrix of Individuals x SNPs
X_pos = []					# list of pos of SNPs
Y = []						# tensor of gene expression
Y_pos = []					# list of pos of genes
mapping_cis = []			# list of (index start, index end)
Z = []						# matrix of Individuals x Batches
## NOTE: the following have the intercept term
beta_cis = []				# tensor of (imcomplete) matrix of Genes x cis- SNPs
beta_cellfactor1 = []		# matrix of first layer cell factor beta
beta_cellfactor2 = []		# tensor (tissue specific) of second layer cell factor beta
beta_batch = []				# matrix of Individuals x Batches








##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============


##================
##==== simu data
##================
## simulate SNPs, and SNP pos
def simu_snp():
	global X
	global N, I

	print "simu X"

	# X
	for n in range(N):
		X.append([])
		for i in range(I):
			# simu
			dosage = np.random.random_sample() * 2
			X[n].append(dosage)
	X = np.array(X)
	print "simulated X shape:",
	print X.shape

	return



def simu_snp_pos():
	global X_pos
	global I, L

	print "simu X_pos"

	# X_pos
	X_pos = []
	repo_temp = {}			# redundancy removing
	while 1:
		if len(X_pos) == I:
			break

		# simu
		pos = np.random.randint(0, L)
		if pos in repo_temp:
			continue
		else:
			X_pos.append(pos)
			repo_temp[pos] = 1
	X_pos = np.array(X_pos)
	X_pos = np.sort(X_pos)
	print "simulated X_pos shape:",
	print X_pos.shape
	#print "and X_pos:",
	#print X_pos

	return



def simu_gene_pos():
	global Y_pos
	global J

	print "simu Y_pos"

	# Y_pos
	Y_pos = []
	repo_temp = {}			# redundancy removing
	while 1:
		if len(Y_pos) == J:
			break

		# simu
		pos = np.random.randint(0, L)
		if pos in repo_temp:
			continue
		else:
			Y_pos.append(pos)
			repo_temp[pos] = 1
	Y_pos = np.array(Y_pos)
	Y_pos = np.sort(Y_pos)
	print "simulated Y_pos shape:",
	print Y_pos.shape
	#print "and Y_pos:",
	#print Y_pos

	return



## calculate cis- mapping information
def cal_snp_gene_map():
	global mapping_cis, X_pos, Y_pos
	global I, J

	print "mapping the cis- region of all genes..."

	# mapping_cis
	mapping_cis = []
	for j in range(J):
		start = 0
		end = 0

		index = 0
		while 1:
			if abs(X_pos[index] - Y_pos[j]) <= 1000000:					# NOTE: define the cis- region
				start = index
				break

			index += 1

		while 1:
			if abs(X_pos[index] - Y_pos[j]) > 1000000:					# NOTE: define the cis- region
				end = index - 1
				break

			if index == (I - 1):
				end = index
				break

			index += 1

		mapping_cis.append((start, end))
	mapping_cis = np.array(mapping_cis)
	print "simulated mapping_cis shape:",
	print mapping_cis.shape
	print "and the mapping_cis:",
	print mapping_cis

	return



## simulating individual associated batch variables
def simu_batch():
	global Z
	global N, B

	# Z
	Z = []
	for n in range(N):
		Z.append([])
		for b in range(B):
			# simu
			batch = np.random.random_sample() * 2
			Z[n].append(batch)
	Z = np.array(Z)
	print "simulated Z shape:",
	print Z.shape
	print "and the Z:",
	print Z

	return






##================
##==== simu para (NOTE: we have the intercept term anywhere)
##================
## simulating cis beta
def simu_beta_cis():
	global beta_cis
	global J, K, mapping_cis

	print "simu beta_cis..."

	# beta_cis
	beta_cis = []

	for k in range(K):
		print "tissue#", k

		beta_cis.append([])
		for j in range(J):
			beta_cis[k].append([])
			amount = mapping_cis[j][1] - mapping_cis[j][0] + 1 + 1				# NOTE: intercept
			for index in range(amount):
				# simu
				beta = np.random.normal()
				beta_cis[k][j].append(beta)
			beta_cis[k][j] = np.array(beta_cis[k][j])
		beta_cis[k] = np.array(beta_cis[k])
		print "simulated beta_cis[k] shape:",
		print beta_cis[k].shape

	beta_cis = np.array(beta_cis)
	print "simulated beta_cis shape:",
	print beta_cis.shape

	return



## simulating cell factor beta
def simu_beta_cellfactor():
	global beta_cellfactor1, beta_cellfactor2
	global D, I, J, K

	print "simu beta_cellfactor1..."

	# beta_cellfactor1
	beta_cellfactor1 = []
	for d in range(D):
		beta_cellfactor1.append([])
		amount = I + 1														# NOTE: intercept
		for index in range(amount):
			# simu
			beta = np.random.normal()
			beta_cellfactor1[d].append(beta)
	beta_cellfactor1 = np.array(beta_cellfactor1)
	print "simulated beta_cellfactor1 shape:",
	print beta_cellfactor1.shape

	print "simu beta_cellfactor2..."

	# beta_cellfactor2
	beta_cellfactor2 = []
	for k in range(K):
		beta_cellfactor2.append([])
		for j in range(J):
			beta_cellfactor2[k].append([])
			amount = D + 1													# NOTE: intercept
			for index in range(amount):
				# simu
				beta = np.random.normal()
				beta_cellfactor2[k][j].append(beta)
	beta_cellfactor2 = np.array(beta_cellfactor2)
	print "simulated beta_cellfactor2 shape:",
	print beta_cellfactor2.shape

	return



## simulating batch effects
def simu_beta_batch():
	global beta_batch
	global J, B

	print "simu beta_batch..."

	# beta_batch
	beta_batch = []
	for j in range(J):
		beta_batch.append([])
		amount = B + 1														# NOTE: intercept
		for index in range(amount):
			# simu
			beta = np.random.normal()
			beta_batch[j].append(beta)
	beta_batch = np.array(beta_batch)
	print "simulated beta_batch shape:",
	print beta_batch.shape

	return






##================
##==== simu output
##================
## simulate genes, and gene pos
## this is called at the very end
def simu_gene():
	global X, Y, Z
	global mapping_cis
	global beta_cis, beta_cellfactor1, beta_cellfactor2, beta_batch
	global I, J, K, D, B, N

	Y = []

	##=============
	## from batch
	##=============
	array_ones = (np.array([np.ones(N)])).T
	Z_new = np.concatenate((Z, array_ones), axis=1)								# N x (B+1)
	beta_batch_reshape = beta_batch.T 											# (B+1) x J
	Y_batch = np.dot(Z_new, beta_batch_reshape)									# N x J
	print "Y_batch shape:",
	print Y_batch.shape



	##=============
	## the other two
	##=============
	for k in range(K):
		print "working on tissue#", k

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
			beta_sub = beta_cis[k][j]												# 1 x (amount+1)
			Y_sub = np.dot(X_sub, beta_sub)											# 1 x N
			Y_cis.append(Y_sub)
		Y_cis = np.array(Y_cis)
		Y_cis = Y_cis.T
		print "Y_cis shape:",
		print Y_cis.shape

		##=============
		## from cell factor
		##=============
		Y_cellfactor = []

		# first layer
		array_ones = (np.array([np.ones(N)])).T
		X_new = np.concatenate((X, array_ones), axis=1)							# N x (I+1)
		beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
		m_factor = np.dot(X_new, beta_cellfactor1_reshape)						# N x D

		# logistic twist
		for n in range(N):
			for d in range(D):
				x = m_factor[n][d]
				m_factor[n][d] = 1.0 / (1.0 + math.exp(-x))

		# second layer
		array_ones = (np.array([np.ones(N)])).T
		m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)
		beta_cellfactor2_reshape = beta_cellfactor2[k].T 						# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new, beta_cellfactor2_reshape)			# N x J
		print "Y_cellfactor shape:",
		print Y_cellfactor.shape

		##== compile
		Y_final = Y_cis + Y_cellfactor + Y_batch
		Y.append(Y_final)

	Y = np.array(Y)
	print "the shape of final expression tensor:",
	print Y.shape

	return






##================
##==== main
##================
if __name__ == "__main__":



	print "now simulating..."
	##==== timer
	start_time = timeit.default_timer()

	##====================================================
	## simu real data
	##====================================================
	##==== simu data
	simu_snp()
	simu_snp_pos()
	simu_gene_pos()
	cal_snp_gene_map()
	simu_batch()

	##==== simu model
	simu_beta_cis()
	simu_beta_cellfactor()
	simu_beta_batch()


	## NOTE: for genome-wide small signal:
	beta_cellfactor1 = beta_cellfactor1 / 10


	##==== cpmpile
	simu_gene()

	##==== save data
	np.save("./data_simu_data/X", X)
	np.save("./data_simu_data/X_pos", X_pos)
	np.save("./data_simu_data/Y", Y)
	np.save("./data_simu_data/Y_pos", Y_pos)
	np.save("./data_simu_data/mapping_cis", mapping_cis)
	np.save("./data_simu_data/Z", Z)
	np.save("./data_simu_data/beta_cis", beta_cis)
	np.save("./data_simu_data/beta_cellfactor1", beta_cellfactor1)
	np.save("./data_simu_data/beta_cellfactor2", beta_cellfactor2)
	np.save("./data_simu_data/beta_batch", beta_batch)



	##====================================================
	## simu another copy as the init (randomly use another copy to init -- we can of course init more wisely)
	##====================================================
	##==== simu model
	simu_beta_cis()
	simu_beta_cellfactor()
	simu_beta_batch()


	## NOTE: for genome-wide small signal:
	beta_cellfactor1 = beta_cellfactor1 / 10


	##==== save data
	np.save("./data_simu_init/beta_cis", beta_cis)
	np.save("./data_simu_init/beta_cellfactor1", beta_cellfactor1)
	np.save("./data_simu_init/beta_cellfactor2", beta_cellfactor2)
	np.save("./data_simu_init/beta_batch", beta_batch)






	##==== timer
	elapsed = timeit.default_timer() - start_time
	print "time spent:", elapsed







