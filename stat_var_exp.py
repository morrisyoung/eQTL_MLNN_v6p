## calculate the sum of squared deviation from mean




import numpy as np




if __name__ == "__main__":



	"""
	#''' for real data, saved as per tissue txt. file
	K = 28				## NOTE: manually specify this
	Y = []

	for k in range(K):
		print "tissue#", k

		data = []
		file = open("./data_real_data/Tensor_tissue_" + str(k) + ".txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break

			line = line.split('\t')
			list_expr = map(lambda x: float(x), line[1:])
			data.append(list_expr)
		file.close()

		Y += data

	Y = np.array(Y)
	print Y.shape
	ave = np.mean(Y, axis=0)
	print (Y - ave).shape
	error = np.sum(np.square(Y - ave))
	print error
	error = error / (len(Y) * len(Y[0]))
	print error
	"""





	"""
	#''' for real data, saved as per tissue txt. file, calculating tissue ave variance for all tissues
	K = 28				## NOTE: manually specify this
	Y = []
	Y_axis = []
	count = 0

	for k in range(K):
		print "tissue#", k

		data = []
		file = open("./data_real_data/Tensor_tissue_" + str(k) + ".txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break

			line = line.split('\t')
			list_expr = map(lambda x: float(x), line[1:])
			data.append(list_expr)
		file.close()

		Y += data
		Y_axis.append([count, count + len(data)])
		count += len(data)

	Y = np.array(Y)
	print Y.shape
	ave = np.mean(Y, axis=0)
	print (Y - ave).shape

	for k in range(K):
		print "tissue#", k,

		start = Y_axis[k][0]
		end = Y_axis[k][1]
		amount = end - start
		print amount,
		error = np.sum(np.square(Y[start: end] - ave))
		error = error / (amount * len(Y[0]))
		print error
	"""





	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============





	"""
	## for simulated full tensor
	Y = np.load("./data_simu_data/Y.npy")
	data = []
	for k in range(len(Y)):
		print "tissue#", k
		temp = Y[k]
		temp = temp.tolist()
		data += temp

	data = np.array(data)
	ave = np.mean(data, axis=0)
	error = np.sum(np.square(data - ave))
	print "error:", error

	print data.shape
	"""






	## for simulated incomp tensor
	K = 13
	Y = []
	for k in range(K):
		data = np.load("./data_simu_data/Tensor_tissue_" + str(k) + ".npy")
		data = data.tolist()
		Y = Y + data
	Y = np.array(Y)
	print Y.shape
	ave = np.mean(Y, axis=0)
	error = np.sum(np.square(Y - ave))
	print "error:", error









