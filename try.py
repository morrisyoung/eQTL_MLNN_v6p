import numpy as np






if __name__ == "__main__":



	"""
	data = [[1,2], [3,4,5], [6,7,8,9]]

	for i in range(len(data)):
		data[i] = np.array(data[i])
	data = np.array(data)

	print data * 2
	print type(data)

	print data + 2 * data
	print type(data)
	"""





	data = np.array([[1,2], [3,4],[4,3]])
	print data

	list = [2, 0]
	data1 = data[list]
	print data1
	print type(data1)

	data1 = data[:, :1]
	print data1
	print type(data1)



