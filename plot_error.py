import matplotlib.pyplot as plt
import numpy as np



if __name__=="__main__":



	##==== total likelihood
	arr = np.load("./result/list_error.npy")




	"""
	arr = []
	file = open("./result/loglike_total_online.txt", 'r')
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		arr.append(float(line))
	file.close()
	"""





	##==== other likelihood terms
	#arr = np.load("result/loglike_data.npy")
	#arr = np.load("result/loglike_Y1.npy")
	#arr = np.load("result/loglike_U1.npy")
	#arr = np.load("result/loglike_V1.npy")
	#arr = np.load("result/loglike_T1.npy")
	#arr = np.load("result/loglike_Y2.npy")
	#arr = np.load("result/loglike_U2.npy")
	#arr = np.load("result/loglike_V2.npy")
	#arr = np.load("result/loglike_T2.npy")
	#arr = np.load("result/loglike_Beta.npy")
	#arr = np.load("result/loglike_alpha.npy")


	#arr = np.load("result/loglike_Y.npy")
	#arr = np.load("result/loglike_U.npy")
	#arr = np.load("result/loglike_V.npy")
	#arr = np.load("result/loglike_T.npy")
	#arr = np.load("result/loglike_alpha.npy")


	#print arr
	plt.plot(arr, 'r')





	plt.xlabel("Number of Batches")
	plt.ylabel("Total squared error in current tissue")
	plt.title("Total squared error v.s. num of batches")
	plt.show()




