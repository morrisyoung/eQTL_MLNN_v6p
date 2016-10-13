import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines



list_tissue_color = ['k', '#988ED5', 'm', '#8172B2', '#348ABD', '#EEEEEE', '#FF9F9A', '#56B4E9', '#8C0900', '#6d904f', 'cyan', 'red', 'g']
list_tissue = ['tissue#0', 'tissue#1', 'tissue#2', 'tissue#3', 'tissue#4', 'tissue#5', 'tissue#6', 'tissue#7', 'tissue#8', 'tissue#9', 'tissue#10', 'tissue#11', 'tissue#12']



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
	print "len of error list:",
	print len(arr)
	#print arr[-10:]
	print arr[:10]
	print len(list_tissue_color)
	print len(list_tissue)
	#plt.plot(arr, 'r')




	## TODO: manually specify something here
	num_iter_out = 7
	num_iter_in = 100
	num_tissue = 13
	count = 0
	for iter1 in range(num_iter_out):
		for k in range(num_tissue):
			x = np.arange(count, count+num_iter_in)
			plt.plot(x, arr[x], '-', color=list_tissue_color[k])
			count += num_iter_in




	## the legend
	list_handle = []
	for k in range(num_tissue):
		line = mlines.Line2D([], [], color=list_tissue_color[k], label=list_tissue[k])
		list_handle.append(line)
	plt.legend(handles=list_handle)











	plt.xlabel("number of batches")
	plt.ylabel("total squared error in current tissue")
	plt.title("total squared error v.s. num of batches")
	plt.grid()
	plt.show()







