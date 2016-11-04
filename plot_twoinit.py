import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines



list_tissue_color = ['k', '#988ED5', 'm', '#8172B2', '#348ABD', '#EEEEEE', '#FF9F9A', '#56B4E9', '#8C0900', '#6d904f', 'cyan', 'red', 'g']
list_tissue = ['tissue#0', 'tissue#1', 'tissue#2', 'tissue#3', 'tissue#4', 'tissue#5', 'tissue#6', 'tissue#7', 'tissue#8', 'tissue#9', 'tissue#10', 'tissue#11', 'tissue#12']



if __name__=="__main__":



	##==== total error
	arr_random = np.load("./result/list_error_10th_random.npy")
	arr_init = np.load("./result/list_error_10th_init.npy")







	##==== random
	#print arr
	print "len of error list:",
	print len(arr_random)
	#print arr[-10:]
	print arr_random[:10]
	print len(list_tissue_color)
	print len(list_tissue)


	## TODO: manually specify something here
	num_iter_out = 100
	num_iter_in = 100
	num_tissue = 13
	count = 0
	for iter1 in range(num_iter_out):
		for k in range(num_tissue):
			x = np.arange(count, count+num_iter_in)
			plt.plot(x, arr_random[x], '-', color=list_tissue_color[k])
			count += num_iter_in




	##==== init
	#print arr
	print "len of error list:",
	print len(arr_init)
	#print arr[-10:]
	print arr_init[:10]
	print len(list_tissue_color)
	print len(list_tissue)


	## TODO: manually specify something here
	num_iter_out = 100
	num_iter_in = 100
	num_tissue = 13
	count = 0
	for iter1 in range(num_iter_out):
		for k in range(num_tissue):
			x = np.arange(count, count+num_iter_in)
			plt.plot(x, arr_init[x], '-', color=list_tissue_color[k])
			count += num_iter_in









	##==== multiple-tissue legend
	list_handle = []
	for k in range(num_tissue):
		line = mlines.Line2D([], [], color=list_tissue_color[k], label=list_tissue[k])
		list_handle.append(line)
	plt.legend(handles=list_handle)



	plt.text(60000, 160000000, 'judicious initialization', fontsize=15, style='italic')
	plt.text(20000, 120000000, 'random initialization', fontsize=15, style='italic')



	plt.xlabel("number of updates")
	plt.ylabel("total squared error in current tissue")
	plt.title("total squared error v.s. num of updates")
	plt.grid()
	plt.show()







