import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines




''' raw data:
speed testing benchmarks:
change data scale (f40): GPU, Python
[#6] 5%, 50000, 1000, 177500000: 0.062sec, 0.375394240893sec
[#7] 7.5%, 75000, 1500, 266250000: 0.092sec, 0.552201977693sec
[#8] 10%, 100000, 2000, 355000000: 0.116sec, 0.643225245292sec
[#9] 15%, 150000, 3000, 532500000: 0.198sec, 1.12621707311sec
[#10] 20%, 200000, 4000, 710000000: 0.228sec, 1.47105268533sec

change number of factors (10% of real data): GPU, Python
[#11] 20: 0.117sec, 0.741367831414sec
[#12] 30: 0.115sec, 0.76264275074sec
[#13] 40: 0.119sec, 0.883213886114sec
[#14] 50: 0.119sec, 0.940753720724sec
[#15] 60: 0.118sec, 0.883177567629sec
'''




if __name__ == "__main__":




	## figure#1
	plt.figure(1)
	plt.subplot(121)
	x = np.arange(5)
	x = np.array([5, 7.5, 10, 15, 20])
	y1 = [375.394240893, 552.201977693, 883.213886114, 1126.21707311, 1471.05268533]
	y2 = [62, 92, 116, 198, 228]
	plt.plot(x, y1, '--*', color='r', markersize=8)
	plt.plot(x, y2, '--o', color='g', markersize=8)
	plt.axis([0, 25, 0, 1500])
	python = mlines.Line2D([], [], color='r', marker='*', markersize=8, linestyle='--', label='Python (Numpy)')
	gpu = mlines.Line2D([], [], color='g', marker='o', markersize=8, linestyle='--', label='C++ (CUDA)')
	plt.legend(handles=[python, gpu], loc=2)
	#plt.legend(handles=[gpu], loc=2)
	plt.xlabel("% of real data scale (for genotype and gene)")
	plt.ylabel("milliseconds per update (SGD, 20 samples)")
	plt.grid()



	## figure#2
	plt.subplot(122)
	x = np.array([20, 30, 40, 50, 60])
	y1 = [741.367831414, 762.64275074, 883.213886114, 940.753720724, 883.177567629]
	y2 = [117, 115, 116, 119, 118]
	plt.plot(x, y1, '--*', color='r', markersize=8)
	plt.plot(x, y2, '--o', color='g', markersize=8)
	plt.axis([10, 70, 0, 1500])
	python = mlines.Line2D([], [], color='r', marker='*', markersize=8, linestyle='--', label='Python (Numpy)')
	gpu = mlines.Line2D([], [], color='g', marker='o', markersize=8, linestyle='--', label='C++ (CUDA)')
	plt.legend(handles=[python, gpu], loc=2)
	plt.xlabel("number of factors")
	#plt.ylabel("seconds per iteration")	
	plt.grid()


	plt.show()





