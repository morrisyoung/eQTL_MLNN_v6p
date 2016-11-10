## mapping the genes to their cis region (indices) on the whole-genome snp list

## dependency:
##	1. "Gene_list.npy"
##	2. "gene_tss_gencode.v19.v6p.txt"
##	3. "./data_real_data/genotype_450_dosage_matrix_qc_trim/chr*/SNP_info.txt"

import numpy as np



def reformat_mapping_cis(matrix, filename):
	shape = matrix.shape
	dimension1 = shape[0]
	dimension2 = shape[1]

	file = open(filename, 'w')
	for i in range(dimension1):
		for j in range(dimension2):
			value = matrix[i][j]
			if j != (dimension2-1):
				file.write(str(value) + '\t')
			else:
				file.write(str(value))
		file.write('\n')
	file.close()
	return



if __name__ == "__main__":


	##============================================================================================================
	##==== gene list
	list_gene = np.load("./data_real_data/Gene_list.npy")
	print "there are # of genes:", len(list_gene)


	##==== gene tss repo
	rep_gene_tss = {}
	file = open("./data_real_data/gene_tss_gencode.v19.v6p.txt", 'r')
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split('\t')
		if line[1] == 'X' or line[1] == 'Y' or line[1] == 'MT':
			continue

		gene = line[0]
		chr = int(line[1])
		tss = int(line[2])
		rep_gene_tss[gene] = (chr, tss)
	file.close()


	##==== snp pos list
	count = 0
	list_snp_pos = []		# matrix
	for i in range(22):
		list_snp_pos.append([])

		chr = i+1
		file = open("./data_real_data/genotype_450_dosage_matrix_qc/chr" + str(chr) + "/SNP_info.txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break

			line = line.split(' ')
			pos = int(line[1])
			list_snp_pos[-1].append(pos)
		file.close()

		print len(list_snp_pos[-1])
		count += len(list_snp_pos[-1])
	print count





	##============================================================================================================
	##==== cal the original range (cis- mapping information)
	print "mapping the cis- region of all genes..."
	mapping_cis = []
	for j in range(len(list_gene)):
		gene = list_gene[j]
		chr = rep_gene_tss[gene][0]
		tss = rep_gene_tss[gene][1]
		I = len(list_snp_pos[chr-1])

		start = 0
		end = 0

		index = 0
		while index < I:
			if abs(list_snp_pos[chr-1][index] - tss) <= 1000000:				# NOTE: define the cis- region
				start = index
				break

			index += 1

		## no cis- region genes
		if index == I:
			mapping_cis.append((0, -1))
			continue

		while 1:
			if abs(list_snp_pos[chr-1][index] - tss) > 1000000:					# NOTE: define the cis- region
				end = index - 1
				break

			if index == (I - 1):
				end = index
				break

			index += 1

		mapping_cis.append((start, end))
	mapping_cis = np.array(mapping_cis)


	##==== mapping back to the whole-genome list
	for j in range(len(list_gene)):

		## non-cis gene
		start = mapping_cis[j][0]
		end = mapping_cis[j][1]
		if (end - start + 1) == 0:
			continue

		## cis- gene
		gene = list_gene[j]
		chr = rep_gene_tss[gene][0]

		for i in range(chr-1):
			mapping_cis[j][0] += len(list_snp_pos[i])
			mapping_cis[j][1] += len(list_snp_pos[i])

	print "mapping_cis shape:",
	print mapping_cis.shape
	np.save("./data_real_data/mapping_cis", mapping_cis)
	reformat_mapping_cis(mapping_cis, "./data_real_data/mapping_cis.txt")




	##==== test
	list_amount = []
	for i in range(len(mapping_cis)):
		pair = mapping_cis[i]
		list_amount.append(pair[1] - pair[0] + 1)
	list_amount = np.array(list_amount)

	print np.amin(list_amount)
	print np.amax(list_amount)





