'''
Qianyu Cheng
qc510
'''

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Get the rank of a certain process
size = comm.Get_size() # Get the size of processes

def parallel_sort(N): # Sending divided list into various processors and finally combining them to get sorted result
	divided_lists = None
	parallel_sorted = None
	if rank == 0:

		# Generate a large unsorted data set (e.g. 10,000 elements)
		random_data = np.random.randint(0, N, N)
		range_data = np.arange(max(random_data) + 1)
		divisions = np.array_split(range_data, size)
		divided_lists = []
		for i in range(size):
			divided_lists.append([number for number in random_data if (number in divisions[i])])
		# Slice it into bins by value and send each bin (except one)
	scattered_lists = comm.scatter(divided_lists, root=0) 
	# Divided lists are scattered into different processors
	gathered_lists = comm.gather(np.sort(scattered_lists), root=0)

	if rank == 0:
		parallel_sorted = np.concatenate(gathered_lists) # Concatenate all sorted lists
	return parallel_sorted

if __name__ == '__main__':
	N = 10
	print(parallel_sort(N))