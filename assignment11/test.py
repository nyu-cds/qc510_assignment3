from mpi4py import MPI
import numpy as np
import unittest
from parallel_sorter import parallel_sort

class test_(unittest.TestCase):

	'''
	Test class for assignment 11 
	'''
	def set(self):
		pass

	def test_length(self): 
		self.assertTrue(len(parallel_sort(5)) == 5)
		# test the length of test is correct or not 

if __name__ == '__main__':
	unittest.main()