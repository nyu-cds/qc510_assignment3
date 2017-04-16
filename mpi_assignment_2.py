from mpi4py import MPI
import numpy as np

# Extract information about rank 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set up buffer(or data)
buff = np.ones(1,dtype=int)

# Raise error if size is less than two 

if(size <2):
	raise ValueError('Please re-enter a process number')

if(rank == 0):
	input_is_valid = False
	while not input_is_valid:
		# Continue if input is a integer; if not, keep asking user to input a valid input
		try:
			user_input = int(input('Please enter a input for process 0'))
		except ValueError as err:
			print('Please re-enter a input that is an integer')
			continue
		# Continue if input is less than 100; if not, keep asking user to input a valid input
		if(user_input>=100):
			print('Please re-enter a input that is an integer less than 100')
		else:
			input_is_valid = True
	# When we validate the input, continue with input and buffer. 
	# Multiple the buffer with user input.
	buff = buff * user_input
		
	comm.Send(buff, dest=1)
	# Receive result from the last process
	comm.Recv(buff, source = size - 1) 
	print(buff[0])
else:
	comm.Recv(buff, source = rank - 1)
	buff = buff * rank
	# For last process, send the value to process 0. 
	if(rank == size -1):
		comm.Send(buff, dest = 0)
	# For all processes with rank other than 0, buffer will be multiple by the rank and sent to the next process
	else:
		comm.Send(buff, dest=rank+1)
	
