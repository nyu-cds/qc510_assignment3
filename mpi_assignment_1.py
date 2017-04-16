from mpi4py import MPI

# Extract information about rank 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Print according to rank: Even rank for hello & Odd rank for Goodbye.
if rank % 2 == 0:
    print("Hello from process", rank)
else:
    print("Goodbye from process", rank)