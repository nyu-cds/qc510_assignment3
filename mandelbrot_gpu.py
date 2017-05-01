# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import math
import numpy as np
from pylab import imshow, show
@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
    For each thread, a block of elements will iteral over the kernel. The width of 
    the block is the ceil of width of image/number of threads in x, 
    and the hight is the ceil of height of image/number of threads in y.
    '''

    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    thread_x=cuda.gridDim.x*cuda.blockDim.x
    thread_y=cuda.gridDim.y*cuda.blockDim.y

    partition_x=math.ceil(width/thread_x)
    partition_y=math.ceil(height/thread_y)


    x_position = cuda.grid(2)[0]
    y_position=cuda.grid(2)[1]

    begin_x=x_position*partition_x
    begin_y=y_position*partition_y

    finish_x=begin_x+partition_x
    finish_y=begin_y+partition_y

    for x in range(begin_x,finish_x):
        if x>width:
            break
        else:
            real = min_x + x * pixel_size_x
            for y in range(begin_y,finish_y):
                if y > height:
                    break
                else:
                    imag = min_y + y * pixel_size_y
                    image[y, x] = mandel(real, imag, iters)

if __name__ == '__main__':
	image = np.zeros((1024, 1536), dtype = np.uint8)
	blockdim = (8, 8)
	griddim = (32, 8)

	image_global_mem = cuda.to_device(image)
	compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
	image_global_mem.copy_to_host()
	imshow(image)
	show()
