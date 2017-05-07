from pyspark import SparkContext
from operator import add
import re
#squareroot of all the numbers and then calculate the average of them all 
if __name__ == '__main__': 
    sc = SparkContext("local", "squareroot average")

    nums = sc.parallelize(range(1, 1001))
    roots = nums.map(lambda x: x ** 1/2)
    sum_all = roots.fold(0, add)
    count = roots.count()
    average = sum_all/count

    print("Average of square root from 0 to 1001:", average)