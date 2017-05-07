from pyspark import SparkContext
from operator import mul

#This product_spark is to find the product of number from 1 to 1,000
if __name__ == '__main__':
sc = SparkContext("local", "product_spark")
    # Create an RDD of numbers range from 1 to 1000
    nums = sc.parallelize(range(1,1001))
    # fold method to calculate.
    print("the product is ", nums.fold(1, mul))