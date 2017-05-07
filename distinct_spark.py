from pyspark import SparkContext
import re

#This distinct_spark is to count the number of distinct key in file. 

def splitter(line):
	line = re.sub(r'^\W+|\W+$', '', line)
	return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
	#define a sc
	sc = SparkContext("local", "distinct_word_count")
	#load file
	text = sc.textFile('pg2701.txt')
	#apply splitter to the text
	words = text.flatMap(splitter)
	#count the number for each word 
	words_pairs = words.map(lambda x: (x,1))
	#Only count the number of distinct words
	counts = words_pairs.distinct() 
	print("Number of distinct words are:", counts.count())