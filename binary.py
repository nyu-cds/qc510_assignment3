import itertools 

def zbits(n,k):
    zeros = "0" * k
    ones = "1" * (n-k)
    binary = ones+zeros
    string = {''.join(i) for i in itertools.permutations(binary, n)}
    return(string)


assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}