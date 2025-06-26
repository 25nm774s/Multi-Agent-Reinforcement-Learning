import numpy as np

def Sigma(start,end):
    res=0
    for i in range(start,end):
        res+=i
    
    return res

arr=np.array([2,Sigma(1,11)])

print(arr)