import numpy as np
_=np.load('./data/abalone.npy')

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=np.array([3,3,3])

c=a*b.T
print c


a=np.asarray([[1,2,3],[4,5,6]])
b=np.asarray(([1,2,3]))
c=a-b
print c