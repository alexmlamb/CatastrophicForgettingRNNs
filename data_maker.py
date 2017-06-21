import numpy as np

def anbn():
    return -1*np.asarray(4*[[0,1,0,1,0,1]]).astype('float32')

def cndn():
    return -1*np.asarray(4*[[2,3,2,3,2,3]]).astype('float32')

'''


'''

def keyvalue():

    mat = np.asarray([[0,2,1,3,0,2],[0,3,1,2,0,3],[1,3,0,2,1,3],[1,2,0,3,1,2]])

    return mat.astype('float32')


print keyvalue().shape
print anbn().shape



