import numpy as np


cimport numpy as np
DTYPE = np.int

ctypedef np.int_t DTYPE_t

def collapse_row(np.ndarray[DTYPE_t, ndim=1] row):

    if not row.any():
        return row
    
    cdef int rowlen = row.shape[0]
    cdef int i, j, k
    for i in range(rowlen-1):
        if row[i] == 0:
            row[i:-1] = row[i+1:]
            row[-1] = 0
    for j in range(rowlen-1):
        if row[j] == row[j+1]:
            row[j] += row[j+1]
            row[j+1] = 0
    for k in range(rowlen-1):
        if row[k] == 0:
            row[k:-1] = row[k+1:]
            row[-1] = 0
        

def move_left(np.ndarray[DTYPE_t, ndim=2] mat):
    cdef int y
    cdef int y_max = mat.shape[0]
    for y in range(y_max):
        collapse_row(mat[y])
