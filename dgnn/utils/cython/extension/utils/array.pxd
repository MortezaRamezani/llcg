# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -std=c++11
# distutils: extra_link_args =

from libcpp.vector cimport vector
from cython cimport Py_buffer

import numpy as np
cimport numpy as np

cdef void npy2vec_int(np.ndarray[int, ndim=1, mode='c'] nda, vector[int] & vec)
cdef void npy2vec_long(np.ndarray[long, ndim=1, mode='c'] nda, vector[long] & vec)
# cdef void npy2vec_float(np.ndarray[int, ndim=1, mode='c'] nda, vector[int] & vec)
# cdef void npy2vec_double(np.ndarray[int, ndim=1, mode='c'] nda, vector[int] & vec)


# https://stackoverflow.com/questions/45133276/passing-c-vector-to-numpy-through-cython-without-copying-and-taking-care-of-me
cdef class ArrayWrapperInt:
    cdef vector[int] vec
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]
    cdef void set_data(self, vector[int] & data)
