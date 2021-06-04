# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -std=c++11
# distutils: extra_link_args =


cdef void npy2vec_int(np.ndarray[int, ndim=1, mode='c'] nda, vector[int] & vec):
    cdef int size = nda.size
    vec.assign(& (nda[0]), & (nda[0]) + size)

cdef void npy2vec_long(np.ndarray[long, ndim=1, mode='c'] nda, vector[long] & vec):
    cdef int size = nda.size
    vec.assign(& (nda[0]), & (nda[0]) + size)

cdef class ArrayWrapperInt:
    cdef void set_data(self, vector[int] & data):
        self.vec.swap(data)

    def __getbuffer__(self, Py_buffer * buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])
        self.shape[0] = self.vec.size()
        self.strides[0] = sizeof(int)
        buffer.buf = <char * > & (self.vec[0])
        buffer.format = 'i'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.vec.size() * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer * buffer):
        pass
