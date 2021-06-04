# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp


from ..utils cimport array
from libc.stdio cimport printf
from libc.stdlib cimport rand, srand
from posix.stdlib cimport rand_r
from libc.time cimport time
from libcpp cimport bool
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort, unique, lower_bound
from libcpp.map cimport map

from cython.parallel import prange, parallel
from cython.operator import dereference, postincrement
from cython cimport Py_buffer

from libcpp.vector cimport vector

import torch
from torch_sparse import SparseTensor
import numpy as np

cimport numpy as np
cimport cython


def sample_neighbors(adj, batch_nodes, int num_neighbors, int num_proc=2, bool replace=False):

    # Adjacency to Cython
    cdef long[:] indptr = adj.storage.rowptr().numpy()
    cdef long[:] indices = adj.storage.col().numpy()

    # Batch to Cython
    numpy_batch_nodes = batch_nodes.numpy().astype(np.int32)
    cdef vector[int] c_batch_nodes
    array.npy2vec_int(numpy_batch_nodes, c_batch_nodes)
    cdef int num_batch_nodes = c_batch_nodes.size()

    # Parallel Sampling Stuff
    cdef vector[vector[int]] sampled_node_ids
    cdef vector[unordered_map[int, int]] sampled_node_id_map
    cdef vector[vector[int]] sampled_cols
    cdef vector[vector[int]] sampled_rowptr

    sampled_node_ids = vector[vector[int]](num_proc)
    sampled_node_id_map = vector[unordered_map[int, int]](num_proc)
    sampled_cols = vector[vector[int]](num_proc)
    sampled_rowptr = vector[vector[int]](num_proc)

    # Split the minibatch for parallel sampling (using numpy to avoid non-divisible cases)
    tmp_batch_split = np.array_split(np.arange(num_batch_nodes), num_proc)
    batch_split = np.asarray(
        [bs[0] for bs in tmp_batch_split] + [num_batch_nodes], dtype=np.int32)
    cdef vector[int] c_batchsplit
    array.npy2vec_int(batch_split, c_batchsplit)

    cdef int p = 0
    cdef int start_bid
    cdef int end_bid
    with nogil, parallel(num_threads=num_proc):
        for p in prange(num_proc, schedule='static'):
            start_bid = c_batchsplit[p]
            end_bid = c_batchsplit[p+1] - 1
            c_sample_neighbors2(indptr, indices, c_batch_nodes,
                                num_neighbors, p, start_bid, end_bid, replace,
                                sampled_node_ids, sampled_node_id_map, sampled_cols, sampled_rowptr)

    # Merge different threads
    cdef unordered_map[int, int] merged_sampled_nodes_map
    cdef vector[int] merged_rowptr, merged_cols

    cdef int row_id, n_id
    cdef unsigned int i, j

    for i in range(c_batch_nodes.size()):
        merged_sampled_nodes_map[c_batch_nodes[i]] = i

    j = c_batch_nodes.size()
    for p in range(num_proc):
        for i in range(sampled_node_ids[p].size()):
            n_id = sampled_node_ids[p][i]
            if merged_sampled_nodes_map.count(n_id) == 0:
                merged_sampled_nodes_map[n_id] = j
                c_batch_nodes.push_back(n_id)
                j += 1

    merged_rowptr.push_back(0)
    j = 0
    for p in range(num_proc):
        for i in range(sampled_cols[p].size()):
            n_id = merged_sampled_nodes_map[sampled_cols[p][i]]
            merged_cols.push_back(n_id)
        for i in range(1, sampled_rowptr[p].size()):
            merged_rowptr.push_back(sampled_rowptr[p][i] + j)
        j = merged_cols.size()

    cdef array.ArrayWrapperInt output_rowptr = array.ArrayWrapperInt()
    cdef array.ArrayWrapperInt output_cols = array.ArrayWrapperInt()
    cdef array.ArrayWrapperInt output_nids = array.ArrayWrapperInt()

    output_rowptr.set_data(merged_rowptr)
    output_cols.set_data(merged_cols)
    output_nids.set_data(c_batch_nodes)

    np_output_rowptr = np.frombuffer(output_rowptr, dtype=np.int32)
    np_output_cols = np.frombuffer(output_cols, dtype=np.int32)
    np_output_nids = np.frombuffer(output_nids, dtype=np.int32)

    rowptr = torch.from_numpy(np_output_rowptr).type(torch.LongTensor)
    cols = torch.from_numpy(np_output_cols).type(torch.LongTensor)
    nid = torch.from_numpy(np_output_nids).type(torch.LongTensor)
    sizes = (rowptr.size(0)-1, nid.size(0))

    sampled_adj = SparseTensor(
        rowptr=rowptr, col=cols, sparse_sizes=sizes, is_sorted=False)

    return sampled_adj, nid



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void c_sample_neighbors2(long[:] & indptr, long[:] & indices, vector[int] & batch_nodes,
                              int num_neighbors, int p, int start_bid, int end_bid, bool replace,
                              vector[vector[int]] & sampled_node_ids,
                              vector[unordered_map[int, int]
                                     ] & sampled_node_id_map,
                              vector[vector[int]] & sampled_cols,
                              vector[vector[int]] & sampled_rowptr,
                              ) nogil:

    cdef int idx = 0
    cdef int i, b, j, z
    cdef int c, e = 0
    cdef int row_start, row_end, row_count
    cdef int start_ptr, end_ptr
    
    cdef unsigned int seed = time(NULL)

    cdef unordered_set[int] perm
    cdef unordered_set[int].iterator perm_it

    sampled_node_ids[p].resize(end_bid-start_bid)

    sampled_rowptr[p].push_back(0)

    j = 0
    for i in range(start_bid, end_bid+1, 1):
        b = batch_nodes[i]
        # sampled_node_ids[p].push_back(b)
        sampled_node_ids[p][j] = b
        sampled_node_id_map[p][b] = j
        j += 1

    # Sampling without replacement using RF Algorithm
    # Adapted from: https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/cpu/sample_cpu.cpp
    j = 0
    for i in range(start_bid, end_bid+1, 1):
        b = batch_nodes[i]
        row_start = indptr[b]
        row_end = indptr[b+1]
        row_count = row_end - row_start
        start_ptr = sampled_cols[p].size()

        perm.clear()
        if row_count <= num_neighbors:
            for z in range(row_count):
                perm.insert(z)
        else:
            for z in range(row_count - num_neighbors, row_count):
                if not (perm.insert(rand_r(&seed) % z).second):
                    perm.insert(z)

        perm_it = perm.begin()
        while perm_it != perm.end():
            b = dereference(perm_it)
            postincrement(perm_it)

            e = row_start + b
            c = indices[e]

            if sampled_node_id_map[p].count(c) == 0:
                sampled_node_id_map[p][c] = sampled_node_ids[p].size()
                sampled_node_ids[p].push_back(c)

            sampled_cols[p].push_back(c)

        sampled_rowptr[p].push_back(
            sampled_rowptr[p][j] + sampled_cols[p].size() - start_ptr)
        j += 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void c_sample_neighbors(long[:] & indptr, long[:] & indices, vector[int] & batch_nodes,
                             int num_neighbors, int p, int start_bid, int end_bid, bool replace,
                             vector[vector[int]] & sampled_node_ids,
                             vector[unordered_map[int, int]
                                    ] & sampled_node_id_map,
                             vector[vector[int]] & sampled_cols,
                             vector[vector[int]] & sampled_rowptr,
                             ) nogil:

    # srand(time(NULL))

    cdef int idx = 0
    cdef int i, b, j, z
    cdef int c, e = 0
    cdef int row_start, row_end, row_count
    cdef int start_ptr, end_ptr

    cdef unordered_set[int] perm
    cdef unordered_set[int].iterator perm_it

    sampled_rowptr[p].push_back(0)

    j = 0
    for i in range(start_bid, end_bid+1, 1):
        b = batch_nodes[i]
        sampled_node_ids[p].push_back(b)
        sampled_node_id_map[p][b] = j
        j += 1

    # No sampling (Not used)
    if num_neighbors < 0:
        j = 0
        for i in range(start_bid, end_bid+1, 1):
            b = batch_nodes[i]
            row_start = indptr[b]
            row_end = indptr[b+1]
            row_count = row_end - row_start
            start_ptr = sampled_cols[p].size()
            for b in range(row_count):
                e = row_start + b
                c = indices[e]

                if sampled_node_id_map[p].count(c) == 0:
                    sampled_node_id_map[p][c] = sampled_node_ids[p].size()
                    sampled_node_ids[p].push_back(c)

                sampled_cols[p].push_back(c)

            sampled_rowptr[p].push_back(
                sampled_rowptr[p][j] + sampled_cols[p].size() - start_ptr)
            j += 1

    # Sampling with replacement
    elif replace:
        j = 0
        for i in range(start_bid, end_bid+1, 1):
            b = batch_nodes[i]
            row_start = indptr[b]
            row_end = indptr[b+1]
            row_count = row_end - row_start
            start_ptr = sampled_cols[p].size()
            for b in range(num_neighbors):
                # e = row_start + b
                e = row_start + rand() % row_count
                c = indices[e]

                if sampled_node_id_map[p].count(c) == 0:
                    sampled_node_id_map[p][c] = sampled_node_ids[p].size()
                    sampled_node_ids[p].push_back(c)

                sampled_cols[p].push_back(c)

            sampled_rowptr[p].push_back(
                sampled_rowptr[p][j] + sampled_cols[p].size() - start_ptr)
            j += 1

    # Sampling without replacement using RF Algorithm
    # Adapted from: https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/cpu/sample_cpu.cpp
    else:
        j = 0
        for i in range(start_bid, end_bid+1, 1):
            b = batch_nodes[i]
            row_start = indptr[b]
            row_end = indptr[b+1]
            row_count = row_end - row_start
            start_ptr = sampled_cols[p].size()

            perm.clear()
            if row_count <= num_neighbors:
                for z in range(row_count):
                    perm.insert(z)
            else:
                for z in range(row_count - num_neighbors, row_count):
                    if not (perm.insert(rand() % z).second):
                        perm.insert(z)

            perm_it = perm.begin()
            while perm_it != perm.end():
                b = dereference(perm_it)
                postincrement(perm_it)

                e = row_start + b
                c = indices[e]

                if sampled_node_id_map[p].count(c) == 0:
                    sampled_node_id_map[p][c] = sampled_node_ids[p].size()
                    sampled_node_ids[p].push_back(c)

                sampled_cols[p].push_back(c)

            sampled_rowptr[p].push_back(
                sampled_rowptr[p][j] + sampled_cols[p].size() - start_ptr)
            j += 1



# Output Cython data
# cdef unordered_map[int, int].iterator it
# print('--'*30)
# print('batch_nodes: ', end='')
# for i in range(c_batch_nodes.size()):
#     print(c_batch_nodes[i], end=' ')
# print()
# for p in range(num_proc):
#     print(p, end=': \n')
#     print('node_ids: ', end='')
#     for i in range(sampled_node_ids[p].size()):
#         print(sampled_node_ids[p][i], end=' ')
#     print('\nnode_id_map:')
#     it = sampled_node_id_map[p].begin()
#     while (it != sampled_node_id_map[p].end()):
#         print(dereference(it).first,'->', dereference(it).second)
#         postincrement(it)
#     print('cols: ', end='')
#     for i in range(sampled_cols[p].size()):
#         print(sampled_cols[p][i], end=' ')
#     print('\nrowptr: ', end='')
#     for i in range(sampled_rowptr[p].size()):
#         print(sampled_rowptr[p][i], end=' ')
#     print('')
#     print('--'*30)
# for i in range(c_batch_nodes.size()):
#     print(c_batch_nodes[i], end=' ')
# print()

# # Final output
# print('cols: ', end='')
# for i in range(merged_cols.size()):
#     print(merged_cols[i], end=' ')
# print('\nrowptr: ', end='')
# for i in range(merged_rowptr.size()):
#     print(merged_rowptr[i], end=' ')
# print()
