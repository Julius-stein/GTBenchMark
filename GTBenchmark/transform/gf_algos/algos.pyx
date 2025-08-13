# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 编译与安全优化建议
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np  # 提供 np.int64_t 等 C 端 dtype

from cython.parallel cimport prange, parallel

def floyd_warshall(adjacency_matrix):
    """
    输入:
      adjacency_matrix: numpy.ndarray，形状 [N, N]，布尔或0/1整型都可
    返回:
      (M, path): 两个 np.int64 的 [N, N] 矩阵
        M[i,j]   = 最短路长度；不可达为 510
        path[i,j]= 中转点 k；不可达为 510；若 i->j 直接相连或 i==j，则为 -1
    """

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    # —— 关键修复：NumPy dtype 用 np.int64，而不是 long ——
    adj_mat_copy = adjacency_matrix.astype(np.int64, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']

    # —— 关键修复：在 ndarray 声明里用 np.int64_t ——
    cdef np.ndarray[np.int64_t, ndim=2, mode='c'] M = adj_mat_copy
    cdef np.ndarray[np.int64_t, ndim=2, mode='c'] path = -1 * np.ones([n, n], dtype=np.int64)

    cdef unsigned int i, j, k
    cdef np.int64_t M_ij, M_ik, cost_ikkj

    # —— 关键修复：指针类型与 dtype 对齐（np.int64_t* 而不是 long*）——
    cdef np.int64_t* M_ptr = &M[0, 0]
    cdef np.int64_t* M_i_ptr
    cdef np.int64_t* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0
            elif M[i, j] == 0:
                M[i, j] = 510

    # floyd algo
    for k in range(n):
        M_k_ptr = M_ptr + n * k
        for i in range(n):
            M_i_ptr = M_ptr + n * i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i, j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i, j] >= 510:
                path[i, j] = 510
                M[i, j] = 510

    return M, path


def get_all_edges(path, i, j):
    cdef int k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):
    """
    输入:
      max_dist: int，最远 hop 数
      path:  np.int64 [N,N]，来自 floyd_warshall
      edge_feat: np.int64 [N,N,Fe]，每条有向边的离散特征编码 (+1 后，0 为 pad)
    返回:
      edge_fea_all: np.int64 [N, N, max_dist, Fe]，沿一条最短路逐跳堆叠的边特征
    """

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    # —— 关键修复：dtype 用 np.int64；保持 C 连续 ——
    path_copy = path.astype(np.int64, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(np.int64, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef np.ndarray[np.int64_t, ndim=4, mode='c'] edge_fea_all = \
        -1 * np.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=np.int64)

    cdef unsigned int i, j, k, num_path

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i, j] == 510:   # 不可达
                continue
            # —— 关键修复：避免与参数名 path 冲突，换名 path_seq ——
            path_seq = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path_seq) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path_seq[k], path_seq[k + 1], :]

    return edge_fea_all
