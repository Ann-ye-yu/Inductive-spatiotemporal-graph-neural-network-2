import numpy as np
import scipy.sparse as ssp


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        # print(csr_matrix.toarray())
        # print(f"csr_matrix.toarray:\n{csr_matrix.toarray()}")
        # print(f"csr_matrix.indptr:\n{csr_matrix.indptr}")
        # print(f"csr_matrix.data:\n{csr_matrix.data}")
        # print(f"csr_matrix.indices:\n{csr_matrix.indices}")
        data = []
        indices = []
        indptr = []
        '''
        csr_matrix是按照行存储的稀疏矩阵:
        indices：list-> 存储的是稀疏矩阵所有行非0元素对应的列索引值组成的数组
        indptr: list->第一个元素为0，之后每个元素表示稀疏矩阵中每行非零元素个数的累计结果,
        data: list->按行优先存储稀疏矩阵的所有非零元素的值
        
        csr_matrix第i行非零元素的列号为indices[indptr[i]:indptr[i+1]], indptr[i+1]-indptr[i]是第i行非零元素的个数
        相应的值为data[indptr[i]:indptr[i+1]]
        '''
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])  # 第row_start行非零元素
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)  # {list:6040},每一个元素都是一个array([])列表，存储每一行非零元素
        self.indices = np.array(indices, dtype=object)  # {list:6040},每一个元素都是一个array([]）列表，存储的是每一行非零元素的索引
        self.indptr = np.array(indptr, dtype=object)  # {list:6040},是一个[]，存储的每一行非零元素个数，indptr:[49 116 ....]
        self.shape = csr_matrix.shape  # shape: (6040,3706)

    # A[list(fringe)=1419].indices
    def __getitem__(self, row_selector):
        # subgraph = Arow [u_nodes][:, v_nodes]
        # print(f"row_selector:\n{row_selector}")
        # print(f"self.indices:\n{self.indices}")
        # print(f"self.indices[row_selector]:\n {self.indices[row_selector]}")
        indices = np.concatenate(self.indices[row_selector])  # 拼接indices[row_selector]为一个列表
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]  # shape的行为用户的个数，列为item的数目
        return ssp.csr_matrix((data, indices, indptr), shape=shape)  # shape[101,3706]


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []
        """
        csr_matrix是按照列存储的稀疏矩阵；
        indices： 存储的是稀疏矩阵所有列非0元素对应的行索引值组成的数组
        indptr: 第一个元素为0，之后每个元素表示稀疏矩阵中每列非零元素个数的累计结果
        data: 按列优先存储稀疏矩阵的所有非零元素的值
        csr_matrix第i列非零元素的行号为indices[indptr[i]:indptr[i+1]], indptr[i+1]-indptr[i]是第i列非零元素的个数
        相应的值为data[indptr[i]:indptr[i+1]]
        """
        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)  # {ndarray:(3706,)},[array([]),array([]),...],存储的是每一列的非零元素的值
        self.indices = np.array(indices, dtype=object)  # {ndarray:(3706,)},[array([]),array([]),...],存储的每一列非零元素的索引
        self.indptr = np.array(indptr, dtype=object)  # {ndarray:(3706,)},[1858,619,.....],存储的是每一列非零元素的个数累计值
        self.shape = csc_matrix.shape  # shape:(6040,3706)

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)