import os
import glob
import sys

sys.path.insert(0, 'project')

import numpy as np
import sparse_coder


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


def compress_feature(fea: np.ndarray, encoder, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    encoder.sparsify(fea).tofile(path)
    return True


def compress(bytes_rate):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    encoder = sparse_coder.SparseVector(out_dim=int(bytes_rate / 2),
                                        in_dim=2048,
                                        min_val=-1,
                                        max_val=6)
    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        compressed_fea_path = os.path.join(compressed_query_fea_dir,
                                           query_basename + '.dat')
        compress_feature(fea, encoder, compressed_fea_path)

    print('Compression Done')
