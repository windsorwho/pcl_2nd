import os
import glob
import sys

sys.path.insert(0, 'project')

import numpy as np
import sparse_coder


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


def reconstruct_feature(path: str) -> np.ndarray:
    fea = np.fromfile(path, dtype=np.uint8)
    return fea


def reconstruct(bytes_rate):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(
        bytes_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    compressed_query_fea_paths = glob.glob(
        os.path.join(compressed_query_fea_dir, '*.*'))
    encoder = sparse_coder.SparseVector(out_dim=int(bytes_rate) // 2,
                                        in_dim=2048,
                                        min_val=-1,
                                        max_val=6)
    assert (len(compressed_query_fea_paths) != 0)
    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        reconstructed_fea = reconstruct_feature(compressed_query_fea_path)
        reconstructed_fea = encoder.densify(reconstructed_fea)
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir,
                                              query_basename + '.dat')
        write_feature_file(reconstructed_fea, reconstructed_fea_path)

    print('Reconstruction Done')
