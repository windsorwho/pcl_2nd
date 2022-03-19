import os
import glob

import numpy as np

import sys

sys.path.insert(0, 'project/')
import pq as pq
import pickle
import sparse_coder

CODEC_BASENAME = 'project/codebooks/r2_r101_'
FEATURE_NORM_THRESHOLD = 1.3


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    # return np.fromfile(path, dtype='<f4')
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea


def compress_feature(fea, pq_codec, sparse_codec, path):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    # fea.astype('<f4')[: target_bytes // 4].tofile(path)
    fea = fea.reshape(1, -1)
    # print(f"fea: {fea.shape} {np.linalg.norm(fea)}")
    if np.linalg.norm(fea) < FEAT_NORM_THRESHOLD:
        code = pq_codec.encode(fea).astype(np.ubyte)
    else:
        code = sparse_codec.sparsify(fea)
    with open(path, 'wb') as f:
        f.write(code.tobytes())

    return 1


def compress(bytes_rate):
    DEBUG = False

    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)

    # pq coder
    pq_codec = pickle.load(open(f"{CODEC_BASENAME}{bytes_rate}.pkl", 'rb'))
    sparse_codec = sparse_coder.SparseVector(out_dim=bytes_rate // 2 - 2,
                                             in_dim=2048,
                                             min_val=-1,
                                             max_val=6)

    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    if DEBUG:
        query_fea_dir = 'project/' + query_fea_dir
        compressed_query_fea_dir = 'project/' + compressed_query_fea_dir

    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))

    assert (len(query_fea_paths) != 0)
    print(len(query_fea_paths))

    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        print(np.linalg.norm(fea))
        compressed_fea_path = os.path.join(compressed_query_fea_dir,
                                           query_basename + '.dat')
        compress_feature(fea, pq_codec, sparse_codec, compressed_fea_path)

    print('Compression Done')


if __name__ == '__main__':
    compress(64)
