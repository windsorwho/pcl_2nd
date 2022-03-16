import os
import glob

import numpy as np

import sys
sys.path.insert(0, 'project/')
import pq as pq
import pickle

CODEC_BASENAME = 'project/r2_r50_'

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    # return np.fromfile(path, dtype='<f4')
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea


def compress_feature(fea, codec, path ):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    # fea.astype('<f4')[: target_bytes // 4].tofile(path)
    fea = fea.reshape(1, -1)
    # print(f"fea: {fea.shape} {np.linalg.norm(fea)}")
    code = codec.encode(fea).astype(np.ubyte)
    with open(path, 'wb') as f:
        f.write(code.tobytes())

    return 1


def compress(bytes_rate):
    DEBUG=False

    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    # pq coder
    pq_codec = pickle.load(open(f"{CODEC_BASENAME}{bytes_rate}.pkl", 'rb'))

    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    if DEBUG:
        query_fea_dir = 'project/' + query_fea_dir
        compressed_query_fea_dir = 'project/' + compressed_query_fea_dir

    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    print(query_fea_dir)
    assert(len(query_fea_paths) != 0)

    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        compress_feature(fea, pq_codec, compressed_fea_path)

    print('Compression Done')


if __name__ == '__main__':
    compress(64)
