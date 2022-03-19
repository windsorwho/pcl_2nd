import os
import glob

import numpy as np
import pickle
import sys

sys.path.insert(0, 'project/')
import pq as pq
import sparse_coder

CODEC_BASENAME = 'project/codebooks/r2_r101_'


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# def write_feature_file(fea: np.ndarray, path: str):
#     assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
#     fea.astype('<f4').tofile(path)
#     return True


def decompress_feature(path, pq_codec, sparse_codec) -> np.ndarray:
    with open(path, 'rb') as f:
        code = np.frombuffer(f.read(), dtype=np.ubyte)
        if code[0] == np.ubyte(255) and code[1] == np.ubyte[255]:
            feature = sparse_codec.densify(code)
        else:
            feature = pq_codec.decode(np.expand_dims(code, axis=0))[0]
    return feature


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    # with open(path, 'wb') as f:
    # f.write(fea.astype('<f4').tobytes())
    fea.astype('<f4').tofile(path)
    return True


def reconstruct(byte_rate):
    DEBUG = False

    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(byte_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(
        byte_rate)
    if DEBUG:
        compressed_query_fea_dir = 'project/' + compressed_query_fea_dir
        reconstructed_query_fea_dir = 'project/' + reconstructed_query_fea_dir

    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    pq_codec = pickle.load(open(f"{CODEC_BASENAME}{byte_rate}.pkl", 'rb'))
    sparse_codec = sparse_coder.SparseVector(out_dim=int(bytes_rate) // 2 - 2,
                                             in_dim=2048,
                                             min_val=-1,
                                             max_val=6)

    compressed_query_fea_paths = glob.glob(
        os.path.join(compressed_query_fea_dir, '*.*'))

    assert (len(compressed_query_fea_paths) != 0)

    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        # reconstructed_fea = reconstruct_feature(compressed_query_fea_path)
        # reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
        # write_feature_file(reconstructed_fea, reconstructed_fea_path)

        reconstructed_fea = decompress_feature(compressed_query_fea_path,
                                               pq_codec, sparse_codec)
        print("reconstructed norm: ",
              np.linalg.norm(reconstructed_fea))  # NOTE:

        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir,
                                              query_basename + '.dat')
        write_feature_file(reconstructed_fea, reconstructed_fea_path)

    print('Reconstruction Done')


if __name__ == '__main__':
    reconstruct(64)
