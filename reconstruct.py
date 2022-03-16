import os
import glob

import numpy as np
import pickle
import sys
sys.path.insert(0, 'project/')
import pq as pq
CODEC_BASENAME = 'project/r2_r50_'

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# def write_feature_file(fea: np.ndarray, path: str):
#     assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
#     fea.astype('<f4').tofile(path)
#     return True

def decompress_feature(path, codec) -> np.ndarray:
    with open(path, 'rb') as f:
        code = np.frombuffer(f.read(), dtype=np.ubyte)
        feature = codec.decode(np.expand_dims(code, axis=0))
    return feature[0]



def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(fea.astype('<f4').tobytes())
    return True

def reconstruct(byte_rate):
    DEBUG=False

    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(byte_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(byte_rate)
    if DEBUG:
        compressed_query_fea_dir = 'project/' + compressed_query_fea_dir
        reconstructed_query_fea_dir = 'project/' + reconstructed_query_fea_dir

    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    pq_codec = pickle.load(open(f"{CODEC_BASENAME}{byte_rate}.pkl", 'rb'))
    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))

    assert(len(compressed_query_fea_paths) != 0)

    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        # reconstructed_fea = reconstruct_feature(compressed_query_fea_path)
        # reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
        # write_feature_file(reconstructed_fea, reconstructed_fea_path)

        reconstructed_fea = decompress_feature(compressed_query_fea_path,
                                               pq_codec)
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir,
                                              query_basename + '.dat')
        write_feature_file(reconstructed_fea, reconstructed_fea_path)

    print('Reconstruction Done')


if __name__ == '__main__':
    reconstruct(64)
