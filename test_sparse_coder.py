import sparse_coder

import glob
import numpy as np


def test_encoding_error(out_length=32, min_val=0.8, max_val=5.0):
    features = read_sample_features()
    coder = sparse_coder.SparseVector(out_dim=out_length,
                                      min_val=min_val,
                                      max_val=max_val)
    error = []
    for x in features:
        x_code = coder.sparsify(x)
        x_hat = coder.densify(x_code)
        error.append(np.linalg.norm(x - x_hat))
    print('MSE for length %d (%f,%f): %f' %
          (out_length, min_val, max_val, np.mean(np.array(error))))


def read_sample_features():
    files = glob.glob('./query_feature_sample/*.dat')
    features = []
    for name in files:
        feature = np.fromfile(name, dtype=np.float32)
        features.append(feature)
    return np.array(features)


if __name__ == '__main__':
    test_encoding_error(out_length=0)
    for i in range(8):
        test_encoding_error(out_length=2**(i + 4) - 2)
        test_encoding_error(out_length=2**(i + 4) - 2, min_val=-1, max_val=6)
