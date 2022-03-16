import pickle

# import pq
from convert_faiss import faiss_to_nanopq

import numpy as np

M = 128
pq_exmaple = pickle.load(open(f"./r2_{M}.pkl", 'rb'))


DATA_FILENAME = '/home/huangkun/fast-reid/feats.npy'

arr = np.load(DATA_FILENAME)

X_code = pq_exmaple.encode(vecs=arr[:10000])
X_reconstructed = pq_exmaple.decode(codes=X_code)

print(arr[:3, :50])
print(X_reconstructed[:3, :50])
