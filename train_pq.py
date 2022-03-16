import numpy as np
import pq
import pickle
import faiss
from convert_faiss import faiss_to_nanopq

DATA_FILENAME = '/home/huangkun/fast-reid/r50_ibn.npy'

arr = np.load(DATA_FILENAME)
print(np.linalg.norm(arr, axis=1)[:10])
norm = np.linalg.norm(arr, axis=1)
print(np.min(norm), np.max(norm))
arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
print("load data: ", arr.shape, arr.dtype)

print(np.linalg.norm(arr, axis=1)[:10])


# queries = data_dict['queries']
# galleries = data_dict['galleries']
# gallery_names = data_dict['gallery_names']
# query_names = data_dict['query_names']
# features = data_dict['features']

# all_features = np.concatenate((queries, galleries, features), axis=0)

res = faiss.StandardGpuResources()

arch = 'r50'
for M in [64, 128, 256]:

    # pq_codec = pq.PQ(M, Ks=256, verbose=True)
    # pq_codec.fit(arr, iter=20, seed=2)
    # pickle.dump(pq_codec, open(f"./r2_codec_M.pkl", 'wb'))
    d = 2048
    # M = 64
    # M = 128
    n_bits = 8
    pq_example = faiss.IndexPQ (d, M, n_bits)
    pq_example.train(arr)

    pq_example, centers = faiss_to_nanopq(pq_example)
    pickle.dump(pq_example, open(f"./r2_{arch}_{M}.pkl", 'wb'))


