import numpy as np
import sys
sys.path.insert(0, 'project')
import pq
import pickle
import faiss
from convert_faiss import faiss_to_nanopq

# DATA_FILENAME = '/home/huangkun/reid_feature/r50_ibn.npy'

arch = 50
arch = 101

DATA_FILENAME = f'/home/huangkun/tr/features/resnet{arch}_all.npy'

arr = np.load(DATA_FILENAME)

norm = np.linalg.norm(arr, axis=1)
print(np.min(norm), np.max(norm))

arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
print("load data: ", arr.shape, arr.dtype)

print(np.linalg.norm(arr, axis=1)[:10])


res = faiss.StandardGpuResources()


for M in [64, 128, 256]:
    # pq_codec = pq.PQ(M, Ks=256, verbose=True)
    # pq_codec.fit(arr, iter=20, seed=2)
    # pickle.dump(pq_codec, open(f"./r2_codec_M.pkl", 'wb'))

    d = 2048
    n_bits = 8
    pq_example = faiss.IndexPQ (d, M, n_bits)
    pq_example.train(arr)

    pq_example, centers = faiss_to_nanopq(pq_example)
    pickle.dump(pq_example, open(f"codebooks/r2_r{arch}_{M}.pkl", 'wb'))


