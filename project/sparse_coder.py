import numpy as np

MIN_VAL = 0.8  #0.8696264
MAX_VAL = 5.0  #4.791605


class ScalarQuantizer(object):

    def __init__(self, min_val=MIN_VAL, max_val=MAX_VAL):
        # Only 8 bit encoding only.
        assert max_val >= min_val
        self.num_bits = 8
        self.min_val = min_val
        self.max_val = max_val
        self.value_range = max_val - min_val
        self.unit_length = self.value_range / 255.0

    def to_code(self, num) -> np.uint8:
        if num < self.min_val:
            return np.uint8(0)
        elif num >= self.max_val:
            return np.uint8(256)
        code = np.uint8((num - self.min_val) / self.value_range * 255)
        return code

    def to_float(self, code) -> np.float32:
        value = np.float32(code) * self.unit_length + self.min_val
        return value


class SparseVector(object):

    def __init__(self,
                 out_dim=32,
                 in_dim=2048,
                 min_val=MIN_VAL,
                 max_val=MAX_VAL):
        self.quantizer = ScalarQuantizer(min_val=min_val, max_val=max_val)
        self.out_dim = int(out_dim)
        self.in_dim = int(in_dim)
        assert in_dim >= out_dim

    def sparsify(self, dense: np.ndarray):
        # Sort the array in decreasing order by absolute value, Then only take t
        if not isinstance(dense, np.ndarray):
            raise TypeError("dense is not a numpy array.")
        abs_dense = np.abs(dense)
        positions = np.sort(np.argsort(-abs_dense)[:self.out_dim])
        prev_idx = 0
        sparse = np.zeros(2 * self.out_dim, dtype=np.uint8)
        i = 0
        for slot in range(self.out_dim):
            idx = positions[i]
            delta = idx - prev_idx  #TODO:improve distance encoding as it is always positive.
            if delta <= 255:
                i = i + 1
            else:
                delta = 255
                idx = prev_idx + delta
            code = self.quantizer.to_code(dense[idx])
            sparse[slot * 2] = np.uint8(delta)
            sparse[slot * 2 + 1] = code
            prev_idx = idx
        return sparse

    def densify(self, sparse: np.ndarray) -> np.ndarray:
        if len(sparse) != self.out_dim * 2:
            raise ValueError("densify(): incorrect sparse array size.")
        dense = np.zeros(self.in_dim, dtype=np.float32)
        idx = 0
        for i in range(len(sparse) // 2):
            delta = sparse[i * 2]
            code = sparse[i * 2 + 1]
            idx += delta
            value = self.quantizer.to_float(code)
            dense[idx] = value
        return dense
