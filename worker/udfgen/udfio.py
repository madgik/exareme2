import numpy as np


def as_tensor_table(array: np.ndarray):
    size = array.size
    shape = array.shape
    indices = np.unravel_index(range(size), shape)
    out = {f"dim{i}": idx for i, idx in enumerate(indices)}
    out["values"] = array.ravel()
    return out


def from_tensor_table(table: dict):
    ndims = len(table) - 1
    multi_index = [table[f"dim{i}"] for i in range(ndims)]
    shape = [max(idx) + 1 for idx in multi_index]
    lin_index = np.ravel_multi_index(multi_index, shape)
    if all(li == i for i, li in enumerate(lin_index)):
        array = table["values"].reshape(shape)
    else:
        array = table["values"][lin_index].reshape(shape)
    return np.array(array)


def as_relational_table(array: np.ndarray, name: str):
    assert len(array.shape) in (1, 2)
    names = (f"{name}_{i}" for i in range(array.shape[1]))
    out = {n: c for n, c in zip(names, array)}
    return out


# class Tensor(np.ndarray):
#     def __new__(cls, input_array):
#         return np.asarray(input_array).view(cls)

#     def to_dict(self):
#         size = self.size
#         shape = self.shape
#         indices = np.unravel_index(range(size), shape)
#         table = {f"dim{i}": idx for i, idx in enumerate(indices)}
#         table['values'] = self.ravel()
#         return table

#     @classmethod
#     def from_dict(cls, table):
#         ndims = len(table) - 1
#         multi_index = [table[f"dim{i}"] for i in range(ndims)]
#         shape = [max(idx) + 1 for idx in multi_index]
#         lin_index = np.ravel_multi_index(multi_index, shape)
#         if all(li == i for i, li in enumerate(lin_index)):
#             array = table['values'].reshape(shape)
#         else:
#             array = table['values'][lin_index].reshape(shape)
#         return cls(array)
