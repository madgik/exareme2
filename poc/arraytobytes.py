import re
import numpy as np


def make_header(shape, dtype):
    hd = f"SHAPE({','.join([str(s) for s in shape])})DTYPE({dtype})"
    hd = bytes(hd, encoding='ascii')
    tl = bytearray(b'\x00' * 128)
    tl[:len(hd)] = hd
    return tl


def array_to_bytes(x: np.ndarray):
    header = make_header(x.shape, x.dtype)
    bytes_array = header + x.tobytes()
    return np.frombuffer(bytes_array)


def array_from_bytes(b):
    b = b.tobytes()
    hd, tl = b[:128], b[128:]
    hd = hd.decode('ascii')
    shape = re.search(r'SHAPE\(([\d,]*)\)', hd)
    shape = shape.group(1)
    shape = tuple([int(val) for val in shape.split(',')])
    dtype = re.search(r'DTYPE\(([a-z0-9]*)\)', hd)
    dtype = dtype.group(1)
    return np.frombuffer(tl, dtype=dtype).reshape(shape)


x = np.array([[1, 2, 3], [10, 20, 30]], dtype=int)
print(x)
y = array_to_bytes(x)
print(array_from_bytes(y))
print()

x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
print(x)
y = array_to_bytes(x)
print(array_from_bytes(y))
print()
