"""Portable numeric-only NumPy pickle arrays across NumPy 1.x/2.x.

The isolated HHTools Python uses NumPy 2; Isaac Python uses NumPy 1. Do not
upgrade either runtime or monkeypatch numpy._core names. Payload values remain
ordinary numpy arrays after decoding with the existing MotionDataManager.
"""
import pickle
from pathlib import Path
import numpy as np


def restore_array(data, dtype, shape):
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError("object arrays are not supported reference data")
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)


class PortableArrayPickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise ValueError("object array in reference")
            array = np.ascontiguousarray(obj)
            return restore_array, (array.tobytes(), array.dtype.str, array.shape)
        if isinstance(obj, np.generic):
            value = obj.item()
            return type(value), (value,)
        return NotImplemented


def dump_portable(path, payload):
    with Path(path).open("xb") as stream:
        PortableArrayPickler(stream, protocol=4).dump(payload)
