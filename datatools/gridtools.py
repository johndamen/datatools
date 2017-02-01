import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


__all__ = []


def expose(o):
    __all__.append(o.__name__)
    return o


@expose
def aggregate(A, blocksize):
    if A.ndim != 2:
        raise NotImplementedError('only 2D arrays supported')

    shape_in = A.shape
    shape_out = [item // blocksize for item in shape_in]
    block_shape = shape_out + [blocksize] * A.ndim

    itemstrides = (shape_in[1] * blocksize, blocksize, shape_in[1], 1)
    bytestrides = np.array(itemstrides) * A.itemsize

    return ast(A, shape=block_shape, strides=bytestrides).reshape(shape_out + [-1])


@expose
def block_mean(A, blocksize):
    return aggregate(A, blocksize).mean(axis=2)


@expose
def block_func(A, blocksize, fn):
    try:
        return fn(aggregate(A, blocksize), axis=2)
    except TypeError as e:
        if 'keyword argument' in e.args[0]:
            return np.apply_along_axis(fn, 2, aggregate(A, blocksize))
        else:
            raise
