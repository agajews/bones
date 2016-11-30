import tensorflow as tf
import numpy as np
from collections import OrderedDict
import random
from pprint import pprint


def unique(l):
    return list(OrderedDict.fromkeys(l))


def flatten(l):
    res = []
    for sub in list(l):
        res += sub
    return res


def mean(l):
    total = 0
    length = 0
    for elem in l:
        total += elem
        length += 1
    return total / length


# def flat_tensors(tensors):
#     if len(tensors) == 0:
#         return []
#     all_tensors = tensors + flatten([flat_tensors([i for i in t.op.inputs])
#                                      for t in tensors])
#     return unique(all_tensors)


# def flat_ops(ops):
#     if len(ops) == 0:
#         return []
#     all_ops = ops + flatten([flat_ops([i.op for i in op.inputs])
#                              for op in ops])
#     return unique(all_ops)


# def all_variables(tensors):
#     if not isinstance(tensors, list):
#         tensors = [tensors]
#     # tensors = flat_tensors(tensors)
#     ops = flat_ops([t.op for t in tensors])
#     variables = [o.outputs[0] for o in ops if o.type == 'Variable']
#     return variables


def prod(l):
    p = 1
    for sub in l:
        p *= sub
    return p


def shuffled(l):
    return random.sample(l, len(l))


def zip_dicts(a, b):
    return {a[k]: b[k] for k in a.keys()}


def to_one_hot(labels, num_categories):
    '''Takes a numpy array of labels and turns it
    into a one-hot numpy array of higher dimension'''
    data = np.zeros(labels.shape + (num_categories,), dtype=labels.dtype)
    for indices, val in np.ndenumerate(labels.astype('int32')):
        data[indices + (val,)] = 1
    return data


def batches(xs, size, processors=None):
    if processors is None:
        processors = [lambda x: x] * len(xs)
    ordering = shuffled(range(min([len(x) for x in xs])))
    for i in range(0, len(ordering), size):
        yield [processors[j](x[i:i + size]) for j, x in enumerate(xs)]


def for_batches(fn, xs, size, steps=None, processors=None):
    if steps is None:
        for batch_num, batch in enumerate(batches(xs, size, processors)):
            fn(batch_num, *batch)
    else:
        all_batches = batches(xs, size, processors)
        epoch = 0
        for step in range(steps):
            try:
                batch = next(all_batches)
            except StopIteration:
                all_batches = batches(xs, size, processors)
                batch = next(all_batches)
                epoch += 1
            fn(step, epoch, *batch)


def for_epochs(fn, xs, size, epochs=None, processors=None):
    for epoch in range(epochs):
        for batch_num, batch in enumerate(batches(xs, size, processors)):
            fn(batch_num, epoch, *batch)


def map_batches(fn, xs, size, processors=None):
    for batch in batches(xs, size, processors):
        yield fn(*batch)


def mean_batches(fn, xs, size=1024, processors=None):
    return mean(map_batches(fn, xs, size, processors))


def n_eq(arr_a, arr_b):
    return np.count_nonzero(arr_a == arr_b)


def cat_acc(model_i, x, y):
    y_hat = np.argmax(model_i(x), axis=1)
    y = np.argmax(y, axis=1)
    return n_eq(y_hat, y) / len(y)


def batch_cat_acc(model_i, x, y, size=1024, processors=None):
    return mean_batches(
        lambda batch_x, batch_y: cat_acc(model_i, batch_x, batch_y),
        [x, y], size, processors)


def call_wrap(outputs, placeholders, sess=None):
    if sess is None:
        sess = tf.get_default_session()
    if not isinstance(placeholders, list) and \
       not isinstance(placeholders, dict):
        placeholders = [placeholders]
    if isinstance(placeholders, list):
        def fn(*inputs):
            return sess.run(outputs,
                feed_dict={p: i for p, i in zip(placeholders, inputs)})
    elif isinstance(placeholders, dict):
        def fn(**inputs):
            return sess.run(outputs,
                feed_dict=zip_dicts(placeholders, inputs))
    else:
        raise Exception('Invalid placeholders input')
    return fn


def placeholder_like(arr, name=None, dynamic_batch=True, dtype=None):
    if isinstance(arr, np.ndarray):
        arr_dtype = tf.as_dtype(arr.dtype)
        shape = list(arr.shape)
        if dynamic_batch:
            shape = [None] + shape[1:]
    else:
        arr_dtype = arr.dtype
        shape = arr.get_shape()
    if dtype is None:
        dtype = arr_dtype
    return tf.placeholder(dtype, shape, name=name)
