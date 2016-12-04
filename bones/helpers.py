import tensorflow as tf
import numpy as np
from collections import OrderedDict
import random
from .core import *
import json
from pprint import pprint


def unique(l):
    return list(OrderedDict.fromkeys(l))


def iter_flatten(l):
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
    step = 0
    for epoch in range(epochs):
        for batch in batches(xs, size, processors):
            fn(step, epoch, *batch)
            step += 1


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


def call_wrap(outputs, positional, keyword=None, sess=None,
              defaults=None, invariants=None):
    if sess is None:
        sess = tf.get_default_session()
    if keyword is None:
        keyword = {}
    if defaults is None:
        defaults = {}
    if invariants is None:
        invariants = {}
    if isinstance(positional, dict):
        keyword.update(positional)
        positional = []
    if not isinstance(positional, list):
        positional = [positional]
    for key, val in list(keyword.items()):
        if not isinstance(key, str):
            keyword.pop(key)
            invariants[key] = val

    def fn(*inputs, **kwinputs):
        feed_dict = {}
        all_kwinputs = kwinputs.copy()
        all_kwinputs.update(defaults)
        feed_dict.update(zip_dicts(keyword, all_kwinputs))
        feed_dict.update(invariants)
        feed_dict.update({p: i for p, i in zip(positional, inputs)})
        return sess.run(outputs, feed_dict=feed_dict)
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


def scalar_placeholder(dtype=tf.float32, name=None):
    return tf.placeholder(dtype, name=name)


def vars_to_dict(var_scope=None, sess=None, run=True, tolist=False):
    if sess is None:
        sess = tf.get_default_session()
    variables = scope_variables(var_scope)
    vars_dict = {}
    for variable in variables:
        parent = vars_dict
        for subscope in variable.name.split('/')[:-1]:
            if subscope not in parent:
                parent[subscope] = {}
            parent = parent[subscope]
        arr = variable
        if run:
            arr = sess.run(arr)
            if tolist:
                arr = arr.tolist()
        parent[variable.name.split('/')[-1]] = arr
    return vars_dict


def _vars_assign_dict(dest_vars, vars_dict, sess=None, run=True):
    if sess is None:
        sess = tf.get_default_session()
    if isinstance(dest_vars, dict) and isinstance(vars_dict, dict):
        res = {}
        for name in vars_dict.keys():
            res[name] = _vars_assign_dict(dest_vars[name], vars_dict[name],
                                          sess, run)
        return res
    else:
        op = dest_vars.assign(np.asarray(vars_dict))
        if run:
            sess.run(op)
        return op


def dict_name_shape(dictionary):
    if not isinstance(dictionary, dict):
        return
    if not any([isinstance(val, dict) for key, val in dictionary.items()]):
        return list(dictionary.keys())
    else:
        return {key: dict_name_shape(val) for key, val in dictionary.items()}


def vars_assign_dict(var_scope, vars_dict, sess=None, run=True):
    dest_vars = vars_to_dict(var_scope, sess, run=False)
    print('Assigning variables:')
    pprint(dict_name_shape(dest_vars))
    return _vars_assign_dict(dest_vars, vars_dict, sess, run)


def save_vars_json(fnm, var_scope=None, sess=None):
    with open(fnm, 'w') as f:
        f.write(json.dumps(vars_to_dict(var_scope, sess, tolist=True)))


def load_vars_json(fnm, var_scope=None, sess=None):
    with open(fnm, 'r') as f:
        vars_assign_dict(var_scope, json.loads(f.read()), sess)


def train(model, loss, optim, x_in, y_in, x, y, test_x=None, test_y=None,
          var_scope=None, train_dict=None, test_dict=None, sess=None,
          epochs=None, steps=None, print_freq=500, batch_size=128,
          print_fn=None):
    # print(test_dict)
    model_fn = call_wrap(model, x_in,
                         None if test_dict is None else test_dict.copy())
    loss_fn = call_wrap(loss, [x_in, y_in],
                        None if test_dict is None else test_dict.copy())
    train_fn = call_wrap([loss, optim], [x_in, y_in],
                         None if train_dict is None else train_dict.copy())
    initialize_vars(var_scope=var_scope, sess=sess)
    if print_fn is None:
        def print_fn(step, epoch, curr_loss, model_fn):
            print('Loss at step {:d}, epoch {:d}: {:0.4f}'
                  .format(step, epoch, curr_loss))

    def update(step, epoch, batch_x, batch_y):
        curr_loss, _ = train_fn(batch_x, batch_y)
        if step % print_freq == 0:
            print_fn(step, epoch, curr_loss, model_fn)
    if epochs is not None:
        for_epochs(update, [x, y], batch_size, epochs=epochs)
    else:
        for_batches(update, [x, y], batch_size, steps=steps)
    print('Final loss: {:0.4f}'.format(
        mean_batches(loss_fn, [x, y])))
    print('Train accuracy: {:0.2f}%'.format(
        batch_cat_acc(model_fn, x, y) * 100))
    if test_x is not None and test_y is not None:
        print('Test accuracy: {:0.2f}%'.format(
            batch_cat_acc(model_fn, test_x, test_y) * 100))
