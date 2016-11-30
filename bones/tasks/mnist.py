from urllib.request import urlretrieve

import gzip

import os

import numpy as np

from bones.helpers import to_one_hot


def download(data_dir, filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, os.path.join(data_dir, filename))


def load_images(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        download(data_dir, filename)
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return data / np.float32(256)


def load_labels(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        download(data_dir, filename)
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return to_one_hot(data, 10)


def load_mnist(base_dir='data', data_dir='mnist', test=False, flat=True):
    data_dir = os.path.join(base_dir, data_dir)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if not test:
        x = load_images(data_dir, 'train-images-idx3-ubyte.gz')
        y = load_labels(data_dir, 'train-labels-idx1-ubyte.gz')
    else:
        x = load_images(data_dir, 't10k-images-idx3-ubyte.gz')
        y = load_labels(data_dir, 't10k-labels-idx1-ubyte.gz')
    if not flat:
        x = np.reshape(x, [x.shape[0], 28, 28, 1])
    return x, y
