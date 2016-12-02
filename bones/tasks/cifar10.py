from urllib.request import urlretrieve
import tarfile
import os
import numpy as np
import pickle as pk
from bones.helpers import to_one_hot


def download(data_dir, filename='cifar-10-python.tar.gz',
             source='http://www.cs.toronto.edu/~kriz/'):
    if not os.path.exists(os.path.join(data_dir, filename)):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, os.path.join(data_dir, filename))


def untar(filename):
    with tarfile.open(filename) as f:
        f.extractall(path=os.path.dirname(filename))


def fetch(data_dir, filename='cifar-10-python.tar.gz',
          source='http://www.cs.toronto.edu/~kriz/'):
    download(data_dir, filename, source)
    untar(os.path.join(data_dir, filename))


def load_images(data_dir, filename):
    path = os.path.join(data_dir, 'cifar-10-batches-py', filename)
    with open(path, 'rb') as f:
        return np.asarray(pk.load(f, encoding='latin1')['data'])


def load_labels(data_dir, filename):
    path = os.path.join(data_dir, 'cifar-10-batches-py', filename)
    with open(path, 'rb') as f:
        labels = np.asarray(pk.load(f, encoding='latin1')['labels'])
        return to_one_hot(labels, 10)


def load_cifar(base_dir='data', data_dir='cifar10', test=False, flat=True):
    data_dir = os.path.join(base_dir, data_dir)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    fetch(data_dir)
    if not test:
        x = np.concatenate([load_images(data_dir, 'data_batch_{:d}'.format(i))
                            for i in range(1, 6)], axis=0)
        y = np.concatenate([load_labels(data_dir, 'data_batch_{:d}'.format(i))
                            for i in range(1, 6)], axis=0)
    else:
        x = load_images(data_dir, 'test_batch')
        y = load_labels(data_dir, 'test_batch')
    x = x / np.float32(255)
    if not flat:
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = np.reshape(x, [-1, 32, 32, 3])
    return x, y
