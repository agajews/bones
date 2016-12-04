from bones.tasks.cifar10 import load_cifar
from bones import *


x, y = load_cifar(flat=False)
x_in = placeholder_like(x, 'x')
y_in = placeholder_like(y, 'y', dtype='float32')
dropout_prob = scalar_placeholder(name='dropout')
model = sequential([
    conv([3, 3], 32, activ=relu),
    maxpool([2, 2], strides=[2, 2]),
    conv([3, 3], 64, activ=relu),
    conv([3, 3], 64, activ=relu),
    maxpool([2, 2], strides=[2, 2]),
    linear(512, activ=relu),
    dropout(dropout_prob),
    linear(10, activ=softmax)
])(x_in)
loss = xentropy(model, y_in)
optim = adam(lr=0.001)(loss)
test_x, test_y = load_cifar(flat=False, test=True)
train(model, loss, optim, x_in, y_in, x, y, test_x, test_y,
      train_dict={dropout_prob: 0.5}, test_dict={dropout_prob: 1.0},
      batch_size=96, epochs=50)
