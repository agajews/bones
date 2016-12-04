from bones.tasks.mnist import load_mnist
from bones import *


x, y = load_mnist(flat=False)
x_test, y_test = load_mnist(flat=False, test=True)
x_in = placeholder_like(x, 'x')
y_in = placeholder_like(y, 'y', dtype='float32')
dropout_prob = scalar_placeholder()
model = sequential([
    conv([5, 5], 192, activ=relu),
    conv([1, 1], 160, activ=relu),
    conv([1, 1], 96, activ=relu),
    maxpool([3, 3], strides=[2, 2]),
    dropout(dropout_prob),
    conv([5, 5], 192, activ=relu),
    conv([1, 1], 192, activ=relu),
    conv([1, 1], 96, activ=relu),
    avgpool([3, 3], strides=[2, 2]),
    dropout(dropout_prob),
    conv([3, 3], 192, activ=relu),
    conv([1, 1], 192, activ=relu),
    conv([1, 1], 10, activ=relu),
    avgpool([8, 8]),
    linear(10, activ=softmax),
])(x_in)
loss = xentropy(model, y_in)
optim = adam(lr=0.001)(loss)
train(model, loss, optim, x_in, y_in, x, y, x_test, y_test,
      train_dict={dropout_prob: 0.5}, test_dict={dropout_prob: 1.0},
      batch_size=96, epochs=50)
