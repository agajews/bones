from bones.tasks.mnist import load_mnist
from bones import *


x, y = load_mnist(flat=False)
x_in = placeholder_like(x, 'x')
y_in = placeholder_like(y, 'y', dtype='float32')
model = sequential([
    conv([5, 5], 32),
    maxpool([2, 2], strides=[2, 2]),
    conv([5, 5], 64),
    maxpool([2, 2], strides=[2, 2]),
    linear(1024),
    dropout(0.5),
    linear(10, activ=softmax)
])(x_in)
loss = xentropy(model, y_in)
optim = adam(lr=0.0001)(model)
model_fn = call_wrap(model, x_in)
loss_fn = call_wrap(loss, [x_in, y_in])
train_fn = call_wrap([loss, optim], {'x': x_in, 'y': y_in})
intialize_vars()


def update(step, epoch, batch_x, batch_y):
    curr_loss, _ = train_fn(x=batch_x, y=batch_y)
    if step % 500 == 0:
        print('Loss at step {:d}, epoch {:d}: {:0.4f}'
              .format(step, epoch, curr_loss))
for_batches(update, [x, y], 128, steps=20000)
print('Final loss: {:0.4f}'.format(mean_batches(loss_fn, [x, y])))
print('Train accuracy: {:0.2f}%'.format(batch_cat_acc(model_fn, x, y) * 100))
test_x, test_y = load_mnist(flat=False, test=True)
print('Test accuracy: {:0.2f}%'.format(
    batch_cat_acc(model_fn, test_x, test_y) * 100))
