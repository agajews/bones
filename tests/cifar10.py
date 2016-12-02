from bones.tasks.cifar10 import load_cifar
from bones import *


x, y = load_cifar(flat=False)
x_in = placeholder_like(x, 'x')
y_in = placeholder_like(y, 'y', dtype='float32')
dropout_prob = scalar_placeholder()
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
model_fn = call_wrap(model, x_in, {dropout_prob: 1.0})
loss_fn = call_wrap(loss, [x_in, y_in], {dropout_prob: 1.0})
train_fn = call_wrap([loss, optim], {'x': x_in, 'y': y_in, dropout_prob: 0.5})
initialize_vars()


def update(step, epoch, batch_x, batch_y):
    curr_loss, _ = train_fn(x=batch_x, y=batch_y)
    if step % 500 == 0:
        print('Loss at step {:d}, epoch {:d}: {:0.4f}'
              .format(step, epoch, curr_loss))
for_epochs(update, [x, y], 96, epochs=50)
print('Final loss: {:0.4f}'.format(mean_batches(loss_fn, [x, y])))
print('Train accuracy: {:0.2f}%'.format(batch_cat_acc(model_fn, x, y) * 100))
test_x, test_y = load_cifar(flat=False, test=True)
print('Test accuracy: {:0.2f}%'.format(
    batch_cat_acc(model_fn, test_x, test_y) * 100))
