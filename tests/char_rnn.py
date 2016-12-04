from bones.tasks import char_gen
from bones import *

x, y, dicts = char_gen.load_text('data/lear.txt', 20)
num_chars = y.shape[1]
x_in = placeholder_like(x, 'x')
y_in = placeholder_like(y, 'y', dtype='float32')
dropout_prob = scalar_placeholder(name='dropout')
model = sequential([
    lstm(512),
    dropout(dropout_prob),
    lstm(512, fold=True),
    linear(1024),
    dropout(dropout_prob),
    linear(num_chars, activ=softmax)
])(x_in)
loss = xentropy(model, y_in)
optim = adam(lr=0.005)(loss)
train(model, loss, optim, x_in, y_in, x, y,
      train_dict={dropout_prob: 0.5}, test_dict={dropout_prob: 1.0},
      batch_size=128, epochs=100, print_fn=char_gen.print_fn(dicts))
