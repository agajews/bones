from bones.tasks.mnist import load_mnist
from bones import *


class Mnist(DistributedBones):
    def build_model(self, gs):
        self.gs = gs
        self.x, self.y = load_mnist(flat=True)
        self.x_in = placeholder_like(self.x, 'x')
        self.y_in = placeholder_like(self.y, 'y', dtype='float32')
        self.dropout_prob = scalar_placeholder()
        self.model = sequential([
            # conv([5, 5], 32),
            # maxpool([2, 2], strides=[2, 2]),
            # conv([5, 5], 64),
            # maxpool([2, 2], strides=[2, 2]),
            # linear(1024),
            # dropout(self.dropout_prob),
            linear(100),
            linear(10, activ=softmax)
        ])(self.x_in)
        self.loss = xentropy(self.model, self.y_in)
        self.optim = adam(lr=0.0001, gs=gs)(self.loss)
        return initialize_vars(execute=False)

    def train(self, sess):
        model_fn = call_wrap(self.model, self.x_in, {self.dropout_prob: 1.0},
                             sess=sess)
        loss_fn = call_wrap(self.loss, [self.x_in, self.y_in],
                            {self.dropout_prob: 1.0}, sess=sess)
        train_fn = call_wrap([self.loss, self.optim, self.gs],
                             {'x': self.x_in, 'y': self.y_in,
                              self.dropout_prob: 0.5}, sess=sess)

        def update(step, epoch, batch_x, batch_y):
            curr_loss, _, gs = train_fn(x=batch_x, y=batch_y)
            if step % 500 == 0:
                print(('Loss at local step {:d}, global step {:d}, '
                       'local epoch {:d}: {:0.4f}')
                      .format(step, gs, epoch, curr_loss))

        for_batches(update, [self.x, self.y], 128, steps=2000)
        print('Final loss: {:0.4f}'.format(
            mean_batches(loss_fn, [self.x, self.y])))
        print('Train accuracy: {:0.2f}%'.format(
            batch_cat_acc(model_fn, self.x, self.y) * 100))

        test_x, test_y = load_mnist(flat=True, test=True)
        print('Test accuracy: {:0.2f}%'.format(
            batch_cat_acc(model_fn, test_x, test_y) * 100))


if __name__ == '__main__':
    mnist = Mnist()
    mnist.run()
