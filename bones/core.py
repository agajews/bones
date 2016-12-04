import tensorflow as tf

sess = tf.InteractiveSession()

scope = tf.variable_scope


uconcat = tf.concat


def concat(dim):
    def fn(*xs):
        return tf.concat(dim, xs)
    return fn


def scope_variables(scope=None):
    if scope is None:
        return tf.global_variables()
    else:
        return tf.contrib.framework.get_variables(scope=scope)


def initialize_vars(var_scope=None, sess=None, execute=True):
    variables = scope_variables(var_scope)
    init_op = tf.variables_initializer(variables)
    if sess is None:
        sess = tf.get_default_session()
    if execute:
        sess.run(init_op)
    return init_op


def sequential(layers, name='sequential'):
    def fn(x):
        with scope(name):
            for l in layers:
                x = l(x)
            return x
    return fn


def normal(mean=0, std=0.01):
    return lambda shape: tf.truncated_normal(shape, mean, std)


zeros = tf.zeros


tanh = tf.tanh
sigmoid = tf.sigmoid
softmax = tf.nn.softmax
relu = tf.nn.relu
identity = lambda x: x

flatten = tf.contrib.layers.flatten


def linear(nodes, W_init=normal(), b_init=zeros, activ=relu, name='fc'):
    def fn(x):
        shape = x.get_shape().as_list()
        if len(shape) > 2:
            x = flatten(x)
            shape = x.get_shape().as_list()
        with scope(name):
            W = tf.Variable(W_init([shape[1], nodes]), name='W')
            b = tf.Variable(b_init([nodes]), name='b')
            return activ(tf.matmul(x, W) + b)
    return fn


def conv2d(filter_size, n_filters, strides=[1, 1], pad='SAME',
           W_init=normal(), b_init=zeros, activ=relu, name='conv'):
    def fn(x):
        shape = x.get_shape().as_list()
        n_channels = shape[3]
        with scope(name):
            W = tf.Variable(W_init(filter_size + [n_channels, n_filters]),
                            name='W')
            b = tf.Variable(b_init([n_filters]), name='b')
            return activ(tf.nn.conv2d(x, W, strides=[1] + strides + [1],
                                      padding=pad) + b)
    return fn


def maxpool2d(pool_size, strides=[1, 1], pad='SAME'):
    def fn(x):
        return tf.nn.max_pool(x, ksize=[1] + pool_size + [1],
                              strides=[1] + strides + [1], padding=pad)
    return fn


def avgpool2d(pool_size, strides=[1, 1], pad='SAME'):
    def fn(x):
        return tf.nn.avg_pool(x, ksize=[1] + pool_size + [1],
                              strides=[1] + strides + [1], padding=pad)
    return fn


conv = conv2d
maxpool = maxpool2d
avgpool = avgpool2d


placeholder = tf.placeholder


def dropout(keep_prob):
    return lambda x: tf.nn.dropout(x, keep_prob)


log = tf.log


def rmean(axis=None):
    return lambda x: tf.reduce_mean(x, axis=axis)


umean = tf.reduce_mean


def rsum(axis=None):
    return lambda x: tf.reduce_sum(x, axis=axis)


usum = tf.reduce_sum


def clip(lo=0, hi=1.0):
    return lambda x: tf.clip_by_value(x, lo, hi)


def clip_norm(norm, axes=None):
    return lambda x: tf.clip_by_norm(x, norm, axes)


def clip_avg(norm):
    return lambda x: tf.clip_by_average_norm(x, norm)


def clip_global(norm):
    def fn(xs, use_norm=None):
        return tf.clip_by_global_norm(xs, norm, use_norm)
    return fn


def crossentropy(log_clip=1e-10):
    def fn(y_hat, y):
        log_y_hat = log(clip(log_clip, 1.0)(y_hat))
        return umean(-usum(y * log_y_hat, axis=1))
    return fn


xentropy = crossentropy()


def mse(y_hat, y):
    return umean((y_hat - y) ** 2)


def grads(x, selector=None):
    if selector is None:
        variables = tf.global_variables()
    elif callable(selector):
        variables = [v for v in tf.global_variables(x) if selector(v)]
    else:  # scope
        variables = tf.contrib.framework.get_variables(scope=selector)
    grads = tf.gradients(x, variables)
    return zip(grads, variables)


def sgd(lr=0.1, gs=None):
    def fn(x, var_scope=None):
        variables = scope_variables(var_scope)
        optim = tf.train.GradientDescentOptimizer(learning_rate=lr)
        return optim.minimize(x, var_list=variables, global_step=gs)
    return fn


def adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, gs=None,
         name='adam'):
    def fn(x, var_scope=None):
        variables = scope_variables(var_scope)
        # for variable in variables:
        #     print(variable.name, variable.get_shape())
        with scope(name):
            optim = tf.train.AdamOptimizer(learning_rate=lr,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
            return optim.minimize(x, var_list=variables, global_step=gs)
    return fn


def swapaxes(axis1, axis2):
    def fn(x):
        x_shape = x.get_shape().as_list()
        perm = list(range(len(x_shape)))
        perm[axis1] = axis2
        perm[axis2] = axis1
        return tf.transpose(x, perm)
    return fn


def uswapaxes(x, axis1, axis2):
    return swapaxes(axis1, axis2)(x)


def map_accuml(outer_fn, init_fn, swap=True):
    def fn(x):
        hidden_fn, output_fn = outer_fn(x)
        if swap:
            x = uswapaxes(x, 0, 1)
        res = tf.scan(hidden_fn, x, initializer=init_fn(x))
        res = tf.map_fn(output_fn, res)
        if swap:
            res = uswapaxes(res, 0, 1)
        return res
    return fn


def foldl(outer_fn, init_fn, swap=True):
    def fn(x):
        hidden_fn, output_fn = outer_fn(x)
        if swap:
            x = uswapaxes(x, 0, 1)
        res = tf.scan(hidden_fn, x, initializer=init_fn(x))
        return output_fn(res[-1])
    return fn


def basic_rnn_cell(nodes, activ_h=tanh, activ_o=tanh,
                   W_h_init=normal(), b_h_init=zeros,
                   W_i_init=normal(), W_o_init=normal(), b_o_init=zeros,
                   name='basic_rnn'):
    def outer(x):
        shape = x.get_shape().as_list()
        with scope(name):
            W_h = tf.Variable(W_h_init([nodes, nodes]), name='W_h')
            b_h = tf.Variable(b_h_init([nodes]), name='b_h')
            W_i = tf.Variable(W_i_init([shape[2], nodes]), name='W_i')
            W_o = tf.Variable(W_o_init([nodes, nodes]))
            b_o = tf.Variable(b_o_init([nodes]), name='b_o')

        def hidden_fn(hidden, x):
            return activ_h(tf.matmul(hidden, W_h) + tf.matmul(x, W_i) + b_h)

        def output_fn(hidden):
            return activ_o(tf.matmul(hidden, W_o) + b_o)
        return hidden_fn, output_fn
    return outer


def basic_rnn(nodes, activ_h=tanh, activ_o=tanh,
              W_h_init=normal(), b_h_init=zeros,
              W_i_init=normal(), W_o_init=normal(), b_o_init=zeros,
              name='basic_rnn', swap=True, fold=False):
    def init_fn(x):
        shape = x.get_shape().as_list()
        return tf.matmul(x[0, :, :], tf.zeros([shape[2], nodes]))
    cell = basic_rnn_cell(nodes, activ_h, activ_o,
                          W_h_init, b_h_init,
                          W_i_init, W_o_init, b_o_init,
                          name)
    if not fold:
        return map_accuml(cell, init_fn, swap)
    else:
        return foldl(cell, init_fn, swap)


def lstm_cell(nodes, activ_i=sigmoid, activ_c=tanh, activ_o=relu,
              W_i_init=normal(), b_i_init=zeros, U_i_init=normal(),
              W_f_init=normal(), b_f_init=zeros, U_f_init=normal(),
              W_g_init=normal(), b_g_init=zeros, U_g_init=normal(),
              W_c_init=normal(), b_c_init=zeros, U_c_init=normal(),
              W_o_init=normal(), b_o_init=zeros,
              name='lstm'):
    def outer(x):
        shape = x.get_shape().as_list()
        with scope(name):
            W_i = tf.Variable(W_i_init([shape[2], nodes]), name='W_i')
            b_i = tf.Variable(b_i_init([nodes]), name='b_i')
            U_i = tf.Variable(U_i_init([nodes, nodes]), name='U_i')

            W_f = tf.Variable(W_f_init([shape[2], nodes]), name='W_f')
            b_f = tf.Variable(b_f_init([nodes]), name='b_f')
            U_f = tf.Variable(U_f_init([nodes, nodes]), name='U_f')

            W_g = tf.Variable(W_g_init([shape[2], nodes]), name='W_g')
            b_g = tf.Variable(b_g_init([nodes]), name='b_g')
            U_g = tf.Variable(U_g_init([nodes, nodes]), name='U_g')

            W_c = tf.Variable(W_c_init([shape[2], nodes]), name='W_c')
            b_c = tf.Variable(b_c_init([nodes]), name='b_c')
            U_c = tf.Variable(U_c_init([nodes, nodes]), name='U_c')

            W_o = tf.Variable(W_o_init([nodes, nodes]), 'W_o')
            b_o = tf.Variable(b_o_init([nodes]), name='b_o')

        def hidden_fn(hidden, x):
            h_prev, c_prev = tf.unpack(hidden)
            i = activ_i(tf.matmul(x, W_i) + tf.matmul(h_prev, U_i) + b_i)
            f = activ_i(tf.matmul(x, W_f) + tf.matmul(h_prev, U_f) + b_f)
            g = activ_i(tf.matmul(x, W_g) + tf.matmul(h_prev, U_g) + b_g)
            c_s = activ_c(tf.matmul(x, W_c) + tf.matmul(h_prev, U_c) + b_c)
            c = f * c_prev + i * c_s
            h = g * activ_c(c)
            return tf.pack([h, c])

        def output_fn(hidden):
            return activ_o(tf.matmul(hidden[0, :, :], W_o) + b_o)
        return hidden_fn, output_fn
    return outer


def lstm(nodes, activ_i=sigmoid, activ_c=tanh, activ_o=relu,
         W_i_init=normal(), b_i_init=zeros, U_i_init=normal(),
         W_f_init=normal(), b_f_init=zeros, U_f_init=normal(),
         W_g_init=normal(), b_g_init=zeros, U_g_init=normal(),
         W_c_init=normal(), b_c_init=zeros, U_c_init=normal(),
         W_o_init=normal(), b_o_init=zeros,
         name='lstm', swap=True, fold=False):
    def init_fn(x):
        shape = x.get_shape().as_list()
        init = tf.matmul(x[0, :, :], tf.zeros([shape[2], nodes]))
        return tf.pack([init, init])
    cell = lstm_cell(nodes, activ_i, activ_c, activ_o,
                     W_i_init, b_i_init, U_i_init,
                     W_f_init, b_f_init, U_f_init,
                     W_g_init, b_g_init, U_g_init,
                     W_c_init, b_c_init, U_c_init,
                     W_o_init, b_o_init,
                     name='lstm')
    if not fold:
        return map_accuml(cell, init_fn, swap)
    else:
        return foldl(cell, init_fn, swap)
