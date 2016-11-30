import tensorflow as tf

sess = tf.InteractiveSession()

scope = tf.variable_scope


def scope_variables(scope):
    if scope is None:
        return tf.global_variables()
    else:
        return tf.contrib.framework.get_variables(scope=scope)


def intialize_vars(scope=None, sess=None):
    variables = scope_variables(scope)
    if sess is None:
        sess = tf.get_default_session()
    sess.run(tf.variables_initializer(variables))


def sequential(layers):
    def fn(x):
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


def linear(nodes, W_init=normal(), b_init=zeros, activ=relu):
    def fn(x):
        shape = x.get_shape().as_list()
        if len(shape) > 2:
            x = tf.contrib.layers.flatten(x)
            shape = x.get_shape().as_list()
        W = tf.Variable(W_init([shape[1], nodes]), name='W')
        b = tf.Variable(b_init([nodes]), name='b')
        return activ(tf.matmul(x, W) + b)
    return fn


flatten = tf.contrib.layers.flatten


def conv2d(filter_size, n_filters, strides=[1, 1], pad='SAME',
           W_init=normal(), b_init=zeros, activ=relu):
    def fn(x):
        shape = x.get_shape().as_list()
        n_channels = shape[3]
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


conv = conv2d
maxpool = maxpool2d


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


def grads(x, selector=None):
    if selector is None:
        variables = tf.global_variables()
    elif callable(selector):
        variables = [v for v in tf.global_variables(x) if selector(v)]
    else:  # scope
        variables = tf.contrib.framework.get_variables(scope=selector)
    print('variables: ', variables)
    grads = tf.gradients(x, variables)
    return zip(grads, variables)


def sgd(lr=0.1):
    def fn(grads_and_vars):
        optim = tf.train.GradientDescentOptimizer(learning_rate=lr)
        return optim.apply_gradients(grads_and_vars)
    return fn


def adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
    def fn(x, var_scope=None):
        variables = scope_variables(var_scope)
        optim = tf.train.AdamOptimizer(learning_rate=lr,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
        return optim.minimize(x, var_list=variables)
    return fn
