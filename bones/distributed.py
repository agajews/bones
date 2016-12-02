import tensorflow as tf


class DistributedBones(object):
    def __init__(self, model_fn=None, train_fn=None):
        self.model_fn = model_fn
        self.train_fn = train_fn

    def build_model(self, global_step):
        self.model_fn(global_step)

    def train(self, sess):
        self.train_fn(sess)

    def run(self):
        tf.app.flags.DEFINE_string("node_type", "", "Either 'ps' or 'worker'")
        tf.app.flags.DEFINE_string("ps_ip", "", "parameter server ip")
        tf.app.flags.DEFINE_string("worker_ips", "", "json array of ips")
        tf.app.flags.DEFINE_integer("idx", 0, "Index of task within the job")
        FLAGS = tf.app.flags.FLAGS

        parameter_servers = [FLAGS.ps_ip + ':2222']
        worker_ips = FLAGS.worker_ips.split(',')
        workers = [ip + ':2222' for ip in worker_ips]
        cluster = tf.train.ClusterSpec({"ps": parameter_servers,
                                        "worker": workers})

        # start a server for a specific task
        server = tf.train.Server(cluster,
                                 job_name=FLAGS.node_type,
                                 task_index=FLAGS.idx)
        if FLAGS.node_type == "ps":
            # print("I'm a parameter server")
            server.join()
        elif FLAGS.node_type == "worker":
            # print("I'm a worker")
            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.idx,
                    cluster=cluster)):
                global_step = tf.get_variable(
                    'global_step', [], initializer=tf.constant_initializer(0),
                    trainable=False, dtype='int32')
                init_op = self.build_model(global_step)
            sv = tf.train.Supervisor(is_chief=(FLAGS.idx == 0),
                                     global_step=global_step,
                                     init_op=init_op)
            with sv.prepare_or_wait_for_session(server.target) as sess:
                self.train(sess)
