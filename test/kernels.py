import tensorflow as tf

# The below code is a snippet adapted from the
# canonical HMLSTM implementation, specifically
# the code here:
#
# https://github.com/n-s-f/hierarchical-rnn/blob/f23a71cf1eff0bb8d0cbf9810f05c2eae8e10723/hmlstm/hmlstm_cell.py#L78-L83

def hmlstm_update_c(z, zb, c, f, i, g):
    i = tf.sigmoid(i)
    g = tf.tanh(g)
    f = tf.sigmoid(f)
    return tf.where(
        tf.equal(z, tf.constant(1., dtype=tf.float32)),
        tf.multiply(i, g),
        tf.where(
            tf.equal(zb, tf.constant(0., dtype=tf.float32)),
            tf.identity(c),
            tf.add(tf.multiply(f, c), tf.multiply(i, g))
        )
    )

class Benchmark:
    def __init__(self, dims):
        # set up control variables
        z = tf.Variable(tf.cast(tf.less(tf.random_uniform([dims]), 0.5), dtype=tf.float32))
        zb = tf.Variable(tf.cast(tf.less(tf.random_uniform([dims]), 0.5), dtype=tf.float32))

        # set up differentiable variables
        c = tf.Variable(tf.random_uniform([dims, dims]))
        f = tf.Variable(tf.random_uniform([dims, dims]))
        i = tf.Variable(tf.random_uniform([dims, dims]))
        g = tf.Variable(tf.random_uniform([dims, dims]))

        # build the computation graph for our kernel
        new_c = hmlstm_update_c(z, zb, c, f, i, g)
        self.c_grad, self.f_grad, self.i_grad, self.g_grad = tf.gradients(new_c, [c, f, i, g])

    def warmup(self, sess):
        sess.run(tf.global_variables_initializer())
        self.run(sess)

    def run(self, sess, **kwargs):
        sess.run([self.c_grad, self.f_grad, self.i_grad, self.g_grad], **kwargs)
