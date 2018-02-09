import argparse
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument('dims', default=2048, type=int, help='dimension of differentiable variables')
parser.add_argument('device', default="/device:GPU:0", type=str, help='which device should be used to execute the benchmark')

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
        tf.multiply(i, g, name='c'),
        tf.where(
            tf.equal(zb, tf.constant(0., dtype=tf.float32)),
            tf.identity(c),
            tf.add(tf.multiply(f, c), tf.multiply(i, g))
        )
    )

def benchmark(dims, device_name):
    # set up control variables
    z = tf.Variable(tf.cast(tf.less(tf.random_uniform([dims]), 0.5), dtype=tf.float32), name='z')
    zb = tf.Variable(tf.cast(tf.less(tf.random_uniform([dims]), 0.5), dtype=tf.float32), name='zb')

    # set up differentiable variables
    c = tf.Variable(tf.random_uniform([dims, dims]), name='c')
    f = tf.Variable(tf.random_uniform([dims, dims]), name='f')
    i = tf.Variable(tf.random_uniform([dims, dims]), name='i')
    g = tf.Variable(tf.random_uniform([dims, dims]), name='g')

    # build the computation graph for our kernel
    new_c = hmlstm_update_c(z, zb, c, f, i, g)
    c_grad, f_grad, i_grad, g_grad = tf.gradients(new_c, [c, f, i, g])

    # execute the benchmark
    with tf.Session() as sess:
        with tf.device(device_name):
            sess.run(tf.global_variables_initializer())
            start_time = time.clock()
            sess.run([c_grad, f_grad, i_grad, g_grad])
            elapsed_time = time.clock() - start_time

    return elapsed_time

if __name__ == '__main__':
    args = parser.parse_args()
    print(benchmark(args.dims, args.device))
