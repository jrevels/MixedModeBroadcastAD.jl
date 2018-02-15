import argparse
import tensorflow as tf
from tensorflow.python.client import timeline
import timeit
import pycuda
import pycuda.autoinit

parser = argparse.ArgumentParser()
parser.add_argument('dims', nargs='?', default=2048, type=int, help='dimension of differentiable variables')
parser.add_argument('device', nargs='?', default="/device:GPU:0", type=str, help='which device should be used to execute the benchmark')

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

class Benchmark:
    def __init__(self, dims):
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
        self.c_grad, self.f_grad, self.i_grad, self.g_grad = tf.gradients(new_c, [c, f, i, g])

    def warmup(self, sess):
        sess.run(tf.global_variables_initializer())
        self.run(sess)

    def run(self, sess, **kwargs):
        sess.run([self.c_grad, self.f_grad, self.i_grad, self.g_grad], **kwargs)

def pretty_print_time(s):
    if s < 1e-6:
        unit = "n"
        factor = 1e9
    elif s < 1e-3:
        unit = "u"
        factor = 1e6
    elif s < 1:
        unit = "m"
        factor = 1e3
    else:
        unit = ""
        factor = 1
    print("{0:.2f} {1}s".format(s*factor, unit))

# turn on the XLA JIT compiler
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

if __name__ == '__main__':
    args = parser.parse_args()
    b = Benchmark(args.dims)
    with tf.Session(config=config) as sess:
        with tf.device(args.device):
            b.warmup(sess)

            t = timeit.Timer("b.run(sess)", globals=globals())
            its, total_time = t.autorange()
            pretty_print_time(total_time / its)

            pycuda.driver.start_profiler()
            b.run(sess)
            pycuda.driver.stop_profiler()

            # alternatively, use TF's built-in tracer
            # (this breaks nvprof as it uses CUPTI too, hence commented out)

            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # b.run(sess, options=options, run_metadata=run_metadata)

            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('hmlstm.json', 'w') as f:
            #     f.write(chrome_trace)
