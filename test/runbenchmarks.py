import argparse
import tensorflow as tf
import kernels
import timeit

parser = argparse.ArgumentParser()
parser.add_argument('dims', nargs='?', default=2048, type=int, help='dimension of differentiable variables')
parser.add_argument('device', nargs='?', default="/device:GPU:0", type=str, help='which device should be used to execute the benchmark')

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
    b = kernels.Benchmark(args.dims)
    with tf.Session(config=config) as sess:
        with tf.device(args.device):
            b.warmup(sess)
            t = timeit.Timer("b.run(sess)", globals=globals())
            its, total_time = t.autorange()
            pretty_print_time(total_time / its)
