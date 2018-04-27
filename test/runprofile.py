import argparse
import tensorflow as tf
from tensorflow.python.client import timeline
import kernels
import pycuda
import pycuda.autoinit
import ctypes
import gc

nvtx = ctypes.CDLL("/usr/local/cuda-9.1/lib64/libnvToolsExt.so.1.0.0")

memory_initial = pycuda.driver.mem_get_info()[0]

parser = argparse.ArgumentParser()
parser.add_argument('dims', nargs='?', default=2048, type=int, help='dimension of differentiable variables')
parser.add_argument('iterations', nargs='?', default=1, type=int, help='iterations to run')

# turn on the XLA JIT compiler
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

if __name__ == '__main__':
    args = parser.parse_args()
    b = kernels.Benchmark(args.dims)
    with tf.Session(config=config) as sess:
        with tf.device("/device:GPU:0"):
            b.warmup(sess)
            memory = pycuda.driver.mem_get_info()[0]
            pycuda.driver.start_profiler()
            nvtx.nvtxMarkA(ctypes.c_char_p(b"Memory usage: %d" % (memory_initial-memory)))
            for i in range(args.iterations):
                gc.collect()
                nvtx.nvtxRangePushA(ctypes.c_char_p(b"kernel"))
                b.run(sess)
                nvtx.nvtxRangePop()
            pycuda.driver.stop_profiler()
            # # alternatively, use TF's built-in tracer
            # # (this breaks nvprof as it uses CUPTI too, hence commented out)
            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # b.run(sess, options=options, run_metadata=run_metadata)
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('hmlstm.json', 'w') as f:
            #     f.write(chrome_trace)
