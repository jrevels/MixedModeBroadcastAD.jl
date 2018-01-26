.PHONY: all
all: kernels.so

NVCC=nvcc
%.so: %.cu
	nvcc $(NVCCFLAGS) -shared -Xcompiler -fPIC -o $@ $<
