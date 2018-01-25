.PHONY: all
all: cuda.so

NVCC=nvcc
%.so: %.cu
	nvcc $(NVCCFLAGS) -shared -Xcompiler -fPIC -o $@ $<
