.PHONY: all
all: cuda.so

NVCC=nvcc
%: %.cu
	nvcc $(NVCCFLAGS) -o $@ $<
%.so: %.cu
	nvcc $(NVCCFLAGS) -shared -Xcompiler -fPIC -o $@ $<
