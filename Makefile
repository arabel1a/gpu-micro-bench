NVCC = nvcc
ARCH = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86
FLAGS = -O3 $(ARCH) -Xcompiler -fopenmp -lgomp

all: memtest

memtest: memtest.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f memtest

.PHONY: all clean
