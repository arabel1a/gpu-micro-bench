NVCC = nvcc
ARCH = -gencode arch=compute_75,code=sm_75 \
       -gencode arch=compute_80,code=sm_80 \
       -gencode arch=compute_86,code=sm_86
FLAGS = -O3 $(ARCH) -Xcompiler -fopenmp -lgomp

all: bin/memtest bin/mmvq_bench bin/arithmtest bin/arithmtest_gen

bin/memtest: src/memtest.cu | bin
	$(NVCC) $(FLAGS) -o $@ $<

bin/mmvq_bench: src/mmvq_bench.cu | bin
	$(NVCC) $(FLAGS) -o $@ $<

bin/arithmtest: src/arithmtest.cu | bin
	$(NVCC) $(FLAGS) -o $@ $<

bin/arithmtest_gen: src/arithmtest_gen.cu | bin
	$(NVCC) $(FLAGS) -o $@ $<

# Regenerate the comprehensive benchmark from instruction definitions
gen: gen_arithm.py
	python gen_arithm.py --gen-only

bin:
	mkdir -p bin

clean:
	rm -f bin/memtest bin/mmvq_bench bin/arithmtest bin/arithmtest_gen

.PHONY: all clean gen
