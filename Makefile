NVCC = nvcc
ARCHS = 75 80 86
GENCODE = $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
FLAGS = -O3 $(GENCODE) -Xcompiler -fopenmp -lgomp

SRC = memtest mmvq_bench arithmtest arithmtest_gen

all: $(addprefix bin/,$(SRC)) bin/mmvq_bench_nodp4a bin/mmvq_bench_dp2a ptx

# ---- binaries ----
bin/%: src/%.cu | bin
	$(NVCC) $(FLAGS) -o $@ $<

bin/mmvq_bench_nodp4a: src/mmvq_bench.cu | bin
	$(NVCC) $(FLAGS) -DDISABLE_DP4A -o $@ $<

bin/mmvq_bench_dp2a: src/mmvq_bench.cu | bin
	$(NVCC) $(FLAGS) -DDP4A_REPL_DP2A -o $@ $<

# ---- PTX targets ----
# Generate explicit rules for each (source, arch) pair
define ptx_rule
ptx_dump/$(1)_sm$(2).ptx: src/$(1).cu | ptx_dump
	$(NVCC) -O3 -gencode arch=compute_$(2),code=compute_$(2) -ptx -o $$@ $$<
endef

# Create all PTX targets and their rules
PTX_TARGETS :=
$(foreach s,$(SRC),\
  $(foreach a,$(ARCHS),\
    $(eval $(call ptx_rule,$(s),$(a)))\
    $(eval PTX_TARGETS += ptx_dump/$(s)_sm$(a).ptx)\
  )\
)

ptx: $(PTX_TARGETS)

# directories
bin:
	mkdir -p bin

ptx_dump:
	mkdir -p ptx_dump

# misc
gen: gen_arithm.py
	python gen_arithm.py --gen-only

clean:
	rm -f bin/* ptx_dump/*
	
