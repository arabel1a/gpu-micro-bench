#!/usr/bin/env python3
"""
Tensor-core (mma.sync) PTX benchmark — Ampere-supported shapes.

Coverage on sm_80/sm_86 (CMP 170HX / CMP 90HX targets):
- m16n8k16 f16.f16.f32      (HMMA)
- m16n8k16 bf16.bf16.f32    (HMMA bf16)
- m16n8k8  tf32.tf32.f32    (HMMA tf32)
- m16n8k32 s8.s8.s32        (IMMA s8)
- m16n8k32 u8.u8.s32        (IMMA u8)
- m8n8k4   f64.f64.f64.f64  (DMMA — the prime "is FP64 tensor core gimped?" test)

Each op gets two kernels:
- _lat: single D accumulator chain (D fed back as C), 8 mmas/iter — measures
  back-to-back issue latency.
- _tput: 4 independent D accumulators, 8 mmas/iter (round-robin), so ILP=4 —
  measures pipelined issue throughput.

All inputs (A, B fragments) are warp-uniform constants loaded from immediate
register inits; no shared-memory traffic in the timed loop.

Output written as one scalar per lane to volatile global to defeat dead-code.
"""

from bench_common import Insn, gen_and_build


CATEGORIES = {
    "HMMA f16": ["mma_m16n8k16_f16f16f32"],
    "HMMA bf16": ["mma_m16n8k16_bf16bf16f32"],
    "HMMA tf32": ["mma_m16n8k8_tf32tf32f32"],
    "IMMA":     ["mma_m16n8k32_s8s8s32", "mma_m16n8k32_u8u8s32"],
    "DMMA":     ["mma_m8n8k4_f64"],
}


LAUNCH = "__global__ void __launch_bounds__(128,1)"


# ----------------------------------------------------------------
# A single mma instruction descriptor
#
#   ptx          : full PTX mnemonic (e.g. "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
#   d_n / d_ct   : number of D regs and their C-type ("float", "int", "double")
#                  + asm constraint letter ("f", "r", "d")
#   a_n / a_ct   : number of A regs (always packed into "unsigned" / "r"
#                  except f64 which is "double" / "d")
#   b_n          : number of B regs
#   a_init / b_init : C expressions to initialize each A/B reg
#   d_ct_inline  : "float" / "int" / "double" — used to declare D vars
# ----------------------------------------------------------------

def _mma_kernel(name, ptx, d_n, d_ct, d_ctra, a_n, a_ct, a_ctra, a_init,
                b_n, b_ct, b_ctra, b_init, mode, min_sm=80):
    n_per_iter = 8
    # Operand placeholder layout for one mma instruction:
    #   D = %0 .. %{d_n-1}
    #   A = %{d_n} .. %{d_n+a_n-1}
    #   B = %{d_n+a_n} .. %{d_n+a_n+b_n-1}
    #   (C reuses D placeholders since C==D for the chain)

    def make_one_mma(d_off=0, a_off=None, b_off=None):
        """Build one mma asm string. d_off shifts D register indices.
        a_off/b_off shift A/B placeholder indices (default = right after all D)."""
        if a_off is None:
            a_off = d_n  # start of A placeholders
        if b_off is None:
            b_off = a_off + a_n  # ditto for B
        d_args = ",".join(f"%{d_off + i}" for i in range(d_n))
        a_args = ",".join(f"%{a_off + i}" for i in range(a_n))
        b_args = ",".join(f"%{b_off + i}" for i in range(b_n))
        # C = D (chain)
        c_args = d_args
        return f"{ptx} {{{d_args}}}, {{{a_args}}}, {{{b_args}}}, {{{c_args}}};"

    if mode == "lat":
        # Single D-accumulator (d_n regs), 8 mmas in chain.
        # PTX placeholders: D = %0..%{d_n-1}, A = next a_n, B = next b_n
        d_decls = ", ".join(f"d{i} = ({d_ct})0" for i in range(d_n))
        d_outs = ",".join(f'"+{d_ctra}"(d{i})' for i in range(d_n))
        a_decls = "\n    ".join(
            f"{a_ct} a{i} = {a_init};" for i in range(a_n))
        b_decls = "\n    ".join(
            f"{b_ct} b{i} = {b_init};" for i in range(b_n))
        a_ins = ",".join(f'"{a_ctra}"(a{i})' for i in range(a_n))
        b_ins = ",".join(f'"{b_ctra}"(b{i})' for i in range(b_n))
        ins = a_ins + ("," if a_ins and b_ins else "") + b_ins

        mma_lines = [make_one_mma() for _ in range(n_per_iter)]
        body = "\n        ".join(
            [f'"{l}\\n\\t"' for l in ["{"] + mma_lines + ["}"]])

        d_sum = " + ".join(f"d{i}" for i in range(d_n))
        return f"""
{LAUNCH}
kern_{name}_lat(volatile {d_ct}* __restrict__ out, int n) {{
#if __CUDA_ARCH__ >= {min_sm * 10}
    {d_ct} {d_decls};
    {a_decls}
    {b_decls}
    #pragma unroll 1
    for(int i=0;i<n;i++) asm volatile(
        {body}
        :{d_outs}
        :{ins});
    out[threadIdx.y*32+threadIdx.x] = ({d_ct})({d_sum});
#endif
}}
"""

    # tput: 4 independent D-accumulators, round-robin 8 mmas/iter.
    # Total D regs = 4*d_n. Placeholder layout per mma at chain k:
    #   D_k = %{k*d_n} .. %{(k+1)*d_n - 1}
    n_chains = 4
    n_per_iter_tput = 8  # 8 mmas total, 2 per chain
    a_off = n_chains * d_n
    b_off = a_off + a_n

    d_var_decls = ", ".join(
        f"d{c}_{i} = ({d_ct})0"
        for c in range(n_chains) for i in range(d_n))
    d_outs = ",".join(
        f'"+{d_ctra}"(d{c}_{i})'
        for c in range(n_chains) for i in range(d_n))
    a_decls = "\n    ".join(f"{a_ct} a{i} = {a_init};" for i in range(a_n))
    b_decls = "\n    ".join(f"{b_ct} b{i} = {b_init};" for i in range(b_n))
    a_ins = ",".join(f'"{a_ctra}"(a{i})' for i in range(a_n))
    b_ins = ",".join(f'"{b_ctra}"(b{i})' for i in range(b_n))
    ins = a_ins + ("," if a_ins and b_ins else "") + b_ins

    mma_lines = []
    for k in range(n_per_iter_tput):
        c = k % n_chains
        mma_lines.append(make_one_mma(d_off=c * d_n, a_off=a_off, b_off=b_off))
    body = "\n        ".join(
        [f'"{l}\\n\\t"' for l in ["{"] + mma_lines + ["}"]])

    # Sum across all chain accumulators
    d_sum = " + ".join(
        f"d{c}_{i}" for c in range(n_chains) for i in range(d_n))
    return f"""
{LAUNCH}
kern_{name}_tput(volatile {d_ct}* __restrict__ out, int n) {{
#if __CUDA_ARCH__ >= {min_sm * 10}
    {d_ct} {d_var_decls};
    {a_decls}
    {b_decls}
    #pragma unroll 1
    for(int i=0;i<n;i++) asm volatile(
        {body}
        :{d_outs}
        :{ins});
    out[threadIdx.y*32+threadIdx.x] = ({d_ct})({d_sum});
#endif
}}
"""


# ----------------------------------------------------------------
# Per-shape dispatch
# ----------------------------------------------------------------

def _hmma_f16():
    """m16n8k16 f16.f16.f32 — D=4xf32, A=4xb32(f16x2), B=2xb32(f16x2)."""
    return dict(
        ptx="mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        d_n=4, d_ct="float", d_ctra="f",
        a_n=4, a_ct="unsigned", a_ctra="r", a_init="0x3C003C00u",  # 1.0,1.0
        b_n=2, b_ct="unsigned", b_ctra="r", b_init="0x3C003C00u",
    )


def _hmma_bf16():
    """m16n8k16 bf16.bf16.f32 — D=4xf32, A=4xb32(bf16x2), B=2xb32(bf16x2)."""
    return dict(
        ptx="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        d_n=4, d_ct="float", d_ctra="f",
        a_n=4, a_ct="unsigned", a_ctra="r", a_init="0x3F803F80u",  # 1.0,1.0 bf16
        b_n=2, b_ct="unsigned", b_ctra="r", b_init="0x3F803F80u",
    )


def _hmma_tf32():
    """m16n8k8 tf32.tf32.f32 — D=4xf32, A=4xb32(tf32), B=2xb32(tf32).
    tf32 stored as a b32 register holding the upper 19 bits of an f32."""
    return dict(
        ptx="mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        d_n=4, d_ct="float", d_ctra="f",
        # tf32(1.0) = upper 19 bits of f32(1.0) = 0x3F800000 (low bits zero)
        a_n=4, a_ct="unsigned", a_ctra="r", a_init="0x3F800000u",
        b_n=2, b_ct="unsigned", b_ctra="r", b_init="0x3F800000u",
    )


def _imma_s8():
    """m16n8k32 s8.s8.s32 — D=4xs32, A=4xb32(4xs8 packed), B=2xb32(4xs8 packed)."""
    return dict(
        ptx="mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        d_n=4, d_ct="int", d_ctra="r",
        a_n=4, a_ct="unsigned", a_ctra="r", a_init="0x01010101u",
        b_n=2, b_ct="unsigned", b_ctra="r", b_init="0x01010101u",
    )


def _imma_u8():
    return dict(
        ptx="mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32",
        d_n=4, d_ct="int", d_ctra="r",
        a_n=4, a_ct="unsigned", a_ctra="r", a_init="0x01010101u",
        b_n=2, b_ct="unsigned", b_ctra="r", b_init="0x01010101u",
    )


def _dmma_f64():
    """m8n8k4 f64.f64.f64.f64 — D=2xf64, A=1xf64, B=1xf64."""
    return dict(
        ptx="mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64",
        d_n=2, d_ct="double", d_ctra="d",
        a_n=1, a_ct="double", a_ctra="d", a_init="1.0",
        b_n=1, b_ct="double", b_ctra="d", b_init="1.0",
    )


def make_instructions():
    I = []
    specs = [
        ("mma_m16n8k16_f16f16f32",   "HMMA f16",  _hmma_f16),
        ("mma_m16n8k16_bf16bf16f32", "HMMA bf16", _hmma_bf16),
        ("mma_m16n8k8_tf32tf32f32",  "HMMA tf32", _hmma_tf32),
        ("mma_m16n8k32_s8s8s32",     "IMMA",      _imma_s8),
        ("mma_m16n8k32_u8u8s32",     "IMMA",      _imma_u8),
        ("mma_m8n8k4_f64",           "DMMA",      _dmma_f64),
    ]
    for name, cat, factory in specs:
        spec = factory()
        I.append(Insn(
            name, spec["ptx"], "",
            spec["d_ct"], spec["d_ctra"], "0", [], 80, cat,
            custom_lat=_mma_kernel(name, **spec, mode="lat"),
            custom_tput=_mma_kernel(name, **spec, mode="tput"),
        ))
    return I


if __name__ == "__main__":
    instructions = make_instructions()
    gen_and_build("tensor", instructions, CATEGORIES)
