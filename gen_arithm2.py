#!/usr/bin/env python3
"""
Extra-format PTX arithmetic benchmarks (companion to gen_arithm.py).

Covers what gen_arithm.py doesn't:
- FP64 scalar (add/sub/mul/fma/min/max/neg/abs/div/rcp/sqrt)
- TF32 cvt (the only scalar PTX op for tf32 — actual tf32 mul lives in mma)
- Predicate / select: setp+selp, slct
- Packed video ops: vadd2, vadd4, vabsdiff4.add, vmin/vmax.s32

Generates src/arithm2_bench.cu with throughput + latency kernels per op.
Same CSV output and SASS verification as gen_arithm.py.
"""

from bench_common import (
    Insn, _db, _du, _dt, _ib, _ub,
    gen_and_build,
)


CATEGORIES = {
    "FP64 core":   ["f64_add", "f64_sub", "f64_mul", "f64_fma",
                    "f64_min", "f64_max"],
    "FP64 multi-insn": ["f64_div_rn", "f64_rcp_rn", "f64_sqrt_rn"],
    "TF32 cvt":    ["tf32_cvt_rna_add"],
    "Predicate":   ["setp_selp_lt_s32", "setp_selp_lt_f32", "slct_s32"],
    "Video SIMD":  ["vadd2_u32", "vsub2_u32", "vadd4_u32", "vsub4_u32",
                    "vabsdiff2_u32", "vabsdiff4_u32",
                    "vmin_s32", "vmax_s32"],
}


def make_instructions():
    I = []

    # ==================================================================
    # FP64
    # ==================================================================
    I += [
        _db("f64_add", "add.rn.f64", "add.rn.f64 %0, %0, %1;", "1.0", "0.000001"),
        _db("f64_sub", "sub.rn.f64", "sub.rn.f64 %0, %0, %1;", "1e6", "0.000001"),
        _db("f64_mul", "mul.rn.f64", "mul.rn.f64 %0, %0, %1;", "1.0", "0.999999"),
        _dt("f64_fma", "fma.rn.f64", "fma.rn.f64 %0, %0, %1, %0;", "1.0", "1.000001"),
        _db("f64_min", "min.f64", "min.f64 %0, %0, %1;", "1.0", "0.5"),
        _db("f64_max", "max.f64", "max.f64 %0, %0, %1;", "0.1", "0.5"),
        # NOTE: neg.f64 / abs.f64 omitted — on Ampere these are operand
        # modifiers folded into consumers (DFMA src negation), no isolated insn.
        # Multi-insn: emulated on most chips — typically lowered into many DFMAs
        _db("f64_div_rn", "div.rn.f64", "div.rn.f64 %0, %0, %1;", "1.0", "0.9999"),
        _du("f64_rcp_rn", "rcp.rn.f64", "rcp.rn.f64 %0, %0;", "2.0"),
        _du("f64_sqrt_rn", "sqrt.rn.f64", "sqrt.rn.f64 %0, %0;", "4.0"),
    ]

    # ==================================================================
    # TF32 cvt
    # tf32 is a b32-stored format. Round-trip through a temp .b32 reg so
    # the accumulator stays float. Measures cvt.f32->tf32 + mov.b32.
    # ==================================================================
    cat = "tf32"
    # Plain "cvt + mov" reaches a fixed point after 1 iter (value snaps to
    # nearest TF32 grid point) and ptxas folds the loop. Add a small
    # accumulator drift so each iter's input differs — this measures
    # cvt+add, but cvt is the dominant op (add is ~4 cycles, cvt is...).
    I += [
        Insn("tf32_cvt_rna_add", "cvt.rna.tf32.f32+add",
             "{ .reg .b32 t; cvt.rna.tf32.f32 t, %0; mov.b32 %0, t; "
             "add.rn.f32 %0, %0, %1; }",
             "float", "f", "1.0f", [("float", "k0", "0.000001f")], 80, cat),
    ]

    # ==================================================================
    # Predicate / select
    # All wrap setp+selp / slct in a PTX scope so the accumulator stays
    # the same scalar type (no inline-asm predicate constraint needed).
    # ==================================================================
    cat = "pred"
    I += [
        Insn("setp_selp_lt_s32", "setp.lt+selp.b32",
             "{ .reg .pred p; setp.lt.s32 p, %0, %1; "
             "selp.b32 %0, %0, %1, p; }",
             "int", "r", "1", [("int", "k0", "2")], 0, cat),
        Insn("setp_selp_lt_f32", "setp.lt+selp.b32",
             "{ .reg .pred p; setp.lt.f32 p, %0, %1; "
             "selp.b32 %0, %0, %1, p; }",
             "float", "f", "1.0f", [("float", "k0", "0.5f")], 0, cat),
        # slct: dst = (src3 >= 0) ? src1 : src2.
        Insn("slct_s32", "slct.s32.s32",
             "slct.s32.s32 %0, %0, %1, %0;",
             "int", "r", "1", [("int", "k0", "2")], 0, cat),
    ]

    # ==================================================================
    # Video SIMD (sm_61+; not deprecated on Ampere)
    # ==================================================================
    cat = "video"
    I += [
        _ub("vadd2_u32", "vadd2.u32",
            "vadd2.u32.u32.u32 %0, %0, %1, %0;", "1u", "1u", cat=cat),
        _ub("vsub2_u32", "vsub2.u32",
            "vsub2.u32.u32.u32 %0, %0, %1, %0;", "1000000u", "1u", cat=cat),
        _ub("vadd4_u32", "vadd4.u32",
            "vadd4.u32.u32.u32 %0, %0, %1, %0;", "1u", "1u", cat=cat),
        _ub("vsub4_u32", "vsub4.u32",
            "vsub4.u32.u32.u32 %0, %0, %1, %0;", "0x40404040u", "0x01010101u", cat=cat),
        _ub("vabsdiff2_u32", "vabsdiff2.u32",
            "vabsdiff2.u32.u32.u32 %0, %0, %1, %0;",
            "0x10101010u", "0x01010101u", cat=cat),
        _ub("vabsdiff4_u32", "vabsdiff4.u32",
            "vabsdiff4.u32.u32.u32 %0, %0, %1, %0;",
            "0x10101010u", "0x01010101u", cat=cat),
        Insn("vmin_s32", "vmin.s32",
             "vmin.s32.s32.s32 %0, %0, %1, %0;",
             "int", "r", "100", [("int", "k0", "50")], 0, cat),
        Insn("vmax_s32", "vmax.s32",
             "vmax.s32.s32.s32 %0, %0, %1, %0;",
             "int", "r", "1", [("int", "k0", "50")], 0, cat),
    ]

    return I


if __name__ == "__main__":
    instructions = make_instructions()
    gen_and_build("arithm2", instructions, CATEGORIES)
