#!/usr/bin/env python3
"""
Generate comprehensive PTX arithmetic instruction throughput/latency benchmark.

Generates src/arithm_bench.cu with ~100 instruction benchmarks.
Each instruction gets a throughput kernel (8 independent chains) and
a latency kernel (1 dependent chain, 8 ops per iteration).

Usage:
    python gen_arithm.py                   # generate .cu + compile + verify SASS
    python gen_arithm.py --gen-only        # just write .cu file
    python gen_arithm.py --table-only      # regenerate results.md from JSONs
"""

from bench_common import (
    Insn, _fb, _fu, _ft, _ib, _ub, _iu, _uu, _hb, _hu,
    gen_and_build, standard_argparser,
)


CATEGORIES = {
    "FP32 core": ["f32_add", "f32_sub", "f32_mul", "f32_fma", "f32_neg", "f32_abs", "f32_min", "f32_max"],
    "FP32 SFU": ["f32_rcp_approx", "f32_sqrt_approx", "f32_rsqrt", "f32_sin", "f32_cos", "f32_lg2", "f32_ex2", "f32_tanh"],
    "FP32 multi-insn": ["f32_div_approx", "f32_div_rn", "f32_rcp_rn", "f32_sqrt_rn"],
    "FP16x2": ["f16x2_add", "f16x2_sub", "f16x2_mul", "f16x2_fma",
               "f16x2_neg", "f16x2_abs", "f16x2_min", "f16x2_max",
               "f16x2_ex2", "f16x2_tanh"],
    "FP16": ["f16_add", "f16_sub", "f16_mul", "f16_fma",
             "f16_neg", "f16_abs", "f16_min", "f16_max",
             "f16_ex2", "f16_tanh"],
    "BF16x2": ["bf16x2_add", "bf16x2_sub", "bf16x2_mul", "bf16x2_fma",
               "bf16x2_neg", "bf16x2_abs", "bf16x2_min", "bf16x2_max"],
    "BF16": ["bf16_add", "bf16_sub", "bf16_mul", "bf16_fma",
             "bf16_neg", "bf16_abs"],
    "INT32 arith": ["s32_add", "s32_sub", "u32_add", "u32_sub",
                    "s32_mul_lo", "u32_mul_lo", "s32_mul_hi", "u32_mul_hi",
                    "s32_mad_lo", "u32_mad_lo", "s32_mad_hi",
                    "s32_mul24_lo", "u32_mul24_lo", "s32_mad24_lo", "u32_mad24_lo",
                    "s32_abs", "s32_neg", "s32_min", "u32_min", "s32_max", "u32_max",
                    "s32_sad"],
    "INT32 div/rem": ["s32_div", "u32_div", "s32_rem", "u32_rem"],
    "Bit ops": ["b32_and", "b32_or", "b32_xor", "b32_not", "b32_cnot",
                "b32_shl", "b32_shr", "s32_shr", "b32_lop3"],
    "Bit manip": ["b32_popc", "b32_clz", "b32_brev",
                  "s32_bfe", "u32_bfe", "b32_bfi", "s32_bfind", "b32_prmt"],
    "Dot product": ["dp4a_ss", "dp4a_uu", "dp4a_su", "dp4a_us",
                    "dp2a_lo_ss", "dp2a_hi_ss"],
    "Carry ops": ["s32_add_cc", "s32_sub_cc", "s32_addcc_addc"],
}


def make_instructions():
    """Define all arithmetic instructions to benchmark."""
    I = []

    # ==================================================================
    # FP32
    # ==================================================================
    I += [
        _fb("f32_add", "add.rn.f32", "add.rn.f32 %0, %0, %1;", "1.0f", "0.000001f"),
        _fb("f32_sub", "sub.rn.f32", "sub.rn.f32 %0, %0, %1;", "1e6f", "0.000001f"),
        _fb("f32_mul", "mul.rn.f32", "mul.rn.f32 %0, %0, %1;", "1.0f", "0.999999f"),
        _ft("f32_fma", "fma.rn.f32", "fma.rn.f32 %0, %0, %1, %0;", "1.0f", "1.000001f"),
        _fb("f32_min", "min.f32",    "min.f32 %0, %0, %1;", "1.0f", "0.5f"),
        _fb("f32_max", "max.f32",    "max.f32 %0, %0, %1;", "0.1f", "0.5f"),
        # Division
        _fb("f32_div_approx", "div.approx.f32", "div.approx.f32 %0, %0, %1;",
            "1.0f", "0.9999f"),
        _fb("f32_div_rn", "div.rn.f32", "div.rn.f32 %0, %0, %1;",
            "1.0f", "0.9999f"),
        # Unary
        _fu("f32_neg", "neg.f32", "neg.f32 %0, %0;", "1.0f"),
        _fu("f32_abs", "abs.f32", "abs.f32 %0, %0;", "-1.0f"),
        # SFU: approx
        _fu("f32_rcp_approx", "rcp.approx.f32", "rcp.approx.f32 %0, %0;", "2.0f"),
        _fu("f32_rcp_rn",     "rcp.rn.f32",     "rcp.rn.f32 %0, %0;",     "2.0f"),
        _fu("f32_sqrt_approx","sqrt.approx.f32", "sqrt.approx.f32 %0, %0;","4.0f"),
        _fu("f32_sqrt_rn",    "sqrt.rn.f32",     "sqrt.rn.f32 %0, %0;",    "4.0f"),
        _fu("f32_rsqrt",      "rsqrt.approx.f32","rsqrt.approx.f32 %0, %0;","4.0f"),
        _fu("f32_sin",  "sin.approx.f32",  "sin.approx.f32 %0, %0;",  "1.0f"),
        _fu("f32_cos",  "cos.approx.f32",  "cos.approx.f32 %0, %0;",  "1.0f"),
        _fu("f32_lg2",  "lg2.approx.f32",  "lg2.approx.f32 %0, %0;",  "2.0f"),
        _fu("f32_ex2",  "ex2.approx.f32",  "ex2.approx.f32 %0, %0;",  "0.001f"),
        _fu("f32_tanh", "tanh.approx.f32", "tanh.approx.f32 %0, %0;", "0.5f", min_sm=75),
    ]

    # ==================================================================
    # FP16x2 (packed in unsigned, "r" constraint)
    # ==================================================================
    H1 = "0x3C003C00u"      # (1.0h, 1.0h)
    Hk = "0x00010001u"      # tiny
    Hm = "0x3BFF3BFFu"      # ~0.9998
    Hh = "0x38003800u"      # (0.5, 0.5)
    I += [
        _hb("f16x2_add", "add.f16x2", "add.f16x2 %0, %0, %1;", H1, Hk),
        _hb("f16x2_sub", "sub.f16x2", "sub.f16x2 %0, %0, %1;", H1, Hk),
        _hb("f16x2_mul", "mul.f16x2", "mul.f16x2 %0, %0, %1;", H1, Hm),
        Insn("f16x2_fma", "fma.rn.f16x2", "fma.rn.f16x2 %0, %0, %1, %0;",
             "unsigned", "r", H1, [("unsigned", "k0", Hk)], 0, "fp16x2"),
        _hu("f16x2_neg", "neg.f16x2", "neg.f16x2 %0, %0;", H1),
        _hu("f16x2_abs", "abs.f16x2", "abs.f16x2 %0, %0;", H1),
        _hb("f16x2_min", "min.f16x2", "min.f16x2 %0, %0, %1;", H1, Hh, min_sm=80),
        _hb("f16x2_max", "max.f16x2", "max.f16x2 %0, %0, %1;", "0x28002800u", Hh, min_sm=80),
        _hu("f16x2_tanh", "tanh.approx.f16x2", "tanh.approx.f16x2 %0, %0;", Hh, min_sm=75),
        _hu("f16x2_ex2",  "ex2.approx.f16x2",  "ex2.approx.f16x2 %0, %0;",  Hk, min_sm=75),
    ]

    # ==================================================================
    # FP16 scalar (unsigned short + "h" constraint, hex bit patterns)
    # ==================================================================
    cat = "fp16"
    h1  = "(unsigned short)0x3C00u"
    hk  = "(unsigned short)0x211Eu"
    hm  = "(unsigned short)0x3BFFu"
    hh  = "(unsigned short)0x3800u"
    h01 = "(unsigned short)0x2E66u"
    _h = "unsigned short"
    I += [
        Insn("f16_add", "add.f16", "add.f16 %0, %0, %1;",
             _h, "h", h1, [(_h, "k0", hk)], 0, cat),
        Insn("f16_sub", "sub.f16", "sub.f16 %0, %0, %1;",
             _h, "h", h1, [(_h, "k0", hk)], 0, cat),
        Insn("f16_mul", "mul.f16", "mul.f16 %0, %0, %1;",
             _h, "h", h1, [(_h, "k0", hm)], 0, cat),
        Insn("f16_fma", "fma.rn.f16", "fma.rn.f16 %0, %0, %1, %0;",
             _h, "h", h1, [(_h, "k0", hk)], 0, cat),
        Insn("f16_neg", "neg.f16", "neg.f16 %0, %0;",
             _h, "h", h1, [], 0, cat),
        Insn("f16_abs", "abs.f16", "abs.f16 %0, %0;",
             _h, "h", h1, [], 0, cat),
        Insn("f16_min", "min.f16", "min.f16 %0, %0, %1;",
             _h, "h", h1, [(_h, "k0", hh)], 80, cat),
        Insn("f16_max", "max.f16", "max.f16 %0, %0, %1;",
             _h, "h", h01, [(_h, "k0", hh)], 80, cat),
        Insn("f16_tanh", "tanh.approx.f16", "tanh.approx.f16 %0, %0;",
             _h, "h", hh, [], 75, cat),
        Insn("f16_ex2", "ex2.approx.f16", "ex2.approx.f16 %0, %0;",
             _h, "h", hk, [], 75, cat),
    ]

    # ==================================================================
    # BF16x2 (packed in unsigned, "r" constraint, sm_90+)
    # ==================================================================
    B1 = "0x3F803F80u"
    Bk = "0x2C002C00u"
    Bm = "0x3F7F3F7Fu"
    Bh = "0x3F003F00u"
    cat = "bf16x2"
    I += [
        _hb("bf16x2_add", "add.rn.bf16x2", "add.rn.bf16x2 %0, %0, %1;", B1, Bk, 90, cat),
        _hb("bf16x2_sub", "sub.rn.bf16x2", "sub.rn.bf16x2 %0, %0, %1;", B1, Bk, 90, cat),
        _hb("bf16x2_mul", "mul.rn.bf16x2", "mul.rn.bf16x2 %0, %0, %1;", B1, Bm, 90, cat),
        Insn("bf16x2_fma", "fma.rn.bf16x2", "fma.rn.bf16x2 %0, %0, %1, %0;",
             "unsigned", "r", B1, [("unsigned", "k0", Bk)], 90, cat),
        _hu("bf16x2_neg", "neg.bf16x2", "neg.bf16x2 %0, %0;", B1, 90, cat),
        _hu("bf16x2_abs", "abs.bf16x2", "abs.bf16x2 %0, %0;", B1, 90, cat),
        _hb("bf16x2_min", "min.bf16x2", "min.bf16x2 %0, %0, %1;", B1, Bh, 90, cat),
        _hb("bf16x2_max", "max.bf16x2", "max.bf16x2 %0, %0, %1;", "0x2E002E00u", Bh, 90, cat),
    ]

    # ==================================================================
    # BF16 scalar (sm_90+)
    # ==================================================================
    cat = "bf16"
    bf1 = "(unsigned short)0x3F80u"
    bfk = "(unsigned short)0x3C24u"
    bfm = "(unsigned short)0x3F7Fu"
    _b = "unsigned short"
    I += [
        Insn("bf16_add", "add.rn.bf16", "add.rn.bf16 %0, %0, %1;",
             _b, "h", bf1, [(_b, "k0", bfk)], 90, cat),
        Insn("bf16_sub", "sub.rn.bf16", "sub.rn.bf16 %0, %0, %1;",
             _b, "h", bf1, [(_b, "k0", bfk)], 90, cat),
        Insn("bf16_mul", "mul.rn.bf16", "mul.rn.bf16 %0, %0, %1;",
             _b, "h", bf1, [(_b, "k0", bfm)], 90, cat),
        Insn("bf16_fma", "fma.rn.bf16", "fma.rn.bf16 %0, %0, %1, %0;",
             _b, "h", bf1, [(_b, "k0", bfk)], 90, cat),
        Insn("bf16_neg", "neg.bf16", "neg.bf16 %0, %0;",
             _b, "h", bf1, [], 90, cat),
        Insn("bf16_abs", "abs.bf16", "abs.bf16 %0, %0;",
             _b, "h", bf1, [], 90, cat),
    ]

    # ==================================================================
    # INT32 signed
    # ==================================================================
    cat = "int32"
    I += [
        _ib("s32_add", "add.s32", "add.s32 %0, %0, %1;", "1", "1"),
        _ib("s32_sub", "sub.s32", "sub.s32 %0, %0, %1;", "1000000", "1"),
        _ib("s32_mul_lo", "mul.lo.s32", "mul.lo.s32 %0, %0, %1;", "1", "3"),
        _ib("s32_mul_hi", "mul.hi.s32", "mul.hi.s32 %0, %0, %1;", "1000000", "1000000"),
        Insn("s32_mad_lo", "mad.lo.s32", "mad.lo.s32 %0, %0, %1, %0;",
             "int", "r", "1", [("int", "k0", "3")], 0, cat),
        Insn("s32_mad_hi", "mad.hi.s32", "mad.hi.s32 %0, %0, %1, %0;",
             "int", "r", "1", [("int", "k0", "3")], 0, cat),
        _ib("s32_min", "min.s32", "min.s32 %0, %0, %1;", "100", "50"),
        _ib("s32_max", "max.s32", "max.s32 %0, %0, %1;", "1", "50"),
        _iu("s32_abs", "abs.s32", "abs.s32 %0, %0;", "-1"),
        _iu("s32_neg", "neg.s32", "neg.s32 %0, %0;", "1"),
        _ib("s32_div", "div.s32", "div.s32 %0, %0, %1;", "1000000000", "7"),
        _ib("s32_rem", "rem.s32", "rem.s32 %0, %0, %1;", "1000000000", "7"),
        _ib("s32_mul24_lo", "mul24.lo.s32", "mul24.lo.s32 %0, %0, %1;", "1", "3"),
        Insn("s32_mad24_lo", "mad24.lo.s32", "mad24.lo.s32 %0, %0, %1, %0;",
             "int", "r", "1", [("int", "k0", "3")], 0, cat),
        Insn("s32_sad", "sad.s32", "sad.s32 %0, %0, %1, %0;",
             "int", "r", "0", [("int", "k0", "7")], 0, cat),
    ]

    # ==================================================================
    # INT32 unsigned
    # ==================================================================
    I += [
        _ub("u32_add", "add.u32", "add.u32 %0, %0, %1;", "1u", "1u"),
        _ub("u32_sub", "sub.u32", "sub.u32 %0, %0, %1;", "1000000u", "1u"),
        _ub("u32_mul_lo", "mul.lo.u32", "mul.lo.u32 %0, %0, %1;", "1u", "3u"),
        _ub("u32_mul_hi", "mul.hi.u32", "mul.hi.u32 %0, %0, %1;", "1000000u", "1000000u"),
        Insn("u32_mad_lo", "mad.lo.u32", "mad.lo.u32 %0, %0, %1, %0;",
             "unsigned", "r", "1u", [("unsigned", "k0", "3u")], 0, cat),
        _ub("u32_min", "min.u32", "min.u32 %0, %0, %1;", "100u", "50u"),
        _ub("u32_max", "max.u32", "max.u32 %0, %0, %1;", "1u", "50u"),
        _ub("u32_div", "div.u32", "div.u32 %0, %0, %1;", "1000000000u", "7u"),
        _ub("u32_rem", "rem.u32", "rem.u32 %0, %0, %1;", "1000000000u", "7u"),
    ]

    # ==================================================================
    # Bitwise / shift
    # ==================================================================
    cat = "bits"
    I += [
        Insn("b32_and", "and.b32", "and.b32 %0, %0, %1;",
             "unsigned", "r", "0xFFFFFFFFu", [("unsigned", "k0", "0xAAAAAAAAu")], 0, cat),
        Insn("b32_or",  "or.b32",  "or.b32 %0, %0, %1;",
             "unsigned", "r", "0x55555555u", [("unsigned", "k0", "0x00FF00FFu")], 0, cat),
        Insn("b32_xor", "xor.b32", "xor.b32 %0, %0, %1;",
             "unsigned", "r", "0xDEADBEEFu", [("unsigned", "k0", "0x12345678u")], 0, cat),
        _uu("b32_not",  "not.b32", "not.b32 %0, %0;", "0xDEADBEEFu"),
        Insn("b32_shl", "shl.b32", "shl.b32 %0, %0, %1;",
             "unsigned", "r", "1u", [("unsigned", "k0", "1u")], 0, cat),
        Insn("b32_shr", "shr.b32", "shr.b32 %0, %0, %1;",
             "unsigned", "r", "0x80000000u", [("unsigned", "k0", "1u")], 0, cat),
        Insn("s32_shr", "shr.s32", "shr.s32 %0, %0, %1;",
             "int", "r", "-2", [("int", "k0", "1")], 0, cat),
        Insn("b32_cnot", "cnot.b32", "cnot.b32 %0, %0;",
             "unsigned", "r", "0u", [], 0, cat),
    ]

    # ==================================================================
    # Bit manipulation
    # ==================================================================
    cat = "bitmanip"
    I += [
        _uu("b32_popc",  "popc.b32",  "popc.b32 %0, %0;",  "0xDEADBEEFu", cat=cat),
        _uu("b32_clz",   "clz.b32",   "clz.b32 %0, %0;",   "0x00001000u", cat=cat),
        Insn("s32_bfind", "bfind.s32", "bfind.s32 %0, %0;",
             "int", "r", "12345", [], 0, cat),
        _uu("b32_brev",  "brev.b32",  "brev.b32 %0, %0;",  "0xDEADBEEFu", cat=cat),
        Insn("s32_bfe", "bfe.s32", "bfe.s32 %0, %0, %1, %2;",
             "int", "r", "0x12345678",
             [("int", "k0", "4"), ("int", "k1", "8")], 0, cat),
        Insn("u32_bfe", "bfe.u32", "bfe.u32 %0, %0, %1, %2;",
             "unsigned", "r", "0x12345678u",
             [("unsigned", "k0", "4u"), ("unsigned", "k1", "8u")], 0, cat),
        Insn("b32_bfi", "bfi.b32", "bfi.b32 %0, %1, %0, %2, %3;",
             "unsigned", "r", "0xDEADBEEFu",
             [("unsigned", "k0", "0x12345678u"),
              ("unsigned", "k1", "4u"),
              ("unsigned", "k2", "8u")], 0, cat),
    ]

    # ==================================================================
    # Special integer: LOP3, PRMT, DP4A, DP2A
    # ==================================================================
    cat = "special"
    I.append(Insn("b32_lop3", "lop3.b32", "lop3.b32 %0, %0, %1, %2, 0xE8;",
                  "int", "r", "1",
                  [("int", "k0", "0xAAAAAAAA"), ("int", "k1", "0x55555555")], 0, cat))
    I.append(Insn("b32_prmt", "prmt.b32", "prmt.b32 %0, %0, %1, %2;",
                  "unsigned", "r", "0xDEADBEEFu",
                  [("unsigned", "k0", "0xCAFEBABEu"), ("unsigned", "k1", "0x3210u")], 0, cat))

    for at, bt in [("s32","s32"), ("u32","u32"), ("s32","u32"), ("u32","s32")]:
        atag = at[0]
        btag = bt[0]
        I.append(Insn(
            f"dp4a_{atag}{btag}", f"dp4a.{at}.{bt}",
            f"dp4a.{at}.{bt} %0, %1, %2, %0;",
            "int", "r", "0",
            [("int", "k0", "0x01010101"), ("int", "k1", "0x01010101")], 0, cat))

    for hilo in ["lo", "hi"]:
        I.append(Insn(
            f"dp2a_{hilo}_ss", f"dp2a.{hilo}.s32.s32",
            f"dp2a.{hilo}.s32.s32 %0, %1, %2, %0;",
            "int", "r", "0",
            [("int", "k0", "0x00010001"), ("int", "k1", "0x00010001")], 0, cat))

    # ==================================================================
    # Carry chain (add.cc / sub.cc)
    # ==================================================================
    cat = "carry"
    I += [
        _ib("s32_add_cc", "add.cc.s32", "add.cc.s32 %0, %0, %1;", "1", "1", cat=cat),
        _ib("s32_sub_cc", "sub.cc.s32", "sub.cc.s32 %0, %0, %1;", "1000000", "1", cat=cat),
        Insn("s32_addcc_addc", "add.cc+addc",
             "add.cc.s32 %0, %0, %1;\\n\\taddc.s32 %0, %0, %1;",
             "int", "r", "1", [("int", "k0", "1")], 0, cat),
    ]

    return I


if __name__ == "__main__":
    instructions = make_instructions()
    gen_and_build("arithm", instructions, CATEGORIES)
