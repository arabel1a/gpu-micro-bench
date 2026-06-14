#!/usr/bin/env python3
"""
Control-flow & warp-collective PTX benchmark.

Covers ops that don't fit the simple accumulator template in gen_arithm.py:
- Warp shuffles (shfl.sync.bfly/idx/down)
- Warp vote / ballot
- Warp reduce (redux.sync — sm_80+; key throttle question for CMP cards)
- Match (match.sync.any)
- Block / warp barriers (bar.sync, bar.warp.sync, bar.arrive)
- Predicated branch (uniform taken, divergent)

Each op uses custom_tput / custom_lat. Every asm volatile body is wrapped in
`{ ... }` PTX scope so ptxas's loop-unrolling doesn't trigger duplicate-symbol
errors when `.reg` declarations are in the body.
"""

from bench_common import Insn, gen_and_build


CATEGORIES = {
    "Warp shfl":   ["shfl_bfly_b32", "shfl_idx_b32", "shfl_down_b32"],
    "Warp vote":   ["vote_ballot", "vote_all", "vote_any"],
    "Warp redux":  ["redux_add_s32", "redux_min_u32", "redux_or_b32"],
    "Warp match":  ["match_any_b32"],
    "Barrier":     ["syncwarp", "bar_sync_0", "bar_arrive_0"],
    "Branch":      ["pred_branch_taken", "divergent_branch"],
}


LAUNCH = "__global__ void __launch_bounds__(128,1)"


def _asm_block(lines, outs, ins, clobbers=""):
    """Render an asm volatile() with the body wrapped in { ... } PTX scope."""
    body = "\n        ".join([f'"{l}\\n\\t"' for l in ["{"] + lines + ["}"]])
    cl = f':{clobbers}' if clobbers else ''
    return f"""    for(int i=0;i<n;i++) asm volatile(
        {body}
        :{outs}
        :{ins}{cl});"""


# ----------------------------------------------------------------
# Warp shuffle
# ----------------------------------------------------------------
def _shfl_kernel(name, ptx_op, mode):
    """shfl.sync.<mode>.b32 d, a, b, c, membermask  -> 5 operands."""
    if mode == "tput":
        # 8 dst/src regs (%0..%7), then lane=%8, clamp=%9, membermask=%10
        lines = [f"{ptx_op} %{i}, %{i}, %8, %9, %10;" for i in range(8)]
        outs = ",".join(f'"+r"(r{i})' for i in range(8))
        block = _asm_block(lines, outs, '"r"(lane),"r"(clamp),"r"(mask)')
        return f"""
{LAUNCH}
kern_{name}_tput(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0=threadIdx.x, r1=r0+1u, r2=r0+2u, r3=r0+3u,
             r4=r0+4u, r5=r0+5u, r6=r0+6u, r7=r0+7u;
    unsigned lane=1u, clamp=0x1Fu, mask=0xFFFFFFFFu;
{block}
    out[threadIdx.y*32+threadIdx.x] = r0^r1^r2^r3^r4^r5^r6^r7;
}}
"""
    lines = [f"{ptx_op} %0, %0, %1, %2, %3;"] * 8
    block = _asm_block(lines, '"+r"(r0)',
                       '"r"(lane),"r"(clamp),"r"(mask)')
    return f"""
{LAUNCH}
kern_{name}_lat(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    unsigned lane=1u, clamp=0x1Fu, mask=0xFFFFFFFFu;
{block}
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


# ----------------------------------------------------------------
# Vote
# ----------------------------------------------------------------
def _vote_ballot_kernel(name, mode):
    if mode == "tput":
        lines = [".reg .pred p0,p1,p2,p3,p4,p5,p6,p7;"]
        for i in range(8):
            lines.append(f"setp.ne.u32 p{i}, %{i}, 0;")
        for i in range(8):
            lines.append(f"vote.sync.ballot.b32 %{i}, p{i}, %8;")
        outs = ",".join(f'"+r"(r{i})' for i in range(8))
        block = _asm_block(lines, outs, '"r"(mask)')
        return f"""
{LAUNCH}
kern_{name}_tput(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0=threadIdx.x, r1=r0+1u, r2=r0+2u, r3=r0+3u,
             r4=r0+4u, r5=r0+5u, r6=r0+6u, r7=r0+7u;
    unsigned mask=0xFFFFFFFFu;
{block}
    out[threadIdx.y*32+threadIdx.x] = r0^r1^r2^r3^r4^r5^r6^r7;
}}
"""
    lines = [".reg .pred p;"]
    for _ in range(8):
        lines.append("setp.ne.u32 p, %0, 0;")
        lines.append("vote.sync.ballot.b32 %0, p, %1;")
    block = _asm_block(lines, '"+r"(r0)', '"r"(mask)')
    return f"""
{LAUNCH}
kern_{name}_lat(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    unsigned mask=0xFFFFFFFFu;
{block}
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


def _vote_pred_kernel(name, agg, mode):
    """vote.sync.all.pred / .any.pred — pred dst, pred src.
    Same kernel for tput and lat (single chain because pred dst is hard to
    keep 8 independent without inflating register pressure)."""
    lines = [".reg .pred px, py;"]
    for _ in range(8):
        lines.append("setp.ne.u32 px, %0, 0;")
        lines.append(f"vote.sync.{agg}.pred py, px, %1;")
        lines.append("selp.b32 %0, 1, 0, py;")
    block = _asm_block(lines, '"+r"(r0)', '"r"(mask)')
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x & 1;
    unsigned mask=0xFFFFFFFFu;
{block}
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


# ----------------------------------------------------------------
# Redux (sm_80+)
# ----------------------------------------------------------------
def _redux_kernel(name, op_suffix, mode):
    lines = [f"redux.sync.{op_suffix} %0, %0, %1;"] * 8
    block = _asm_block(lines, '"+r"(r0)', '"r"(mask)')
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
#if __CUDA_ARCH__ >= 800
    unsigned r0 = threadIdx.x + 1;
    unsigned mask = 0xFFFFFFFFu;
{block}
    out[threadIdx.y*32+threadIdx.x] = r0;
#endif
}}
"""


# ----------------------------------------------------------------
# Match
# ----------------------------------------------------------------
def _match_kernel(name, mode):
    lines = ["match.sync.any.b32 %0, %0, %1;"] * 8
    block = _asm_block(lines, '"+r"(r0)', '"r"(mask)')
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    unsigned mask = 0xFFFFFFFFu;
{block}
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


# ----------------------------------------------------------------
# Barriers
# ----------------------------------------------------------------
def _syncwarp_kernel(name, mode):
    lines = ["bar.warp.sync 0xFFFFFFFF;"] * 8
    block = _asm_block(lines, "", "", clobbers='"memory"')
    # No outs/ins — adjust the block manually
    body = "\n        ".join([f'"{l}\\n\\t"' for l in ["{"] + lines + ["}"]])
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    for(int i=0;i<n;i++) asm volatile(
        {body}
        :::"memory");
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


def _bar_sync_kernel(name, mode):
    lines = ["bar.sync 0;"] * 8
    body = "\n        ".join([f'"{l}\\n\\t"' for l in ["{"] + lines + ["}"]])
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    for(int i=0;i<n;i++) asm volatile(
        {body}
        :::"memory");
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


def _bar_arrive_kernel(name, mode):
    # bar.arrive doesn't block; pair with bar.sync at end of iter to keep
    # barrier-state coherent (otherwise the arrival counter overflows the
    # barrier and undefined behavior).
    lines = ["bar.arrive 0, 128;"] * 8 + ["bar.sync 0;"]
    body = "\n        ".join([f'"{l}\\n\\t"' for l in ["{"] + lines + ["}"]])
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    for(int i=0;i<n;i++) asm volatile(
        {body}
        :::"memory");
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


# ----------------------------------------------------------------
# Branches
# ----------------------------------------------------------------
def _pred_branch_kernel(name, mode):
    """Uniform predicated branch — always taken to immediately following insn.
    Each branch needs a unique label since PTX scope `{ }` doesn't scope labels.
    `#pragma unroll 1` prevents ptxas from unrolling the for-loop and duplicating
    the asm block (which would re-create label collisions)."""
    lines = [".reg .pred p;",
             "setp.ne.u32 p, %0, 0xFFFFFFFF;"]
    for k in range(8):
        lines.append(f"@p bra Lp{k};")
        lines.append(f"Lp{k}: add.u32 %0, %0, 1;")
    body = "\n        ".join([f'"{l}\\n\\t"' for l in ["{"] + lines + ["}"]])
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    #pragma unroll 1
    for(int i=0;i<n;i++) asm volatile(
        {body}
        :"+r"(r0));
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


def _divergent_branch_kernel(name, mode):
    """Half-and-half divergence based on threadIdx low bit. 4 branches/iter."""
    lines = [".reg .pred p;",
             "and.b32 %0, %0, 1;",
             "setp.ne.u32 p, %0, 0;"]
    for tag in ["a", "b", "c", "d"]:
        lines += [
            f"@p bra LD{tag};",
            "add.u32 %0, %0, 1;",
            f"bra LD{tag}_end;",
            f"LD{tag}: add.u32 %0, %0, 2;",
            f"LD{tag}_end:",
            "setp.ne.u32 p, %0, 0;",
        ]
    body = "\n        ".join([f'"{l}\\n\\t"' for l in ["{"] + lines + ["}"]])
    return f"""
{LAUNCH}
kern_{name}_{mode}(volatile unsigned* __restrict__ out, int n) {{
    unsigned r0 = threadIdx.x;
    #pragma unroll 1
    for(int i=0;i<n;i++) asm volatile(
        {body}
        :"+r"(r0));
    out[threadIdx.y*32+threadIdx.x] = r0;
}}
"""


def make_instructions():
    I = []
    cat = "warp_shfl"
    for name, op in [("shfl_bfly_b32",  "shfl.sync.bfly.b32"),
                     ("shfl_idx_b32",   "shfl.sync.idx.b32"),
                     ("shfl_down_b32",  "shfl.sync.down.b32")]:
        I.append(Insn(name, op, "", "unsigned", "r", "0", [], 75, cat,
                      custom_tput=_shfl_kernel(name, op, "tput"),
                      custom_lat=_shfl_kernel(name, op, "lat")))

    cat = "warp_vote"
    I.append(Insn("vote_ballot", "vote.sync.ballot.b32", "",
                  "unsigned", "r", "0", [], 75, cat,
                  custom_tput=_vote_ballot_kernel("vote_ballot", "tput"),
                  custom_lat=_vote_ballot_kernel("vote_ballot", "lat")))
    for name, agg in [("vote_all", "all"), ("vote_any", "any")]:
        I.append(Insn(name, f"vote.sync.{agg}.pred", "",
                      "unsigned", "r", "0", [], 75, cat,
                      custom_tput=_vote_pred_kernel(name, agg, "tput"),
                      custom_lat=_vote_pred_kernel(name, agg, "lat")))

    cat = "warp_redux"
    for name, sfx in [("redux_add_s32", "add.s32"),
                      ("redux_min_u32", "min.u32"),
                      ("redux_or_b32",  "or.b32")]:
        I.append(Insn(name, f"redux.sync.{sfx}", "",
                      "unsigned", "r", "0", [], 80, cat,
                      custom_tput=_redux_kernel(name, sfx, "tput"),
                      custom_lat=_redux_kernel(name, sfx, "lat")))

    cat = "warp_match"
    I.append(Insn("match_any_b32", "match.sync.any.b32", "",
                  "unsigned", "r", "0", [], 75, cat,
                  custom_tput=_match_kernel("match_any_b32", "tput"),
                  custom_lat=_match_kernel("match_any_b32", "lat")))

    cat = "barrier"
    I.append(Insn("syncwarp", "bar.warp.sync", "",
                  "unsigned", "r", "0", [], 75, cat,
                  custom_tput=_syncwarp_kernel("syncwarp", "tput"),
                  custom_lat=_syncwarp_kernel("syncwarp", "lat")))
    I.append(Insn("bar_sync_0", "bar.sync 0", "",
                  "unsigned", "r", "0", [], 75, cat,
                  custom_tput=_bar_sync_kernel("bar_sync_0", "tput"),
                  custom_lat=_bar_sync_kernel("bar_sync_0", "lat")))
    I.append(Insn("bar_arrive_0", "bar.arrive 0,N", "",
                  "unsigned", "r", "0", [], 75, cat,
                  custom_tput=_bar_arrive_kernel("bar_arrive_0", "tput"),
                  custom_lat=_bar_arrive_kernel("bar_arrive_0", "lat")))

    cat = "branch"
    I.append(Insn("pred_branch_taken", "@p bra", "",
                  "unsigned", "r", "0", [], 75, cat,
                  custom_tput=_pred_branch_kernel("pred_branch_taken", "tput"),
                  custom_lat=_pred_branch_kernel("pred_branch_taken", "lat")))
    I.append(Insn("divergent_branch", "@p bra (divergent)", "",
                  "unsigned", "r", "0", [], 75, cat,
                  custom_tput=_divergent_branch_kernel("divergent_branch", "tput"),
                  custom_lat=_divergent_branch_kernel("divergent_branch", "lat")))

    return I


if __name__ == "__main__":
    instructions = make_instructions()
    gen_and_build("ctrl", instructions, CATEGORIES)
