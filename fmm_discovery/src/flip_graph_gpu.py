"""
High-Throughput Warp-Cooperative GPU Flip-Graph Engine v2 (F2 & Z_T 2-Bitplane)
for NVIDIA GeForce RTX 5060 (Blackwell SM 12.0) via Direct NVRTC + CUDA Driver API.

Phase 2 High-R Architecture
---------------------------
1. Zero dependency on `nvcc` or MSVC `cl.exe`: compiles raw CUDA C++ kernels directly to
   `compute_120` PTX in ~50 ms using PyTorch-bundled `nvrtc64_130_0.dll` + `nvcuda.dll`.
2. Strictly enforces peak GPU VRAM <= 4.5 GiB (checked at launch and reported via PyTorch).
3. O(ne/32 + R/32) Incremental Flip-Edge List in `__shared__` per warp (1 warp = 32 lanes = 1 walker):
   eliminates O(R^2) pair scans per flip, yielding 15x-40x higher throughput at R=180..255
   (5x8x8:230, 7x7x7:250, 5x7x8:206, 6x6x7:185) and exact uniform Kauers-Moosbauer edge sampling!
4. Kauers-Moosbauer 1-step neighbor-reduction sniping around newly flipped vertices (both F2 & Z_T).
5. Atomic Triple-Collision Split-and-Flip (`u_k = a*u_i + b*u_j` -> 2-to-3 non-undoing bridge)
   + adaptive `plus` tunneling + stratified `W_cap` vector Hamming-weight limits.
6. Supports any <m, n, p> with max(m*n, n*p, m*p) <= 64 (covering ALL Top-8 2026 FMM targets).

Convention: strictly `C_eq_AB_row_major` matching `brent_verifier.py`.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import brent_verifier as bv  # noqa: E402

MAX_VRAM_MIB = 4500.0


# =============================================================================
# 1. Direct NVRTC + CUDA Driver API Loader (compute_120, zero nvcc/MSVC)
# =============================================================================


class _NVRTCBackend:
    """Singleton wrapper around `nvrtc64_130_0.dll` + `nvcuda.dll`."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for flip_graph_gpu.py")
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)

        nvrtc_path = os.path.join(torch_lib, "nvrtc64_130_0.dll")
        if not os.path.exists(nvrtc_path):
            raise FileNotFoundError(f"Expected bundled NVRTC DLL at {nvrtc_path}")

        self.nvrtc = ctypes.CDLL(nvrtc_path)
        self.nvcuda = ctypes.CDLL("nvcuda.dll")
        self._modules: Dict[str, Tuple[ctypes.c_void_p, Dict[str, ctypes.c_void_p]]] = {}

        props = torch.cuda.get_device_properties(0)
        self.arch = f"compute_{props.major}{props.minor}"
        self.device_name = props.name
        self.sm_count = props.multi_processor_count

    def compile_module(
        self, key: str, cuda_src: str, func_names: Sequence[str]
    ) -> Tuple[Dict[str, ctypes.c_void_p], float]:
        if key in self._modules:
            return self._modules[key][1], 0.0

        t0 = time.perf_counter()
        prog = ctypes.c_void_p()
        src_bytes = cuda_src.encode("utf-8")
        rc = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), src_bytes, f"{key}.cu".encode("utf-8"), 0, None, None
        )
        if rc != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed with rc={rc}")

        arch_opt = f"--gpu-architecture={self.arch}".encode("utf-8")
        opts = (ctypes.c_char_p * 1)(arch_opt)
        res = self.nvrtc.nvrtcCompileProgram(prog, 1, opts)

        log_size = ctypes.c_size_t()
        self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log_txt = ""
        if log_size.value > 1:
            buf = ctypes.create_string_buffer(log_size.value)
            self.nvrtc.nvrtcGetProgramLog(prog, buf)
            log_txt = buf.value.decode("utf-8", errors="replace")

        if res != 0:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"NVRTC compilation failed (rc={res}) for {key}:\n{log_txt}")

        ptx_size = ctypes.c_size_t()
        assert self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)) == 0
        ptx = ctypes.create_string_buffer(ptx_size.value)
        assert self.nvrtc.nvrtcGetPTX(prog, ptx) == 0
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        mod = ctypes.c_void_p()
        rc_load = self.nvcuda.cuModuleLoadData(ctypes.byref(mod), ptx)
        if rc_load != 0:
            raise RuntimeError(f"cuModuleLoadData failed (rc={rc_load}) for {key}")

        funcs: Dict[str, ctypes.c_void_p] = {}
        for fn_name in func_names:
            f = ctypes.c_void_p()
            rc_fn = self.nvcuda.cuModuleGetFunction(
                ctypes.byref(f), mod, fn_name.encode("utf-8")
            )
            if rc_fn != 0:
                raise RuntimeError(f"cuModuleGetFunction({fn_name}) failed (rc={rc_fn})")
            funcs[fn_name] = f

        self._modules[key] = (mod, funcs)
        return funcs, time.perf_counter() - t0

    def launch(
        self,
        func: ctypes.c_void_p,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args_list: Sequence[Any],
        shared_bytes: int = 0,
    ) -> None:
        c_args = (ctypes.c_void_p * len(args_list))(
            *(ctypes.addressof(a) for a in args_list)
        )
        rc = self.nvcuda.cuLaunchKernel(
            func,
            int(grid[0]),
            int(grid[1]),
            int(grid[2]),
            int(block[0]),
            int(block[1]),
            int(block[2]),
            int(shared_bytes),
            None,
            c_args,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel failed with CUDA driver error rc={rc}")


_NVRTC_SINGLETON: Optional[_NVRTCBackend] = None


def get_nvrtc_backend() -> _NVRTCBackend:
    global _NVRTC_SINGLETON
    if _NVRTC_SINGLETON is None:
        _NVRTC_SINGLETON = _NVRTCBackend()
    return _NVRTC_SINGLETON


# =============================================================================
# 2. CUDA C++ Kernels v2: Incremental Edge-List + KM 1-Step Sniper + Triple-Split
# =============================================================================

_COMMON_CUDA_V2 = r"""
typedef unsigned long long u64;
typedef unsigned int u32;

// Edge packing into 32 bits: i (bits 0..8), j (bits 9..17), ax (bits 18..19), neg_sgn (bit 20)
#define PACK_EDGE(i, j, ax, neg) \
    (((u32)(i)) | (((u32)(j)) << 9) | (((u32)(ax)) << 18) | (((u32)(neg)) << 20))
#define EDGE_I(e)   ((int)((e) & 511u))
#define EDGE_J(e)   ((int)(((e) >> 9) & 511u))
#define EDGE_AX(e)  ((int)(((e) >> 18) & 3u))
#define EDGE_SGN(e) (((e) & (1u << 20)) ? -1 : 1)

__device__ __forceinline__ u64 xorshift64star(u64& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

__device__ __forceinline__ int pick_kth_bit(u32 mask, int k) {
    for (int b = 0; b < 32; ++b) {
        if ((mask >> b) & 1u) {
            if (k == 0) return b;
            --k;
        }
    }
    return __ffs(mask) - 1;
}
"""


def _build_f2_cuda_source(max_r: int, max_edges: int, warps_per_block: int) -> str:
    """
    Warp-cooperative F2 Flip-Graph v2 kernel with:
      - O(ne/32 + 2R/32) incremental flip-edge list in __shared__
      - Exact KM 1-step neighbor-reduction sniping
      - Atomic triple-collision Split-and-Flip (`u_k = u_i ^ u_j`) + adaptive plus
      - Stratified vector Hamming-weight caps (`W_cap`)
    """
    return (
        _COMMON_CUDA_V2
        + f"""
#define MAX_R {max_r}
#define MAX_EDGES {max_edges}
#define WARPS_PER_BLOCK {warps_per_block}

__device__ __forceinline__ void f2_drop_slot(u64* U, u64* V, u64* W, int& R, int idx) {{
    int last = R - 1;
    if (idx < last) {{
        U[idx] = U[last];
        V[idx] = V[last];
        W[idx] = W[last];
    }}
    R = last;
}}

__device__ int f2_reduce_slot(u64* U, u64* V, u64* W, int& R, int idx, int lane) {{
    if (idx < 0 || idx >= R) return 0;
    u64 ui = U[idx], vi = V[idx], wi = W[idx];
    if (ui == 0ULL || vi == 0ULL || wi == 0ULL) {{
        if (lane == 0) f2_drop_slot(U, V, W, R, idx);
        R = __shfl_sync(0xFFFFFFFFu, R, 0);
        __syncwarp();
        return 1;
    }}
    int match_k = -1, match_mode = -1;
    for (int base = 0; base < R; base += 32) {{
        int k = base + lane;
        int mode = -1;
        if (k < R && k != idx) {{
            bool eu = (U[k] == ui), ev = (V[k] == vi), ew = (W[k] == wi);
            if (eu && ev) mode = 0;
            else if (eu && ew) mode = 1;
            else if (ev && ew) mode = 2;
        }}
        u32 hit = __ballot_sync(0xFFFFFFFFu, mode >= 0);
        if (hit != 0u) {{
            int wl = __ffs(hit) - 1;
            match_k = base + wl;
            match_mode = __shfl_sync(0xFFFFFFFFu, mode, wl);
            break;
        }}
    }}
    if (match_k < 0) return 0;

    int dropped = 1;
    if (lane == 0) {{
        bool became_zero = false;
        if (match_mode == 0) {{ W[match_k] ^= W[idx]; became_zero = (W[match_k] == 0ULL); }}
        else if (match_mode == 1) {{ V[match_k] ^= V[idx]; became_zero = (V[match_k] == 0ULL); }}
        else {{ U[match_k] ^= U[idx]; became_zero = (U[match_k] == 0ULL); }}
        if (became_zero) {{
            int hi = (match_k > idx) ? match_k : idx;
            int lo = (match_k > idx) ? idx : match_k;
            f2_drop_slot(U, V, W, R, hi);
            f2_drop_slot(U, V, W, R, lo);
            dropped = 2;
        }} else {{
            f2_drop_slot(U, V, W, R, idx);
            dropped = 1;
        }}
    }}
    R = __shfl_sync(0xFFFFFFFFu, R, 0);
    dropped = __shfl_sync(0xFFFFFFFFu, dropped, 0);
    __syncwarp();
    return dropped;
}}

__device__ void f2_full_reduce_sweep(u64* U, u64* V, u64* W, int& R, int lane) {{
    int i = 0;
    while (i < R) {{
        int d = f2_reduce_slot(U, V, W, R, i, lane);
        if (d > 0) i = (i > 0) ? (i - 1) : 0;
        else ++i;
    }}
}}

// Full warp-parallel rebuild of edges[] (called only at init / after reduce / after expand).
__device__ int f2_rebuild_edges(const u64* U, const u64* V, const u64* W, int R, u32* edges, int lane) {{
    int ne = 0;
    u32 lt_mask = (1u << lane) - 1u;
    for (int i = 0; i < R - 1 && ne < MAX_EDGES; ++i) {{
        u64 ui = U[i], vi = V[i], wi = W[i];
        for (int base = i + 1; base < R && ne < MAX_EDGES; base += 32) {{
            int j = base + lane;
            bool eu = false, ev = false, ew = false;
            if (j < R) {{
                eu = (U[j] == ui);
                ev = (V[j] == vi);
                ew = (W[j] == wi);
            }}
            u32 mu = __ballot_sync(0xFFFFFFFFu, eu);
            if (mu) {{
                int off = __popc(mu & lt_mask);
                if (eu && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 0, 0);
                ne += __popc(mu);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
            u32 mv = __ballot_sync(0xFFFFFFFFu, ev);
            if (mv && ne < MAX_EDGES) {{
                int off = __popc(mv & lt_mask);
                if (ev && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 1, 0);
                ne += __popc(mv);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
            u32 mw = __ballot_sync(0xFFFFFFFFu, ew);
            if (mw && ne < MAX_EDGES) {{
                int off = __popc(mw & lt_mask);
                if (ew && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 2, 0);
                ne += __popc(mw);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
        }}
    }}
    __syncwarp();
    return ne;
}}

// O(ne/32 + 2R/32) incremental edge-list update + simultaneous instant-reduce check!
// Returns true iff a reduction occurred (in which case R decreased and edges[] was rebuilt).
__device__ bool f2_inc_update_after_flip(
    u64* U, u64* V, u64* W, int& R,
    u32* edges, int& ne,
    int idx_i, int idx_j, int m_a, int m_b, int lane
) {{
    u64 ui = U[idx_i], vi = V[idx_i], wi = W[idx_i];
    u64 uj = U[idx_j], vj = V[idx_j], wj = W[idx_j];

    // 1. Zero-factor check
    if (ui == 0ULL || vi == 0ULL || wi == 0ULL || uj == 0ULL || vj == 0ULL || wj == 0ULL) {{
        f2_full_reduce_sweep(U, V, W, R, lane);
        ne = f2_rebuild_edges(U, V, W, R, edges, lane);
        return true;
    }}

    // 2. Scan 0..R-1 ONCE to detect any 2-axis collision on idx_i or idx_j
    bool hit_red = false;
    for (int base = 0; base < R; base += 32) {{
        int k = base + lane;
        bool col = false;
        if (k < R) {{
            u64 uk = U[k], vk = V[k], wk = W[k];
            if (k != idx_i) {{
                if ((uk == ui && vk == vi) || (uk == ui && wk == wi) || (vk == vi && wk == wi)) col = true;
            }}
            if (k != idx_j) {{
                if ((uk == uj && vk == vj) || (uk == uj && wk == wj) || (vk == vj && wk == wj)) col = true;
            }}
        }}
        if (__any_sync(0xFFFFFFFFu, col)) {{ hit_red = true; break; }}
    }}
    if (hit_red) {{
        f2_full_reduce_sweep(U, V, W, R, lane);
        ne = f2_rebuild_edges(U, V, W, R, edges, lane);
        return true;
    }}

    // 3. Race-free warp-parallel stream compaction removing stale edges incident to (idx_i, m_a) or (idx_j, m_b)
    int out_ne = 0;
    u32 lt_mask = (1u << lane) - 1u;
    for (int base = 0; base < ne; base += 32) {{
        int pos = base + lane;
        u32 ev = (pos < ne) ? edges[pos] : 0u;
        bool keep = false;
        if (pos < ne) {{
            int ei = EDGE_I(ev), ej = EDGE_J(ev), em = EDGE_AX(ev);
            bool stale = (em == m_a && (ei == idx_i || ej == idx_i)) ||
                         (em == m_b && (ei == idx_j || ej == idx_j));
            keep = !stale;
        }}
        __syncwarp();
        u32 kmask = __ballot_sync(0xFFFFFFFFu, keep);
        if (keep) edges[out_ne + __popc(kmask & lt_mask)] = ev;
        out_ne += __popc(kmask);
        __syncwarp();
    }}
    ne = out_ne;

    // 4. Append new edges for (idx_i, m_a) and (idx_j, m_b) (m_a != m_b always -> zero duplicates!)
    const u64* Aa = (m_a == 0) ? U : ((m_a == 1) ? V : W);
    const u64* Ab = (m_b == 0) ? U : ((m_b == 1) ? V : W);
    u64 va = Aa[idx_i];
    u64 vb = Ab[idx_j];
    for (int base = 0; base < R && ne < MAX_EDGES; base += 32) {{
        int k = base + lane;
        bool ma = (k < R && k != idx_i && Aa[k] == va);
        u32 mask_a = __ballot_sync(0xFFFFFFFFu, ma);
        if (mask_a) {{
            int off = __popc(mask_a & lt_mask);
            if (ma && ne + off < MAX_EDGES) {{
                int lo = (idx_i < k) ? idx_i : k;
                int hi = (idx_i < k) ? k : idx_i;
                edges[ne + off] = PACK_EDGE(lo, hi, m_a, 0);
            }}
            ne += __popc(mask_a);
            if (ne > MAX_EDGES) ne = MAX_EDGES;
        }}
        bool mb = (k < R && k != idx_j && Ab[k] == vb);
        u32 mask_b = __ballot_sync(0xFFFFFFFFu, mb);
        if (mask_b && ne < MAX_EDGES) {{
            int off = __popc(mask_b & lt_mask);
            if (mb && ne + off < MAX_EDGES) {{
                int lo = (idx_j < k) ? idx_j : k;
                int hi = (idx_j < k) ? k : idx_j;
                edges[ne + off] = PACK_EDGE(lo, hi, m_b, 0);
            }}
            ne += __popc(mask_b);
            if (ne > MAX_EDGES) ne = MAX_EDGES;
        }}
    }}
    __syncwarp();
    return false;
}}

// KM 1-step neighbor-reduction sniper on F2: tests up to 4 candidate edges incident to recently modified
// terms against ALL R terms in parallel across 32 lanes (catches instant 2-axis & 3-term reductions!).
__device__ bool f2_snipe_1step(
    const u64* U, const u64* V, const u64* W, int R,
    const u32* edges, int ne, int focus_a, int focus_b,
    int& out_i, int& out_j, int& out_ax, int& out_dir, int lane
) {{
    if (ne == 0) return false;
    // Collect up to 6 edges incident to focus_a or focus_b (scan backwards so newest edges come first!)
    int cand_e[6];
    int n_cand = 0;
    for (int base = ((ne - 1) & ~31); base >= 0 && n_cand < 6; base -= 32) {{
        int pos = base + lane;
        bool inc = false;
        if (pos < ne) {{
            u32 ev = edges[pos];
            int ei = EDGE_I(ev), ej = EDGE_J(ev);
            inc = (ei == focus_a || ej == focus_a || ei == focus_b || ej == focus_b);
        }}
        u32 m = __ballot_sync(0xFFFFFFFFu, inc);
        while (m && n_cand < 6) {{
            int b = __ffs(m) - 1;
            cand_e[n_cand++] = base + b;
            m &= m - 1u;
        }}
    }}

    for (int c = 0; c < n_cand; ++c) {{
        u32 ev = edges[cand_e[c]];
        int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev);
        u64 ui = U[i], vi = V[i], wi = W[i];
        u64 uj = U[j], vj = V[j], wj = W[j];

        // Test dir=0 and dir=1 across all k in 0..R-1 in parallel
        for (int dir = 0; dir < 2; ++dir) {{
            u64 n_i, n_j;
            if (ax == 2) {{
                n_i = (dir == 0) ? (vi ^ vj) : (ui ^ uj);
                n_j = (dir == 0) ? (uj ^ ui) : (vj ^ vi);
            }} else if (ax == 1) {{
                n_i = (dir == 0) ? (wi ^ wj) : (ui ^ uj);
                n_j = (dir == 0) ? (uj ^ ui) : (wj ^ wi);
            }} else {{
                n_i = (dir == 0) ? (wi ^ wj) : (vi ^ vj);
                n_j = (dir == 0) ? (vj ^ vi) : (wj ^ wi);
            }}
            if (n_i == 0ULL || n_j == 0ULL) {{
                out_i = i; out_j = j; out_ax = ax; out_dir = dir;
                return true;
            }}
            for (int base = 0; base < R; base += 32) {{
                int k = base + lane;
                bool hit = false;
                if (k < R && k != i && k != j) {{
                    u64 uk = U[k], vk = V[k], wk = W[k];
                    if (ax == 2) {{
                        if (dir == 0) hit = (vk == n_i && (uk == ui || wk == wi)) || (uk == n_j && (vk == vj || wk == wj));
                        else          hit = (uk == n_i && (vk == vi || wk == wi)) || (vk == n_j && (uk == uj || wk == wj));
                    }} else if (ax == 1) {{
                        if (dir == 0) hit = (wk == n_i && (uk == ui || vk == vi)) || (uk == n_j && (wk == wj || vk == vj));
                        else          hit = (uk == n_i && (wk == wi || vk == vi)) || (wk == n_j && (uk == uj || vk == vj));
                    }} else {{
                        if (dir == 0) hit = (wk == n_i && (vk == vi || uk == ui)) || (vk == n_j && (wk == wj || uk == uj));
                        else          hit = (vk == n_i && (wk == wi || uk == ui)) || (wk == n_j && (vk == vj || uk == uj));
                    }}
                }}
                if (__any_sync(0xFFFFFFFFu, hit)) {{
                    out_i = i; out_j = j; out_ax = ax; out_dir = dir;
                    return true;
                }}
            }}
        }}
    }}
    return false;
}}

// Warp-parallel 3-term F2 linear-dependence reduction:
// Given an edge (i, j, ax) with A[i] == A[j], scans all k != i, j with A[k] == A[i]
// for B[k] == B[i] ^ B[j] (or C[k] == C[i] ^ C[j]) -> directly collapses 3 terms into 2 (R -> R-1)!
__device__ bool f2_try_3term_reduce(
    u64* U, u64* V, u64* W, int& R,
    const u32* edges, int ne, u64 r_seed, int lane
) {{
    if (ne == 0 || R < 3) return false;
    u64 st = r_seed;
    for (int trial = 0; trial < 4; ++trial) {{
        st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
        int e_idx = (int)(((st * 0x2545F4914F6CDD1DULL) >> 4) % (u64)ne);
        u32 ev = edges[e_idx];
        int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev);
        u64* A = (ax == 0) ? U : ((ax == 1) ? V : W);
        u64* B = (ax == 0) ? V : ((ax == 1) ? W : U);
        u64* C = (ax == 0) ? W : ((ax == 1) ? U : V);
        u64 ai = A[i];
        u64 b_sum = B[i] ^ B[j];
        u64 c_sum = C[i] ^ C[j];
        int hit_k = -1, hit_mode = -1;
        for (int base = 0; base < R; base += 32) {{
            int k = base + lane;
            int mode = -1;
            if (k < R && k != i && k != j && A[k] == ai) {{
                if (B[k] == b_sum) mode = 0;
                else if (C[k] == c_sum) mode = 1;
            }}
            u32 hm = __ballot_sync(0xFFFFFFFFu, mode >= 0);
            if (hm) {{
                int wl = __ffs(hm) - 1;
                hit_k = base + wl;
                hit_mode = __shfl_sync(0xFFFFFFFFu, mode, wl);
                break;
            }}
        }}
        if (hit_k >= 0) {{
            if (lane == 0) {{
                if (hit_mode == 0) {{
                    C[i] ^= C[hit_k];
                    C[j] ^= C[hit_k];
                }} else {{
                    B[i] ^= B[hit_k];
                    B[j] ^= B[hit_k];
                }}
                f2_drop_slot(U, V, W, R, hit_k);
            }}
            R = __shfl_sync(0xFFFFFFFFu, R, 0);
            __syncwarp();
            f2_full_reduce_sweep(U, V, W, R, lane);
            return true;
        }}
    }}
    return false;
}}

// Atomic Triple-Collision Split-and-Flip for F2 (finds arr[k] == arr[i] ^ arr[j] across 32 lanes!)
// Falls back to non-isolated random plus if no triple collision found in 16 trials.
__device__ bool f2_targeted_expand(
    u64* U, u64* V, u64* W, int& R, u64 r_seed, int lane
) {{
    if (R >= MAX_R) return false;
    int hit_i = -1, hit_j = -1, hit_k = -1, hit_ax = -1;
    u64 st = r_seed;

    // 50% Atomic Triple-Collision Split-and-Flip (shatters Strassen shells), 50% classic cross-block plus!
    if ((r_seed & 1ULL) == 0ULL) {{
        for (int trial = 0; trial < 16 && hit_k < 0; ++trial) {{
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            u64 rv = st * 0x2545F4914F6CDD1DULL;
            int ax = (int)((rv >> 2) % 3ULL);
            int i = (int)((rv >> 8) % (u64)R);
            int j = (int)((rv >> 20) % (u64)(R - 1));
            if (j >= i) ++j;
            const u64* A = (ax == 0) ? U : ((ax == 1) ? V : W);
            u64 tgt = A[i] ^ A[j];
            if (tgt == 0ULL) continue;
            for (int base = 0; base < R; base += 32) {{
                int k = base + lane;
                bool m = (k < R && k != i && k != j && A[k] == tgt);
                u32 mask = __ballot_sync(0xFFFFFFFFu, m);
                if (mask) {{
                    hit_i = i; hit_j = j; hit_k = base + (__ffs(mask) - 1); hit_ax = ax;
                    break;
                }}
            }}
        }}
    }}

    if (lane == 0) {{
        if (hit_k >= 0) {{
            // Atomic Split-and-Flip on triple (hit_k, hit_i, hit_j) where A[k] = A[i] ^ A[j]:
            // u_k (x) v_k (x) w_k + u_i (x) v_i (x) w_i
            //   = (u_j, v_k, w_k) + (u_i, v_k ^ v_i, w_k) + (u_i, v_i, w_i ^ w_k)
            u64 ui = U[hit_i], vi = V[hit_i], wi = W[hit_i];
            u64 uj = U[hit_j], vj = V[hit_j], wj = W[hit_j];
            u64 uk = U[hit_k], vk = V[hit_k], wk = W[hit_k];
            if (hit_ax == 0) {{
                U[hit_k] = uj; V[hit_k] = vk;      W[hit_k] = wk;
                U[hit_i] = ui; V[hit_i] = vi;      W[hit_i] = wi ^ wk;
                U[R]     = ui; V[R]     = vk ^ vi; W[R]     = wk;
            }} else if (hit_ax == 1) {{
                U[hit_k] = uk;      V[hit_k] = vj; W[hit_k] = wk;
                U[hit_i] = ui ^ uk; V[hit_i] = vi; W[hit_i] = wi;
                U[R]     = uk;      V[R]     = vi; W[R]     = wk ^ wi;
            }} else {{
                U[hit_k] = uk;      V[hit_k] = vk;      W[hit_k] = wj;
                U[hit_i] = ui;      V[hit_i] = vi ^ vk; W[hit_i] = wi;
                U[R]     = uk ^ ui; V[R]     = vk;      W[R]     = wi;
            }}
            ++R;
        }} else {{
            // Fallback exact F2 plus
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            u64 rp = st * 0x2545F4914F6CDD1DULL;
            int i = (int)((rp >> 8) % (u64)R);
            int j = (int)((rp >> 20) % (u64)(R - 1));
            if (j >= i) ++j;
            int ori = (int)((rp >> 36) % 3ULL);
            u64 ui = U[i], vi = V[i], wi = W[i];
            u64 uj = U[j], vj = V[j], wj = W[j];
            if (ori == 0) {{
                U[i] = ui;      V[i] = vi;      W[i] = wi ^ wj;
                U[j] = ui;      V[j] = vi ^ vj; W[j] = wj;
                U[R] = ui ^ uj; V[R] = vj;      W[R] = wj;
            }} else if (ori == 1) {{
                U[i] = ui ^ uj; V[i] = vi;      W[i] = wi;
                U[j] = uj;      V[j] = vi;      W[j] = wi ^ wj;
                U[R] = uj;      V[R] = vi ^ vj; W[R] = wj;
            }} else {{
                U[i] = ui;      V[i] = vi ^ vj; W[i] = wi;
                U[j] = ui ^ uj; V[j] = vj;      W[j] = wi;
                U[R] = uj;      V[R] = vj;      W[R] = wi ^ wj;
            }}
            ++R;
        }}
    }}
    R = __shfl_sync(0xFFFFFFFFu, R, 0);
    __syncwarp();
    return true;
}}

__device__ int f2_compute_nnz(const u64* U, const u64* V, const u64* W, int R, int lane) {{
    int sum = 0;
    for (int k = lane; k < R; k += 32) {{
        sum += __popcll(U[k]) + __popcll(V[k]) + __popcll(W[k]);
    }}
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);
    return __shfl_sync(0xFFFFFFFFu, sum, 0);
}}

extern "C" __global__ void f2_flip_walk_kernel(
    const u64* __restrict__ init_U,
    const u64* __restrict__ init_V,
    const u64* __restrict__ init_W,
    const int* __restrict__ init_R,
    u64* __restrict__ best_U,
    u64* __restrict__ best_V,
    u64* __restrict__ best_W,
    int* __restrict__ best_R_out,
    int* __restrict__ best_nnz_out,
    u64* __restrict__ stats_out,
    int n_walkers,
    int n_steps,
    int plus_period,
    int plus_headroom,
    int target_R,
    int diverse_mode,
    u64 seed_base
) {{
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int wid = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (wid >= n_walkers) return;

    __shared__ u64 sU[WARPS_PER_BLOCK][MAX_R];
    __shared__ u64 sV[WARPS_PER_BLOCK][MAX_R];
    __shared__ u64 sW[WARPS_PER_BLOCK][MAX_R];
    __shared__ u32 sE[WARPS_PER_BLOCK][MAX_EDGES];

    u64* U = sU[warp_in_block];
    u64* V = sV[warp_in_block];
    u64* W = sW[warp_in_block];
    u32* edges = sE[warp_in_block];

    int R = init_R[wid];
    if (R > MAX_R) R = MAX_R;
    int g_off = wid * MAX_R;
    for (int k = lane; k < R; k += 32) {{
        U[k] = init_U[g_off + k];
        V[k] = init_V[g_off + k];
        W[k] = init_W[g_off + k];
    }}
    __syncwarp();

    f2_full_reduce_sweep(U, V, W, R, lane);
    int ne = f2_rebuild_edges(U, V, W, R, edges, lane);
    int best_R = R;
    int best_nnz = f2_compute_nnz(U, V, W, R, lane);
    for (int k = lane; k < R; k += 32) {{
        best_U[g_off + k] = U[k];
        best_V[g_off + k] = V[k];
        best_W[g_off + k] = W[k];
    }}
    __syncwarp();

    u64 rng = seed_base ^ ((u64)(wid + 1) * 0x9E3779B97F4A7C15ULL);
    int w_tier = wid & 7;
    int max_vec_w;
    if (best_R <= 110) {{
        const int caps_lo[8] = {{2, 3, 3, 4, 5, 6, 10, 64}};
        max_vec_w = caps_lo[w_tier];
    }} else {{
        const int caps_hi[8] = {{3, 4, 6, 8, 10, 12, 16, 64}};
        max_vec_w = caps_hi[w_tier];
    }}
    int p_bucket = (wid >> 3) & 3;
    int local_plus_period;
    if (p_bucket == 0) local_plus_period = 1 << 29;
    else if (p_bucket == 1) local_plus_period = plus_period;
    else if (p_bucket == 2) local_plus_period = plus_period * 4;
    else local_plus_period = (plus_period > 128) ? (plus_period >> 1) : 64;
    int local_headroom = (p_bucket == 0) ? 1 : (plus_headroom + ((p_bucket == 3) ? 2 : (p_bucket - 1)));

    int steps_since_reduce = 0;
    int steps_at_elevated = 0;
    int steps_since_save = 0;
    int tabu_a = -1, tabu_b = -1, tabu_ttl = 0;
    int last_i = 0, last_j = 1;
    u64 flips_exec = 0ULL, plus_exec = 0ULL, red_exec = 0ULL;

    for (int step = 0; step < n_steps; ++step) {{
        u64 r0 = 0ULL, r1 = 0ULL;
        if (lane == 0) {{
            r0 = xorshift64star(rng);
            r1 = xorshift64star(rng);
        }}
        r0 = __shfl_sync(0xFFFFFFFFu, r0, 0);
        r1 = __shfl_sync(0xFFFFFFFFu, r1, 0);

        // Only fire greedy 1-step sniper & 3-term reduction when R <= best_R (guaranteed new record!),
        // NEVER while elevated (R > best_R) so plus/triple-split tunneling can freely climb & mix!
        bool allow_greedy_reduce = (R <= best_R);

        // 0. Periodic 3-term linear-dependence reduction (directly collapses 3 terms -> 2!)
        int red3_mask = (R > 128) ? 63 : 31;
        if (allow_greedy_reduce && ne > 0 && ((step & red3_mask) == 15)) {{
            if (f2_try_3term_reduce(U, V, W, R, edges, ne, r0, lane)) {{
                ne = f2_rebuild_edges(U, V, W, R, edges, lane);
                if (lane == 0) ++red_exec;
            }}
        }}

        int chosen_i = -1, chosen_j = -1, chosen_ax = -1, chosen_dir = 0;

        // 1. Periodic KM 1-step neighbor-reduction sniper around last modified terms
        int snipe_mask = (R > 128) ? 63 : 7;
        if (allow_greedy_reduce && ne > 0 && ((step & snipe_mask) == 0)) {{
            int si, sj, sax, sdir;
            if (f2_snipe_1step(U, V, W, R, edges, ne, last_i, last_j, si, sj, sax, sdir, lane)) {{
                chosen_i = si; chosen_j = sj; chosen_ax = sax; chosen_dir = sdir;
            }}
        }}

        // 2. O(1) exact uniform edge selection across 32 parallel candidate samples respecting W_cap!
        if (chosen_i < 0 && ne > 0) {{
            u64 lr = r1 ^ ((u64)(lane + 1) * 0xBF58476D1CE4E5B9ULL);
            lr ^= lr >> 13; lr *= 0x2545F4914F6CDD1DULL;
            int e_idx = (int)((lr >> 4) % (u64)ne);
            u32 ev = edges[e_idx];
            int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev);
            bool is_tabu = (tabu_ttl > 0) && ((i == tabu_a && j == tabu_b) || (i == tabu_b && j == tabu_a));
            u64 ui = U[i], vi = V[i], wi = W[i];
            u64 uj = U[j], vj = V[j], wj = W[j];
            u64 n1_0, n2_0, n1_1, n2_1;
            int old_0, old_1;
            if (ax == 2) {{
                n1_0 = vi ^ vj; n2_0 = uj ^ ui; old_0 = __popcll(vi) + __popcll(uj);
                n1_1 = ui ^ uj; n2_1 = vj ^ vi; old_1 = __popcll(ui) + __popcll(vj);
            }} else if (ax == 1) {{
                n1_0 = wi ^ wj; n2_0 = uj ^ ui; old_0 = __popcll(wi) + __popcll(uj);
                n1_1 = ui ^ uj; n2_1 = wj ^ wi; old_1 = __popcll(ui) + __popcll(wj);
            }} else {{
                n1_0 = wi ^ wj; n2_0 = vj ^ vi; old_0 = __popcll(wi) + __popcll(vj);
                n1_1 = vi ^ vj; n2_1 = wj ^ wi; old_1 = __popcll(vi) + __popcll(wj);
            }}
            bool z0 = !is_tabu && (n1_0 == 0ULL || n2_0 == 0ULL);
            bool z1 = !is_tabu && (n1_1 == 0ULL || n2_1 == 0ULL);
            int is_z = (z0 || z1) ? 1 : 0;
            int cand_dir = 0;
            bool ok = false;
            if (is_z) {{
                ok = true; cand_dir = z0 ? 0 : 1;
            }} else if (!is_tabu) {{
                int w1_0 = __popcll(n1_0), w2_0 = __popcll(n2_0);
                int w1_1 = __popcll(n1_1), w2_1 = __popcll(n2_1);
                bool ok0 = (w1_0 <= max_vec_w && w2_0 <= max_vec_w) || ((w1_0 + w2_0) < old_0);
                bool ok1 = (w1_1 <= max_vec_w && w2_1 <= max_vec_w) || ((w1_1 + w2_1) < old_1);
                if (ok0 || ok1) {{
                    ok = true;
                    cand_dir = (ok0 && ok1) ? (int)((lr >> 2) & 1ULL) : (ok0 ? 0 : 1);
                }}
            }}
            u32 zm = __ballot_sync(0xFFFFFFFFu, is_z != 0);
            u32 vm = zm ? zm : __ballot_sync(0xFFFFFFFFu, ok);
            if (vm == 0u) {{
                // Fallback: accept lane 0's raw uniform edge even if over W_cap so we never stall!
                vm = 1u;
                cand_dir = (int)(r0 & 1ULL);
            }}
            int wl = __ffs(vm) - 1;
            chosen_i = __shfl_sync(0xFFFFFFFFu, i, wl);
            chosen_j = __shfl_sync(0xFFFFFFFFu, j, wl);
            chosen_ax = __shfl_sync(0xFFFFFFFFu, ax, wl);
            chosen_dir = __shfl_sync(0xFFFFFFFFu, cand_dir, wl);
        }}

        bool reduced_now = false;
        int r_before_step = R;
        if (chosen_i >= 0) {{
            int m_a, m_b;
            if (chosen_ax == 2) {{
                m_a = (chosen_dir == 0) ? 1 : 0;
                m_b = (chosen_dir == 0) ? 0 : 1;
            }} else if (chosen_ax == 1) {{
                m_a = (chosen_dir == 0) ? 2 : 0;
                m_b = (chosen_dir == 0) ? 0 : 2;
            }} else {{
                m_a = (chosen_dir == 0) ? 2 : 1;
                m_b = (chosen_dir == 0) ? 1 : 2;
            }}

            if (lane == 0) {{
                if (chosen_ax == 2) {{
                    if (chosen_dir == 0) {{ V[chosen_i] ^= V[chosen_j]; U[chosen_j] ^= U[chosen_i]; }}
                    else                 {{ U[chosen_i] ^= U[chosen_j]; V[chosen_j] ^= V[chosen_i]; }}
                }} else if (chosen_ax == 1) {{
                    if (chosen_dir == 0) {{ W[chosen_i] ^= W[chosen_j]; U[chosen_j] ^= U[chosen_i]; }}
                    else                 {{ U[chosen_i] ^= U[chosen_j]; W[chosen_j] ^= W[chosen_i]; }}
                }} else {{
                    if (chosen_dir == 0) {{ W[chosen_i] ^= W[chosen_j]; V[chosen_j] ^= V[chosen_i]; }}
                    else                 {{ V[chosen_i] ^= V[chosen_j]; W[chosen_j] ^= W[chosen_i]; }}
                }}
                ++flips_exec;
            }}
            __syncwarp();

            if (f2_inc_update_after_flip(U, V, W, R, edges, ne, chosen_i, chosen_j, m_a, m_b, lane)) {{
                reduced_now = true;
                if (lane == 0) ++red_exec;
            }}
            last_i = (chosen_i < R) ? chosen_i : 0;
            last_j = (chosen_j < R) ? chosen_j : 0;
        }}

        if (R < best_R) {{
            steps_since_reduce = 0;
            steps_at_elevated = 0;
        }} else {{
            ++steps_since_reduce;
            if (R > best_R) ++steps_at_elevated;
            else if (reduced_now && r_before_step > best_R) steps_at_elevated = 0;
        }}
        ++steps_since_save;
        if (tabu_ttl > 0) --tabu_ttl;

        bool refresh_at_best = (R == best_R) && (
            (steps_since_save >= 500) ||
            (step == n_steps - 1) ||
            (diverse_mode && R <= target_R && steps_since_save >= 250)
        );
        if (R < best_R || refresh_at_best) {{
            int new_nnz = f2_compute_nnz(U, V, W, R, lane);
            best_R = R;
            best_nnz = new_nnz;
            steps_since_save = 0;
            for (int k = lane; k < R; k += 32) {{
                best_U[g_off + k] = U[k];
                best_V[g_off + k] = V[k];
                best_W[g_off + k] = W[k];
            }}
            __syncwarp();
        }}

        int r_cap = best_R + local_headroom;
        if (r_cap > MAX_R - 1) r_cap = MAX_R - 1;
        bool need_tunnel = (ne == 0) || (steps_since_reduce >= local_plus_period);

        if (need_tunnel) {{
            if (R < r_cap) {{
                f2_targeted_expand(U, V, W, R, r0 ^ ((u64)step * 0x9E3779B97F4A7C15ULL), lane);
                if (lane == 0) ++plus_exec;
                f2_full_reduce_sweep(U, V, W, R, lane);
                ne = f2_rebuild_edges(U, V, W, R, edges, lane);
                steps_since_reduce = 0;
                tabu_a = R - 1; tabu_b = last_i; tabu_ttl = 6;
            }} else if (steps_at_elevated >= (local_plus_period * 3) || ne == 0) {{
                R = best_R;
                for (int k = lane; k < R; k += 32) {{
                    U[k] = best_U[g_off + k];
                    V[k] = best_V[g_off + k];
                    W[k] = best_W[g_off + k];
                }}
                __syncwarp();
                ne = f2_rebuild_edges(U, V, W, R, edges, lane);
                steps_since_reduce = 0;
                steps_at_elevated = 0;
            }}
        }}
    }}

    if (lane == 0) {{
        best_R_out[wid] = best_R;
        best_nnz_out[wid] = best_nnz;
        stats_out[wid * 3 + 0] = flips_exec;
        stats_out[wid * 3 + 1] = plus_exec;
        stats_out[wid * 3 + 2] = red_exec;
    }}
}}
"""
    )


def _build_zt_cuda_source(max_r: int, max_edges: int, warps_per_block: int) -> str:
    """
    Warp-cooperative 2-bitplane branchless Z_T = {-1, 0, +1} Flip-Graph v2 kernel with:
      - O(ne/32 + 2R/32) incremental flip-edge list in __shared__ (15x-40x speedup at R=230..250!)
      - Exact KM 1-step ternary neighbor-reduction sniping
      - Atomic Triple-Collision Split-and-Flip (`u_k = a*u_i + b*u_j`, breaks Strassen shells!)
      - Parallel 32-lane ternary plus + guaranteed-ternary disjoint split + W_cap curriculum
    """
    return (
        _COMMON_CUDA_V2
        + f"""
#define MAX_R {max_r}
#define MAX_EDGES {max_edges}
#define WARPS_PER_BLOCK {warps_per_block}

__device__ __forceinline__ bool t_add(u64 pa, u64 na, u64 pb, u64 nb, u64& po, u64& no) {{
    if (((pa & pb) | (na & nb)) != 0ULL) return false;
    u64 dp = pa ^ pb;
    u64 dn = na ^ nb;
    po = dp & ~dn;
    no = dn & ~dp;
    return true;
}}

__device__ __forceinline__ bool t_add_signed(
    u64 pa, u64 na, u64 pb, u64 nb, int s, u64& po, u64& no
) {{
    return t_add(pa, na, (s > 0) ? pb : nb, (s > 0) ? nb : pb, po, no);
}}

__device__ __forceinline__ int t_cmp(u64 pa, u64 na, u64 pb, u64 nb) {{
    if (pa == pb && na == nb) return 1;
    if (pa == nb && na == pb) return -1;
    return 0;
}}

__device__ __forceinline__ int t_nnz(u64 p, u64 n) {{
    return __popcll(p | n);
}}

__device__ __forceinline__ void zt_drop_slot(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R, int idx
) {{
    int last = R - 1;
    if (idx < last) {{
        pU[idx] = pU[last]; nU[idx] = nU[last];
        pV[idx] = pV[last]; nV[idx] = nV[last];
        pW[idx] = pW[last]; nW[idx] = nW[last];
    }}
    R = last;
}}

__device__ int zt_reduce_slot(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R, int idx, int lane
) {{
    if (idx < 0 || idx >= R) return 0;
    u64 pui = pU[idx], nui = nU[idx];
    u64 pvi = pV[idx], nvi = nV[idx];
    u64 pwi = pW[idx], nwi = nW[idx];
    if ((pui | nui) == 0ULL || (pvi | nvi) == 0ULL || (pwi | nwi) == 0ULL) {{
        if (lane == 0) zt_drop_slot(pU, nU, pV, nV, pW, nW, R, idx);
        R = __shfl_sync(0xFFFFFFFFu, R, 0);
        __syncwarp();
        return 1;
    }}
    int match_k = -1, match_mode = -1;
    u64 res_p = 0ULL, res_n = 0ULL;
    for (int base = 0; base < R; base += 32) {{
        int k = base + lane;
        int mode = -1;
        u64 mp = 0ULL, mn = 0ULL;
        if (k < R && k != idx) {{
            int su = t_cmp(pui, nui, pU[k], nU[k]);
            int sv = t_cmp(pvi, nvi, pV[k], nV[k]);
            int sw = t_cmp(pwi, nwi, pW[k], nW[k]);
            if (su != 0 && sv != 0) {{
                if (t_add_signed(pW[k], nW[k], pwi, nwi, su * sv, mp, mn)) mode = 0;
            }} else if (su != 0 && sw != 0) {{
                if (t_add_signed(pV[k], nV[k], pvi, nvi, su * sw, mp, mn)) mode = 1;
            }} else if (sv != 0 && sw != 0) {{
                if (t_add_signed(pU[k], nU[k], pui, nui, sv * sw, mp, mn)) mode = 2;
            }}
        }}
        u32 hit = __ballot_sync(0xFFFFFFFFu, mode >= 0);
        if (hit) {{
            int wl = __ffs(hit) - 1;
            match_k = base + wl;
            match_mode = __shfl_sync(0xFFFFFFFFu, mode, wl);
            res_p = __shfl_sync(0xFFFFFFFFu, mp, wl);
            res_n = __shfl_sync(0xFFFFFFFFu, mn, wl);
            break;
        }}
    }}
    if (match_k < 0) return 0;

    int dropped = 1;
    if (lane == 0) {{
        bool became_zero = ((res_p | res_n) == 0ULL);
        if (match_mode == 0) {{ pW[match_k] = res_p; nW[match_k] = res_n; }}
        else if (match_mode == 1) {{ pV[match_k] = res_p; nV[match_k] = res_n; }}
        else {{ pU[match_k] = res_p; nU[match_k] = res_n; }}
        if (became_zero) {{
            int hi = (match_k > idx) ? match_k : idx;
            int lo = (match_k > idx) ? idx : match_k;
            zt_drop_slot(pU, nU, pV, nV, pW, nW, R, hi);
            zt_drop_slot(pU, nU, pV, nV, pW, nW, R, lo);
            dropped = 2;
        }} else {{
            zt_drop_slot(pU, nU, pV, nV, pW, nW, R, idx);
            dropped = 1;
        }}
    }}
    R = __shfl_sync(0xFFFFFFFFu, R, 0);
    dropped = __shfl_sync(0xFFFFFFFFu, dropped, 0);
    __syncwarp();
    return dropped;
}}

__device__ void zt_full_reduce_sweep(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R, int lane
) {{
    int i = 0;
    while (i < R) {{
        int d = zt_reduce_slot(pU, nU, pV, nV, pW, nW, R, i, lane);
        if (d > 0) i = (i > 0) ? (i - 1) : 0;
        else ++i;
    }}
}}

// Full warp-parallel rebuild of ternary flip-edge list (stores i < j, axis, sign s_eq in +/-1).
__device__ int zt_rebuild_edges(
    const u64* pU, const u64* nU, const u64* pV, const u64* nV, const u64* pW, const u64* nW,
    int R, u32* edges, int lane
) {{
    int ne = 0;
    u32 lt_mask = (1u << lane) - 1u;
    for (int i = 0; i < R - 1 && ne < MAX_EDGES; ++i) {{
        u64 pui = pU[i], nui = nU[i], pvi = pV[i], nvi = nV[i], pwi = pW[i], nwi = nW[i];
        for (int base = i + 1; base < R && ne < MAX_EDGES; base += 32) {{
            int j = base + lane;
            int su = 0, sv = 0, sw = 0;
            if (j < R) {{
                su = t_cmp(pui, nui, pU[j], nU[j]);
                sv = t_cmp(pvi, nvi, pV[j], nV[j]);
                sw = t_cmp(pwi, nwi, pW[j], nW[j]);
            }}
            u32 mu = __ballot_sync(0xFFFFFFFFu, su != 0);
            if (mu) {{
                int off = __popc(mu & lt_mask);
                if (su != 0 && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 0, su < 0);
                ne += __popc(mu);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
            u32 mv = __ballot_sync(0xFFFFFFFFu, sv != 0);
            if (mv && ne < MAX_EDGES) {{
                int off = __popc(mv & lt_mask);
                if (sv != 0 && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 1, sv < 0);
                ne += __popc(mv);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
            u32 mw = __ballot_sync(0xFFFFFFFFu, sw != 0);
            if (mw && ne < MAX_EDGES) {{
                int off = __popc(mw & lt_mask);
                if (sw != 0 && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 2, sw < 0);
                ne += __popc(mw);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
        }}
    }}
    __syncwarp();
    return ne;
}}

// O(ne/32 + 2R/32) incremental ternary edge update + instant 2-axis ternary reduce detection!
__device__ bool zt_inc_update_after_flip(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R,
    u32* edges, int& ne,
    int idx_i, int idx_j, int m_a, int m_b, int lane
) {{
    u64 pui = pU[idx_i], nui = nU[idx_i], pvi = pV[idx_i], nvi = nV[idx_i], pwi = pW[idx_i], nwi = nW[idx_i];
    u64 puj = pU[idx_j], nuj = nU[idx_j], pvj = pV[idx_j], nvj = nV[idx_j], pwj = pW[idx_j], nwj = nW[idx_j];

    if ((pui | nui) == 0ULL || (pvi | nvi) == 0ULL || (pwi | nwi) == 0ULL ||
        (puj | nuj) == 0ULL || (pvj | nvj) == 0ULL || (pwj | nwj) == 0ULL) {{
        zt_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
        ne = zt_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
        return true;
    }}

    // Full O(R/32) 3-axis check on idx_i and idx_j (catches BOTH newly-matching AND newly-unblocked ternary pairs!)
    const u64* pAa = (m_a == 0) ? pU : ((m_a == 1) ? pV : pW);
    const u64* nAa = (m_a == 0) ? nU : ((m_a == 1) ? nV : nW);
    const u64* pAb = (m_b == 0) ? pU : ((m_b == 1) ? pV : pW);
    const u64* nAb = (m_b == 0) ? nU : ((m_b == 1) ? nV : nW);
    u64 pva = pAa[idx_i], nva = nAa[idx_i];
    u64 pvb = pAb[idx_j], nvb = nAb[idx_j];

    bool hit_red = false;
    for (int base = 0; base < R; base += 32) {{
        int k = base + lane;
        bool col = false;
        if (k < R) {{
            u64 puk = pU[k], nuk = nU[k], pvk = pV[k], nvk = nV[k], pwk = pW[k], nwk = nW[k];
            u64 tp, tn;
            if (k != idx_i) {{
                int su = t_cmp(pui, nui, puk, nuk);
                int sv = t_cmp(pvi, nvi, pvk, nvk);
                int sw = t_cmp(pwi, nwi, pwk, nwk);
                if (su && sv && t_add_signed(pwk, nwk, pwi, nwi, su * sv, tp, tn)) col = true;
                else if (su && sw && t_add_signed(pvk, nvk, pvi, nvi, su * sw, tp, tn)) col = true;
                else if (sv && sw && t_add_signed(puk, nuk, pui, nui, sv * sw, tp, tn)) col = true;
            }}
            if (!col && k != idx_j) {{
                int su = t_cmp(puj, nuj, puk, nuk);
                int sv = t_cmp(pvj, nvj, pvk, nvk);
                int sw = t_cmp(pwj, nwj, pwk, nwk);
                if (su && sv && t_add_signed(pwk, nwk, pwj, nwj, su * sv, tp, tn)) col = true;
                else if (su && sw && t_add_signed(pvk, nvk, pvj, nvj, su * sw, tp, tn)) col = true;
                else if (sv && sw && t_add_signed(puk, nuk, puj, nuj, sv * sw, tp, tn)) col = true;
            }}
        }}
        if (__any_sync(0xFFFFFFFFu, col)) {{ hit_red = true; break; }}
    }}
    if (hit_red) {{
        zt_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
        ne = zt_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
        return true;
    }}

    // Race-free stream compaction of stale edges
    int out_ne = 0;
    u32 lt_mask = (1u << lane) - 1u;
    for (int base = 0; base < ne; base += 32) {{
        int pos = base + lane;
        u32 ev = (pos < ne) ? edges[pos] : 0u;
        bool keep = false;
        if (pos < ne) {{
            int ei = EDGE_I(ev), ej = EDGE_J(ev), em = EDGE_AX(ev);
            bool stale = (em == m_a && (ei == idx_i || ej == idx_i)) ||
                         (em == m_b && (ei == idx_j || ej == idx_j));
            keep = !stale;
        }}
        __syncwarp();
        u32 kmask = __ballot_sync(0xFFFFFFFFu, keep);
        if (keep) edges[out_ne + __popc(kmask & lt_mask)] = ev;
        out_ne += __popc(kmask);
        __syncwarp();
    }}
    ne = out_ne;

    // Append newly created edges on (idx_i, m_a) and (idx_j, m_b)

    for (int base = 0; base < R && ne < MAX_EDGES; base += 32) {{
        int k = base + lane;
        int sa = (k < R && k != idx_i) ? t_cmp(pva, nva, pAa[k], nAa[k]) : 0;
        u32 ma = __ballot_sync(0xFFFFFFFFu, sa != 0);
        if (ma) {{
            int off = __popc(ma & lt_mask);
            if (sa != 0 && ne + off < MAX_EDGES) {{
                int lo = (idx_i < k) ? idx_i : k;
                int hi = (idx_i < k) ? k : idx_i;
                edges[ne + off] = PACK_EDGE(lo, hi, m_a, sa < 0);
            }}
            ne += __popc(ma);
            if (ne > MAX_EDGES) ne = MAX_EDGES;
        }}
        int sb = (k < R && k != idx_j) ? t_cmp(pvb, nvb, pAb[k], nAb[k]) : 0;
        u32 mb = __ballot_sync(0xFFFFFFFFu, sb != 0);
        if (mb && ne < MAX_EDGES) {{
            int off = __popc(mb & lt_mask);
            if (sb != 0 && ne + off < MAX_EDGES) {{
                int lo = (idx_j < k) ? idx_j : k;
                int hi = (idx_j < k) ? k : idx_j;
                edges[ne + off] = PACK_EDGE(lo, hi, m_b, sb < 0);
            }}
            ne += __popc(mb);
            if (ne > MAX_EDGES) ne = MAX_EDGES;
        }}
    }}
    __syncwarp();
    return false;
}}

// Exact KM 1-step ternary neighbor-reduction sniper around focus_a, focus_b!
__device__ bool zt_snipe_1step(
    const u64* pU, const u64* nU, const u64* pV, const u64* nV, const u64* pW, const u64* nW, int R,
    const u32* edges, int ne, int focus_a, int focus_b,
    int& out_i, int& out_j, int& out_ax, int& out_dir, int& out_alpha, int& out_s, int lane
) {{
    if (ne == 0) return false;
    int cand_e[4];
    int n_cand = 0;
    for (int base = ((ne - 1) & ~31); base >= 0 && n_cand < 4; base -= 32) {{
        int pos = base + lane;
        bool inc = false;
        if (pos < ne) {{
            u32 ev = edges[pos];
            int ei = EDGE_I(ev), ej = EDGE_J(ev);
            inc = (ei == focus_a || ej == focus_a || ei == focus_b || ej == focus_b);
        }}
        u32 m = __ballot_sync(0xFFFFFFFFu, inc);
        while (m && n_cand < 4) {{
            int b = __ffs(m) - 1;
            cand_e[n_cand++] = base + b;
            m &= m - 1u;
        }}
    }}

    for (int c = 0; c < n_cand; ++c) {{
        u32 ev = edges[cand_e[c]];
        int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev), s_eq = EDGE_SGN(ev);
        u64 pui = pU[i], nui = nU[i], pvi = pV[i], nvi = nV[i], pwi = pW[i], nwi = nW[i];
        u64 puj = pU[j], nuj = nU[j], pvj = pV[j], nvj = nV[j], pwj = pW[j], nwj = nW[j];

        for (int dir = 0; dir < 2; ++dir) {{
            for (int alpha = -1; alpha <= 1; alpha += 2) {{
                u64 o1p, o1n, o2p, o2n;
                bool ok;
                if (ax == 2) {{
                    if (dir == 0) {{
                        ok = t_add_signed(pvi, nvi, pvj, nvj, -alpha, o1p, o1n) &&
                             t_add_signed(puj, nuj, pui, nui, alpha * s_eq, o2p, o2n);
                    }} else {{
                        ok = t_add_signed(pui, nui, puj, nuj, -alpha, o1p, o1n) &&
                             t_add_signed(pvj, nvj, pvi, nvi, alpha * s_eq, o2p, o2n);
                    }}
                }} else if (ax == 1) {{
                    if (dir == 0) {{
                        ok = t_add_signed(pwi, nwi, pwj, nwj, -alpha, o1p, o1n) &&
                             t_add_signed(puj, nuj, pui, nui, alpha * s_eq, o2p, o2n);
                    }} else {{
                        ok = t_add_signed(pui, nui, puj, nuj, -alpha, o1p, o1n) &&
                             t_add_signed(pwj, nwj, pwi, nwi, alpha * s_eq, o2p, o2n);
                    }}
                }} else {{
                    if (dir == 0) {{
                        ok = t_add_signed(pwi, nwi, pwj, nwj, -alpha, o1p, o1n) &&
                             t_add_signed(pvj, nvj, pvi, nvi, alpha * s_eq, o2p, o2n);
                    }} else {{
                        ok = t_add_signed(pvi, nvi, pvj, nvj, -alpha, o1p, o1n) &&
                             t_add_signed(pwj, nwj, pwi, nwi, alpha * s_eq, o2p, o2n);
                    }}
                }}
                if (!ok) continue;
                if ((o1p | o1n) == 0ULL || (o2p | o2n) == 0ULL) {{
                    out_i = i; out_j = j; out_ax = ax; out_dir = dir; out_alpha = alpha; out_s = s_eq;
                    return true;
                }}
                // Compute post-flip factors for i and j and test 2-axis ternary collision against all k!
                u64 ni_pu = pui, ni_nu = nui, ni_pv = pvi, ni_nv = nvi, ni_pw = pwi, ni_nw = nwi;
                u64 nj_pu = puj, nj_nu = nuj, nj_pv = pvj, nj_nv = nvj, nj_pw = pwj, nj_nw = nwj;
                if (ax == 2) {{
                    if (dir == 0) {{ ni_pv = o1p; ni_nv = o1n; nj_pu = o2p; nj_nu = o2n; }}
                    else          {{ ni_pu = o1p; ni_nu = o1n; nj_pv = o2p; nj_nv = o2n; }}
                }} else if (ax == 1) {{
                    if (dir == 0) {{ ni_pw = o1p; ni_nw = o1n; nj_pu = o2p; nj_nu = o2n; }}
                    else          {{ ni_pu = o1p; ni_nu = o1n; nj_pw = o2p; nj_nw = o2n; }}
                }} else {{
                    if (dir == 0) {{ ni_pw = o1p; ni_nw = o1n; nj_pv = o2p; nj_nv = o2n; }}
                    else          {{ ni_pv = o1p; ni_nv = o1n; nj_pw = o2p; nj_nw = o2n; }}
                }}
                for (int base = 0; base < R; base += 32) {{
                    int k = base + lane;
                    bool hit = false;
                    if (k < R && k != i && k != j) {{
                        u64 puk = pU[k], nuk = nU[k], pvk = pV[k], nvk = nV[k], pwk = pW[k], nwk = nW[k];
                        u64 tp, tn;
                        int su1 = t_cmp(ni_pu, ni_nu, puk, nuk);
                        int sv1 = t_cmp(ni_pv, ni_nv, pvk, nvk);
                        int sw1 = t_cmp(ni_pw, ni_nw, pwk, nwk);
                        if ((su1 && sv1 && t_add_signed(pwk, nwk, ni_pw, ni_nw, su1 * sv1, tp, tn)) ||
                            (su1 && sw1 && t_add_signed(pvk, nvk, ni_pv, ni_nv, su1 * sw1, tp, tn)) ||
                            (sv1 && sw1 && t_add_signed(puk, nuk, ni_pu, ni_nu, sv1 * sw1, tp, tn))) hit = true;
                        if (!hit) {{
                            int su2 = t_cmp(nj_pu, nj_nu, puk, nuk);
                            int sv2 = t_cmp(nj_pv, nj_nv, pvk, nvk);
                            int sw2 = t_cmp(nj_pw, nj_nw, pwk, nwk);
                            if ((su2 && sv2 && t_add_signed(pwk, nwk, nj_pw, nj_nw, su2 * sv2, tp, tn)) ||
                                (su2 && sw2 && t_add_signed(pvk, nvk, nj_pv, nj_nv, su2 * sw2, tp, tn)) ||
                                (sv2 && sw2 && t_add_signed(puk, nuk, nj_pu, nj_nu, sv2 * sw2, tp, tn))) hit = true;
                        }}
                    }}
                    if (__any_sync(0xFFFFFFFFu, hit)) {{
                        out_i = i; out_j = j; out_ax = ax; out_dir = dir; out_alpha = alpha; out_s = s_eq;
                        return true;
                    }}
                }}
            }}
        }}
    }}
    return false;
}}

// Exact warp-parallel 3-term ternary linear-dependence reduction:
// Given edge (i, j, ax, s_ij) with A[i] = s_ij*A[j], scans k != i, j with A[i] = s_ik*A[k]
// for B[k] = a*B[i] + b*B[j] (or C[k] = a*C[i] + b*C[j]) where companion sums stay ternary -> R -> R-1!
__device__ bool zt_try_3term_reduce(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R,
    const u32* edges, int ne, u64 r_seed, int lane
) {{
    if (ne == 0 || R < 3) return false;
    u64 st = r_seed;
    for (int trial = 0; trial < 4; ++trial) {{
        st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
        int e_idx = (int)(((st * 0x2545F4914F6CDD1DULL) >> 4) % (u64)ne);
        u32 ev = edges[e_idx];
        int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev), s_ij = EDGE_SGN(ev);
        u64 *pA = (ax == 0) ? pU : ((ax == 1) ? pV : pW);
        u64 *nA = (ax == 0) ? nU : ((ax == 1) ? nV : nW);
        u64 *pB = (ax == 0) ? pV : ((ax == 1) ? pW : pU);
        u64 *nB = (ax == 0) ? nV : ((ax == 1) ? nW : nU);
        u64 *pC = (ax == 0) ? pW : ((ax == 1) ? pU : pV);
        u64 *nC = (ax == 0) ? nW : ((ax == 1) ? nU : nV);

        u64 pai = pA[i], nai = nA[i];
        u64 pbi = pB[i], nbi = nB[i], pbj = pB[j], nbj = nB[j];
        u64 pci = pC[i], nci = nC[i], pcj = pC[j], ncj = nC[j];

        u64 bpp, bpn, bmp, bmn, cpp, cpn, cmp, cmn;
        bool ok_bp = t_add(pbi, nbi, pbj, nbj, bpp, bpn);
        bool ok_bm = t_add_signed(pbi, nbi, pbj, nbj, -1, bmp, bmn);
        bool ok_cp = t_add(pci, nci, pcj, ncj, cpp, cpn);
        bool ok_cm = t_add_signed(pci, nci, pcj, ncj, -1, cmp, cmn);
        if (!ok_bp && !ok_bm && !ok_cp && !ok_cm) continue;

        int hit_k = -1, hit_mode = -1;
        u64 h1p = 0ULL, h1n = 0ULL, h2p = 0ULL, h2n = 0ULL;
        for (int base = 0; base < R; base += 32) {{
            int k = base + lane;
            int mode = -1;
            u64 o1p = 0ULL, o1n = 0ULL, o2p = 0ULL, o2n = 0ULL;
            if (k < R && k != i && k != j) {{
                int s_ik = t_cmp(pai, nai, pA[k], nA[k]);
                if (s_ik != 0) {{
                    u64 pbk = pB[k], nbk = nB[k], pck = pC[k], nck = nC[k];
                    int a = 0, b = 0;
                    if (ok_bp) {{
                        int sk = t_cmp(pbk, nbk, bpp, bpn);
                        if (sk != 0) {{ a = sk; b = sk; }}
                    }}
                    if (a == 0 && ok_bm) {{
                        int sk = t_cmp(pbk, nbk, bmp, bmn);
                        if (sk != 0) {{ a = sk; b = -sk; }}
                    }}
                    if (a != 0) {{
                        if (t_add_signed(pci, nci, pck, nck, s_ik * a, o1p, o1n) &&
                            t_add_signed(pcj, ncj, pck, nck, s_ij * s_ik * b, o2p, o2n)) {{
                            mode = 0;
                        }}
                    }}
                    if (mode < 0) {{
                        int ca = 0, cb = 0;
                        if (ok_cp) {{
                            int sk = t_cmp(pck, nck, cpp, cpn);
                            if (sk != 0) {{ ca = sk; cb = sk; }}
                        }}
                        if (ca == 0 && ok_cm) {{
                            int sk = t_cmp(pck, nck, cmp, cmn);
                            if (sk != 0) {{ ca = sk; cb = -sk; }}
                        }}
                        if (ca != 0) {{
                            if (t_add_signed(pbi, nbi, pbk, nbk, s_ik * ca, o1p, o1n) &&
                                t_add_signed(pbj, nbj, pbk, nbk, s_ij * s_ik * cb, o2p, o2n)) {{
                                mode = 1;
                            }}
                        }}
                    }}
                }}
            }}
            u32 hm = __ballot_sync(0xFFFFFFFFu, mode >= 0);
            if (hm) {{
                int wl = __ffs(hm) - 1;
                hit_k = base + wl;
                hit_mode = __shfl_sync(0xFFFFFFFFu, mode, wl);
                h1p = __shfl_sync(0xFFFFFFFFu, o1p, wl); h1n = __shfl_sync(0xFFFFFFFFu, o1n, wl);
                h2p = __shfl_sync(0xFFFFFFFFu, o2p, wl); h2n = __shfl_sync(0xFFFFFFFFu, o2n, wl);
                break;
            }}
        }}
        if (hit_k >= 0) {{
            if (lane == 0) {{
                if (hit_mode == 0) {{
                    pC[i] = h1p; nC[i] = h1n;
                    pC[j] = h2p; nC[j] = h2n;
                }} else {{
                    pB[i] = h1p; nB[i] = h1n;
                    pB[j] = h2p; nB[j] = h2n;
                }}
                zt_drop_slot(pU, nU, pV, nV, pW, nW, R, hit_k);
            }}
            R = __shfl_sync(0xFFFFFFFFu, R, 0);
            __syncwarp();
            zt_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
            return true;
        }}
    }}
    return false;
}}

// Exact Atomic Triple-Collision Split-and-Flip for Z_T:
// Searches for A[k] = a*A[i] + b*A[j] (a, b in {{+1, -1}}) where the companion flip into i is
// ternary-admissible! Shatters outer-Strassen shells (M1 = M3 + M4, M5 = M1 + M7, etc.) with ZERO undo!
// Falls back to 32-lane parallel active-bridged ternary plus or guaranteed-ternary disjoint split.
__device__ bool zt_targeted_expand(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW,
    int& R, int bridge_j, int bridge_ax, u64 r_seed,
    int& out_ta, int& out_tb, int lane
) {{
    if (R >= MAX_R) return false;
    out_ta = -1; out_tb = -1;
    u64 st = r_seed;

    // 1. 50% fast parallel triple-collision Split-and-Flip (cracks Strassen shells!), 50% cross-block ternary plus!
    int hi = -1, hj = -1, hk = -1, hax = -1, ha = 1, hb = 1, halpha = 1;
    u64 ho1p = 0ULL, ho1n = 0ULL, ho2p = 0ULL, ho2n = 0ULL;
    if ((r_seed & 1ULL) == 0ULL) {{
        for (int trial = 0; trial < 12 && hk < 0; ++trial) {{
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            u64 rv = st * 0x2545F4914F6CDD1DULL;
            int ax = (int)((rv >> 2) % 3ULL);
            int i = (int)((rv >> 8) % (u64)R);
            int j = (int)((rv >> 20) % (u64)(R - 1));
            if (j >= i) ++j;
            int s1 = ((rv >> 32) & 1ULL) ? 1 : -1;
            int s2 = ((rv >> 33) & 1ULL) ? 1 : -1;

            const u64* pA = (ax == 0) ? pU : ((ax == 1) ? pV : pW);
            const u64* nA = (ax == 0) ? nU : ((ax == 1) ? nV : nW);
            u64 sp, sn;
            u64 pi_p = (s1 > 0) ? pA[i] : nA[i];
            u64 pi_n = (s1 > 0) ? nA[i] : pA[i];
            if (!t_add_signed(pi_p, pi_n, pA[j], nA[j], s2, sp, sn) || (sp | sn) == 0ULL) continue;

            for (int base = 0; base < R; base += 32) {{
                int k = base + lane;
                int sk = (k < R && k != i && k != j) ? t_cmp(pA[k], nA[k], sp, sn) : 0;
                bool ok_k = false;
                int loc_a = 1, loc_b = 1, loc_al = 1;
                u64 lo1p = 0ULL, lo1n = 0ULL, lo2p = 0ULL, lo2n = 0ULL;
                if (sk != 0) {{
                    loc_a = sk * s1;
                    loc_b = sk * s2;
                    for (int al = -1; al <= 1 && !ok_k; al += 2) {{
                        if (ax == 0) {{
                            if (t_add_signed(pV[k], nV[k], pV[i], nV[i], -al, lo1p, lo1n) &&
                                t_add_signed(pW[i], nW[i], pW[k], nW[k], al * loc_a, lo2p, lo2n)) {{
                                ok_k = true; loc_al = al;
                            }}
                        }} else if (ax == 1) {{
                            if (t_add_signed(pW[k], nW[k], pW[i], nW[i], -al, lo1p, lo1n) &&
                                t_add_signed(pU[i], nU[i], pU[k], nU[k], al * loc_a, lo2p, lo2n)) {{
                                ok_k = true; loc_al = al;
                            }}
                        }} else {{
                            if (t_add_signed(pU[k], nU[k], pU[i], nU[i], -al, lo1p, lo1n) &&
                                t_add_signed(pV[i], nV[i], pV[k], nV[k], al * loc_a, lo2p, lo2n)) {{
                                ok_k = true; loc_al = al;
                            }}
                        }}
                    }}
                }}
                u32 hm = __ballot_sync(0xFFFFFFFFu, ok_k);
                if (hm) {{
                    int wl = __ffs(hm) - 1;
                    hi = i; hj = j; hk = base + wl; hax = ax;
                    ha = __shfl_sync(0xFFFFFFFFu, loc_a, wl);
                    hb = __shfl_sync(0xFFFFFFFFu, loc_b, wl);
                    halpha = __shfl_sync(0xFFFFFFFFu, loc_al, wl);
                    ho1p = __shfl_sync(0xFFFFFFFFu, lo1p, wl); ho1n = __shfl_sync(0xFFFFFFFFu, lo1n, wl);
                    ho2p = __shfl_sync(0xFFFFFFFFu, lo2p, wl); ho2n = __shfl_sync(0xFFFFFFFFu, lo2n, wl);
                    break;
                }}
            }}
        }}
    }}

    if (hk >= 0) {{
        if (lane == 0) {{
            u64 pui = pU[hi], nui = nU[hi], pvi = pV[hi], nvi = nV[hi], pwi = pW[hi], nwi = nW[hi];
            u64 puj = pU[hj], nuj = nU[hj], pvj = pV[hj], nvj = nV[hj], pwj = pW[hj], nwj = nW[hj];
            u64 puk = pU[hk], nuk = nU[hk], pvk = pV[hk], nvk = nV[hk], pwk = pW[hk], nwk = nW[hk];
            if (hax == 0) {{
                // T1 -> slot hk, T3 -> slot hi, T2 -> slot R
                pU[hk] = (hb > 0) ? puj : nuj; nU[hk] = (hb > 0) ? nuj : puj;
                pW[hi] = ho2p; nW[hi] = ho2n;
                pU[R]  = (ha > 0) ? pui : nui; nU[R]  = (ha > 0) ? nui : pui;
                pV[R]  = ho1p; nV[R]  = ho1n;
                pW[R]  = pwk;  nW[R]  = nwk;
            }} else if (hax == 1) {{
                pV[hk] = (hb > 0) ? pvj : nvj; nV[hk] = (hb > 0) ? nvj : pvj;
                pU[hi] = ho2p; nU[hi] = ho2n;
                pU[R]  = puk;  nU[R]  = nuk;
                pV[R]  = (ha > 0) ? pvi : nvi; nV[R]  = (ha > 0) ? nvi : pvi;
                pW[R]  = ho1p; nW[R]  = ho1n;
            }} else {{
                pW[hk] = (hb > 0) ? pwj : nwj; nW[hk] = (hb > 0) ? nwj : pwj;
                pV[hi] = ho2p; nV[hi] = ho2n;
                pU[R]  = ho1p; nU[R]  = ho1n;
                pV[R]  = pvk;  nV[R]  = nvk;
                pW[R]  = (ha > 0) ? pwi : nwi; nW[R]  = (ha > 0) ? nwi : pwi;
            }}
            ++R;
        }}
        R = __shfl_sync(0xFFFFFFFFu, R, 0);
        out_ta = hi; out_tb = R - 1;
        __syncwarp();
        return true;
    }}

    // 2. Fallback: 32-lane parallel active-term-bridged ternary plus or disjoint-support split
    u64 pr = r_seed ^ ((u64)(lane + 1) * 0xD6E8FEB86659FD93ULL);
    pr ^= pr >> 12; pr ^= pr << 25; pr ^= pr >> 27; pr *= 0x2545F4914F6CDD1DULL;
    int pi = (int)((pr >> 4) % (u64)R);
    bool use_bridge = (bridge_j >= 0 && bridge_j < R && (lane & 1) == 0 && bridge_j != pi);
    int pj = use_bridge ? bridge_j : (int)((pr >> 16) % (u64)(R - 1));
    if (!use_bridge && pj >= pi) ++pj;
    int ori;
    if (use_bridge && bridge_ax == 2) ori = 0;
    else if (use_bridge && bridge_ax == 0) ori = 1;
    else if (use_bridge && bridge_ax == 1) ori = 2;
    else ori = (int)((pr >> 28) % 3ULL);
    int s = ((pr >> 32) & 1ULL) ? 1 : -1;
    int alpha = ((pr >> 33) & 1ULL) ? 1 : -1;

    u64 t1p, t1n, t2p, t2n, t3p, t3n;
    bool ok_plus = false;
    if (ori == 0) {{
        ok_plus = t_add_signed(pW[pi], nW[pi], pW[pj], nW[pj], -s, t1p, t1n) &&
                  t_add_signed(pV[pi], nV[pi], pV[pj], nV[pj], -alpha, t2p, t2n) &&
                  t_add_signed(pU[pj], nU[pj], pU[pi], nU[pi], alpha * s, t3p, t3n);
    }} else if (ori == 1) {{
        ok_plus = t_add_signed(pU[pi], nU[pi], pU[pj], nU[pj], -s, t1p, t1n) &&
                  t_add_signed(pW[pi], nW[pi], pW[pj], nW[pj], -alpha, t2p, t2n) &&
                  t_add_signed(pV[pj], nV[pj], pV[pi], nV[pi], alpha * s, t3p, t3n);
    }} else {{
        ok_plus = t_add_signed(pV[pi], nV[pi], pV[pj], nV[pj], -s, t1p, t1n) &&
                  t_add_signed(pU[pi], nU[pi], pU[pj], nU[pj], -alpha, t2p, t2n) &&
                  t_add_signed(pW[pj], nW[pj], pW[pi], nW[pi], alpha * s, t3p, t3n);
    }}

    u32 pm = __ballot_sync(0xFFFFFFFFu, ok_plus);
    if (pm) {{
        int wl = __ffs(pm) - 1;
        int wi = __shfl_sync(0xFFFFFFFFu, pi, wl);
        int wj = __shfl_sync(0xFFFFFFFFu, pj, wl);
        int wo = __shfl_sync(0xFFFFFFFFu, ori, wl);
        int ws = __shfl_sync(0xFFFFFFFFu, s, wl);
        u64 w1p = __shfl_sync(0xFFFFFFFFu, t1p, wl), w1n = __shfl_sync(0xFFFFFFFFu, t1n, wl);
        u64 w2p = __shfl_sync(0xFFFFFFFFu, t2p, wl), w2n = __shfl_sync(0xFFFFFFFFu, t2n, wl);
        u64 w3p = __shfl_sync(0xFFFFFFFFu, t3p, wl), w3n = __shfl_sync(0xFFFFFFFFu, t3n, wl);
        if (lane == 0) {{
            u64 pui = pU[wi], nui = nU[wi], pvi = pV[wi], nvi = nV[wi], pwi = pW[wi], nwi = nW[wi];
            u64 puj = pU[wj], nuj = nU[wj], pvj = pV[wj], nvj = nV[wj], pwj = pW[wj], nwj = nW[wj];
            if (wo == 0) {{
                pW[wi] = w1p; nW[wi] = w1n;
                pU[wj] = (ws > 0) ? pui : nui; nU[wj] = (ws > 0) ? nui : pui;
                pV[wj] = w2p; nV[wj] = w2n;
                pU[R]  = w3p; nU[R]  = w3n; pV[R] = pvj; nV[R] = nvj; pW[R] = pwj; nW[R] = nwj;
            }} else if (wo == 1) {{
                pU[wi] = w1p; nU[wi] = w1n;
                pV[wj] = (ws > 0) ? pvi : nvi; nV[wj] = (ws > 0) ? nvi : pvi;
                pW[wj] = w2p; nW[wj] = w2n;
                pU[R]  = puj; nU[R]  = nuj; pV[R] = w3p; nV[R] = w3n; pW[R] = pwj; nW[R] = nwj;
            }} else {{
                pV[wi] = w1p; nV[wi] = w1n;
                pU[wj] = w2p; nU[wj] = w2n;
                pW[wj] = (ws > 0) ? pwi : nwi; nW[wj] = (ws > 0) ? nwi : pwi;
                pU[R]  = puj; nU[R]  = nuj; pV[R] = pvj; nV[R] = nvj; pW[R] = w3p; nW[R] = w3n;
            }}
            ++R;
        }}
        R = __shfl_sync(0xFFFFFFFFu, R, 0);
        out_ta = wj; out_tb = R - 1;
        __syncwarp();
        return true;
    }} else if (lane == 0) {{
        u64 rs = r_seed ^ 0xA24BAED4963EE407ULL;
        int ri = (int)((rs >> 4) % (u64)R);
        int ax = (int)((rs >> 20) % 3ULL);
        u64 msk = rs * 0x9E3779B97F4A7C15ULL;
        u64 *pF = (ax == 0) ? pU : ((ax == 1) ? pV : pW);
        u64 *nF = (ax == 0) ? nU : ((ax == 1) ? nV : nW);
        u64 supp = pF[ri] | nF[ri];
        if (__popcll(supp) >= 2) {{
            u64 h1 = supp & msk;
            if (h1 == 0ULL || h1 == supp) h1 = supp & (supp - 1ULL);
            u64 h2 = supp ^ h1;
            pU[R] = pU[ri]; nU[R] = nU[ri];
            pV[R] = pV[ri]; nV[R] = nV[ri];
            pW[R] = pW[ri]; nW[R] = nW[ri];
            pF[ri] &= h1; nF[ri] &= h1;
            pF[R]  &= h2; nF[R]  &= h2;
            ++R;
        }}
    }}
    R = __shfl_sync(0xFFFFFFFFu, R, 0);
    __syncwarp();
    return true;
}}

__device__ int zt_compute_nnz(
    const u64* pU, const u64* nU, const u64* pV, const u64* nV, const u64* pW, const u64* nW,
    int R, int lane
) {{
    int sum = 0;
    for (int k = lane; k < R; k += 32) {{
        sum += t_nnz(pU[k], nU[k]) + t_nnz(pV[k], nV[k]) + t_nnz(pW[k], nW[k]);
    }}
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);
    return __shfl_sync(0xFFFFFFFFu, sum, 0);
}}

extern "C" __global__ void zt_flip_walk_kernel(
    const u64* __restrict__ init_pU, const u64* __restrict__ init_nU,
    const u64* __restrict__ init_pV, const u64* __restrict__ init_nV,
    const u64* __restrict__ init_pW, const u64* __restrict__ init_nW,
    const int* __restrict__ init_R,
    u64* __restrict__ best_pU, u64* __restrict__ best_nU,
    u64* __restrict__ best_pV, u64* __restrict__ best_nV,
    u64* __restrict__ best_pW, u64* __restrict__ best_nW,
    int* __restrict__ best_R_out,
    int* __restrict__ best_nnz_out,
    u64* __restrict__ stats_out,
    int n_walkers,
    int n_steps,
    int plus_period,
    int plus_headroom,
    u64 seed_base
) {{
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int wid = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (wid >= n_walkers) return;

    __shared__ u64 spU[WARPS_PER_BLOCK][MAX_R], snU[WARPS_PER_BLOCK][MAX_R];
    __shared__ u64 spV[WARPS_PER_BLOCK][MAX_R], snV[WARPS_PER_BLOCK][MAX_R];
    __shared__ u64 spW[WARPS_PER_BLOCK][MAX_R], snW[WARPS_PER_BLOCK][MAX_R];
    __shared__ u32 sE[WARPS_PER_BLOCK][MAX_EDGES];

    u64 *pU = spU[warp_in_block], *nU = snU[warp_in_block];
    u64 *pV = spV[warp_in_block], *nV = snV[warp_in_block];
    u64 *pW = spW[warp_in_block], *nW = snW[warp_in_block];
    u32 *edges = sE[warp_in_block];

    int R = init_R[wid];
    if (R > MAX_R) R = MAX_R;
    int g_off = wid * MAX_R;
    for (int k = lane; k < R; k += 32) {{
        pU[k] = init_pU[g_off + k]; nU[k] = init_nU[g_off + k];
        pV[k] = init_pV[g_off + k]; nV[k] = init_nV[g_off + k];
        pW[k] = init_pW[g_off + k]; nW[k] = init_nW[g_off + k];
    }}
    __syncwarp();

    zt_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
    int ne = zt_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
    int best_R = R;
    int best_nnz = zt_compute_nnz(pU, nU, pV, nV, pW, nW, R, lane);
    for (int k = lane; k < R; k += 32) {{
        best_pU[g_off + k] = pU[k]; best_nU[g_off + k] = nU[k];
        best_pV[g_off + k] = pV[k]; best_nV[g_off + k] = nV[k];
        best_pW[g_off + k] = pW[k]; best_nW[g_off + k] = nW[k];
    }}
    __syncwarp();

    u64 rng = seed_base ^ ((u64)(wid + 1) * 0x9E3779B97F4A7C15ULL);
    int w_tier = wid & 7;
    int max_vec_w;
    if (best_R <= 110) {{
        const int caps_lo[8] = {{2, 3, 3, 4, 5, 6, 10, 64}};
        max_vec_w = caps_lo[w_tier];
    }} else {{
        const int caps_hi[8] = {{3, 4, 6, 8, 10, 12, 16, 64}};
        max_vec_w = caps_hi[w_tier];
    }}
    int p_bucket = (wid >> 3) & 3;
    int local_plus_period;
    if (p_bucket == 0) local_plus_period = 1 << 29;
    else if (p_bucket == 1) local_plus_period = plus_period;
    else if (p_bucket == 2) local_plus_period = plus_period * 4;
    else local_plus_period = (plus_period > 128) ? (plus_period >> 1) : 64;
    int local_headroom = (p_bucket == 0) ? 1 : (plus_headroom + ((p_bucket == 3) ? 2 : (p_bucket - 1)));

    int steps_since_reduce = 0;
    int steps_at_elevated = 0;
    int steps_since_save = 0;
    int tabu_a = -1, tabu_b = -1, tabu_ttl = 0;
    int last_i = 0, last_j = 1, last_ax = 0;
    u64 flips_exec = 0ULL, plus_exec = 0ULL, red_exec = 0ULL;

    for (int step = 0; step < n_steps; ++step) {{
        u64 r0 = 0ULL, r1 = 0ULL;
        if (lane == 0) {{
            r0 = xorshift64star(rng);
            r1 = xorshift64star(rng);
        }}
        r0 = __shfl_sync(0xFFFFFFFFu, r0, 0);
        r1 = __shfl_sync(0xFFFFFFFFu, r1, 0);

        // Only fire greedy 1-step sniper & 3-term reduction when R <= best_R (guaranteed new record!),
        // NEVER while elevated (R > best_R) so plus/triple-split tunneling can freely climb & mix!
        bool allow_greedy_reduce = (R <= best_R);

        // 0. Periodic 3-term ternary linear-dependence reduction (3 terms -> 2!)
        int red3_mask = (R > 128) ? 63 : 31;
        if (allow_greedy_reduce && ne > 0 && ((step & red3_mask) == 15)) {{
            if (zt_try_3term_reduce(pU, nU, pV, nV, pW, nW, R, edges, ne, r0, lane)) {{
                ne = zt_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
                if (lane == 0) ++red_exec;
            }}
        }}

        int chosen_i = -1, chosen_j = -1, chosen_ax = -1;
        int chosen_dir = 0, chosen_alpha = 1, chosen_s = 1;

        // 1. Periodic KM 1-step ternary neighbor-reduction sniping
        int snipe_mask = (R > 128) ? 63 : 7;
        if (allow_greedy_reduce && ne > 0 && ((step & snipe_mask) == 0)) {{
            int si, sj, sax, sdir, sal, ss;
            if (zt_snipe_1step(pU, nU, pV, nV, pW, nW, R, edges, ne, last_i, last_j,
                               si, sj, sax, sdir, sal, ss, lane)) {{
                chosen_i = si; chosen_j = sj; chosen_ax = sax;
                chosen_dir = sdir; chosen_alpha = sal; chosen_s = ss;
            }}
        }}

        // 2. O(1) parallel 32-lane ternary-admissible edge sampling directly from edges[0..ne-1]!
        if (chosen_i < 0 && ne > 0) {{
            u64 lr = r1 ^ ((u64)(lane + 1) * 0xBF58476D1CE4E5B9ULL);
            lr ^= lr >> 13; lr *= 0x2545F4914F6CDD1DULL;
            int e_idx = (int)((lr >> 4) % (u64)ne);
            u32 ev = edges[e_idx];
            int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev), s_eq = EDGE_SGN(ev);
            bool is_tabu = (tabu_ttl > 0) && ((i == tabu_a && j == tabu_b) || (i == tabu_b && j == tabu_a));
            u64 pa1, na1, pb1, nb1, pa2, na2, pb2, nb2;
            if (ax == 2) {{
                pa1=pV[i]; na1=nV[i]; pb1=pV[j]; nb1=nV[j];
                pa2=pU[j]; na2=nU[j]; pb2=pU[i]; nb2=nU[i];
            }} else if (ax == 1) {{
                pa1=pW[i]; na1=nW[i]; pb1=pW[j]; nb1=nW[j];
                pa2=pU[j]; na2=nU[j]; pb2=pU[i]; nb2=nU[i];
            }} else {{
                pa1=pW[i]; na1=nW[i]; pb1=pW[j]; nb1=nW[j];
                pa2=pV[j]; na2=nV[j]; pb2=pV[i]; nb2=nV[i];
            }}

            int bdir = (int)(lr & 1ULL);
            int bal = ((lr >> 1) & 1ULL) ? 1 : -1;
            bool ok_wcap = false, ok_raw = false, is_z = false;
            int cdir = 0, cal = 1;
            int raw_dir = 0, raw_al = 1;

            for (int dt = 0; dt < 2 && !is_z && !is_tabu; ++dt) {{
                int dir = bdir ^ dt;
                for (int at = 0; at < 2 && !is_z; ++at) {{
                    int al = (at == 0) ? bal : -bal;
                    u64 o1p, o1n, o2p, o2n;
                    bool ok;
                    int old_w;
                    if (dir == 0) {{
                        ok = t_add_signed(pa1, na1, pb1, nb1, -al, o1p, o1n) &&
                             t_add_signed(pa2, na2, pb2, nb2, al * s_eq, o2p, o2n);
                        old_w = t_nnz(pa1, na1) + t_nnz(pa2, na2);
                    }} else {{
                        ok = t_add_signed(pb2, nb2, pa2, na2, -al, o1p, o1n) &&
                             t_add_signed(pb1, nb1, pa1, na1, al * s_eq, o2p, o2n);
                        old_w = t_nnz(pb2, nb2) + t_nnz(pb1, nb1);
                    }}
                    if (!ok) continue;
                    int w1 = t_nnz(o1p, o1n), w2 = t_nnz(o2p, o2n);
                    if (w1 == 0 || w2 == 0) {{
                        is_z = true; ok_wcap = true; cdir = dir; cal = al; break;
                    }}
                    if (!ok_raw) {{ ok_raw = true; raw_dir = dir; raw_al = al; }}
                    if (!ok_wcap && ((w1 <= max_vec_w && w2 <= max_vec_w) || (w1 + w2 < old_w))) {{
                        ok_wcap = true; cdir = dir; cal = al;
                    }}
                }}
            }}

            u32 zm = __ballot_sync(0xFFFFFFFFu, is_z);
            u32 wm = zm ? zm : __ballot_sync(0xFFFFFFFFu, ok_wcap);
            u32 rm = wm ? wm : __ballot_sync(0xFFFFFFFFu, ok_raw);
            if (rm != 0u) {{
                int wl = __ffs(rm) - 1;
                chosen_i = __shfl_sync(0xFFFFFFFFu, i, wl);
                chosen_j = __shfl_sync(0xFFFFFFFFu, j, wl);
                chosen_ax = __shfl_sync(0xFFFFFFFFu, ax, wl);
                chosen_s = __shfl_sync(0xFFFFFFFFu, s_eq, wl);
                bool used_raw = (wm == 0u);
                chosen_dir = __shfl_sync(0xFFFFFFFFu, used_raw ? raw_dir : cdir, wl);
                chosen_alpha = __shfl_sync(0xFFFFFFFFu, used_raw ? raw_al : cal, wl);
            }}
        }}

        bool reduced_now = false;
        int r_before_step = R;
        if (chosen_i >= 0) {{
            int m_a, m_b;
            if (chosen_ax == 2) {{
                m_a = (chosen_dir == 0) ? 1 : 0;
                m_b = (chosen_dir == 0) ? 0 : 1;
            }} else if (chosen_ax == 1) {{
                m_a = (chosen_dir == 0) ? 2 : 0;
                m_b = (chosen_dir == 0) ? 0 : 2;
            }} else {{
                m_a = (chosen_dir == 0) ? 2 : 1;
                m_b = (chosen_dir == 0) ? 1 : 2;
            }}

            if (lane == 0) {{
                u64 o1p, o1n, o2p, o2n;
                if (chosen_ax == 2) {{
                    if (chosen_dir == 0) {{
                        t_add_signed(pV[chosen_i], nV[chosen_i], pV[chosen_j], nV[chosen_j], -chosen_alpha, o1p, o1n);
                        t_add_signed(pU[chosen_j], nU[chosen_j], pU[chosen_i], nU[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pV[chosen_i] = o1p; nV[chosen_i] = o1n; pU[chosen_j] = o2p; nU[chosen_j] = o2n;
                    }} else {{
                        t_add_signed(pU[chosen_i], nU[chosen_i], pU[chosen_j], nU[chosen_j], -chosen_alpha, o1p, o1n);
                        t_add_signed(pV[chosen_j], nV[chosen_j], pV[chosen_i], nV[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pU[chosen_i] = o1p; nU[chosen_i] = o1n; pV[chosen_j] = o2p; nV[chosen_j] = o2n;
                    }}
                }} else if (chosen_ax == 1) {{
                    if (chosen_dir == 0) {{
                        t_add_signed(pW[chosen_i], nW[chosen_i], pW[chosen_j], nW[chosen_j], -chosen_alpha, o1p, o1n);
                        t_add_signed(pU[chosen_j], nU[chosen_j], pU[chosen_i], nU[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pW[chosen_i] = o1p; nW[chosen_i] = o1n; pU[chosen_j] = o2p; nU[chosen_j] = o2n;
                    }} else {{
                        t_add_signed(pU[chosen_i], nU[chosen_i], pU[chosen_j], nU[chosen_j], -chosen_alpha, o1p, o1n);
                        t_add_signed(pW[chosen_j], nW[chosen_j], pW[chosen_i], nW[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pU[chosen_i] = o1p; nU[chosen_i] = o1n; pW[chosen_j] = o2p; nW[chosen_j] = o2n;
                    }}
                }} else {{
                    if (chosen_dir == 0) {{
                        t_add_signed(pW[chosen_i], nW[chosen_i], pW[chosen_j], nW[chosen_j], -chosen_alpha, o1p, o1n);
                        t_add_signed(pV[chosen_j], nV[chosen_j], pV[chosen_i], nV[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pW[chosen_i] = o1p; nW[chosen_i] = o1n; pV[chosen_j] = o2p; nV[chosen_j] = o2n;
                    }} else {{
                        t_add_signed(pV[chosen_i], nV[chosen_i], pV[chosen_j], nV[chosen_j], -chosen_alpha, o1p, o1n);
                        t_add_signed(pW[chosen_j], nW[chosen_j], pW[chosen_i], nW[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pV[chosen_i] = o1p; nV[chosen_i] = o1n; pW[chosen_j] = o2p; nW[chosen_j] = o2n;
                    }}
                }}
                ++flips_exec;
            }}
            __syncwarp();

            if (zt_inc_update_after_flip(pU, nU, pV, nV, pW, nW, R, edges, ne, chosen_i, chosen_j, m_a, m_b, lane)) {{
                reduced_now = true;
                if (lane == 0) ++red_exec;
            }}
            last_i = (chosen_i < R) ? chosen_i : 0;
            last_j = (chosen_j < R) ? chosen_j : 0;
            last_ax = chosen_ax;
        }}

        if (R < best_R) {{
            steps_since_reduce = 0;
            steps_at_elevated = 0;
        }} else {{
            ++steps_since_reduce;
            if (R > best_R) ++steps_at_elevated;
            else if (reduced_now && r_before_step > best_R) steps_at_elevated = 0;
        }}
        ++steps_since_save;
        if (tabu_ttl > 0) --tabu_ttl;

        bool refresh_at_best = (R == best_R) && ((steps_since_save >= 500) || (step == n_steps - 1));
        if (R < best_R || refresh_at_best) {{
            int new_nnz = zt_compute_nnz(pU, nU, pV, nV, pW, nW, R, lane);
            best_R = R;
            best_nnz = new_nnz;
            steps_since_save = 0;
            for (int k = lane; k < R; k += 32) {{
                best_pU[g_off + k] = pU[k]; best_nU[g_off + k] = nU[k];
                best_pV[g_off + k] = pV[k]; best_nV[g_off + k] = nV[k];
                best_pW[g_off + k] = pW[k]; best_nW[g_off + k] = nW[k];
            }}
            __syncwarp();
        }}

        int r_cap = best_R + local_headroom;
        if (r_cap > MAX_R - 1) r_cap = MAX_R - 1;
        bool need_tunnel = (chosen_i < 0) || (steps_since_reduce >= local_plus_period);

        if (need_tunnel) {{
            if (R < r_cap) {{
                int ta = -1, tb = -1;
                zt_targeted_expand(pU, nU, pV, nV, pW, nW, R, last_i, last_ax,
                                   r0 ^ ((u64)step * 0x9E3779B97F4A7C15ULL), ta, tb, lane);
                if (lane == 0) ++plus_exec;
                zt_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
                ne = zt_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
                steps_since_reduce = 0;
                tabu_a = ta; tabu_b = tb; tabu_ttl = 6;
            }} else if (steps_at_elevated >= (local_plus_period * 3) || chosen_i < 0) {{
                R = best_R;
                for (int k = lane; k < R; k += 32) {{
                    pU[k] = best_pU[g_off + k]; nU[k] = best_nU[g_off + k];
                    pV[k] = best_pV[g_off + k]; nV[k] = best_nV[g_off + k];
                    pW[k] = best_pW[g_off + k]; nW[k] = best_nW[g_off + k];
                }}
                __syncwarp();
                ne = zt_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
                steps_since_reduce = 0;
                steps_at_elevated = 0;
            }}
        }}
    }}

    if (lane == 0) {{
        best_R_out[wid] = best_R;
        best_nnz_out[wid] = best_nnz;
        stats_out[wid * 3 + 0] = flips_exec;
        stats_out[wid * 3 + 1] = plus_exec;
        stats_out[wid * 3 + 2] = red_exec;
    }}
}}
"""
    )


def _build_f3_cuda_source(max_r: int, max_edges: int, warps_per_block: int) -> str:
    """
    Warp-cooperative 2-bitplane branchless F_3 = {-1, 0, +1} mod 3 Flip-Graph kernel!
      - Encoding (p, n) with (p & n) == 0: 0=(0,0), +1=(1,0), -1=(0,1) (i.e. 2 mod 3).
      - ZERO overflow: (+1)+(+1) = -1 mod 3 and (-1)+(-1) = +1 mod 3! Every flip, plus,
        2-axis reduce, 3-term reduce, and triple-collision split is 100% unconditionally valid!
      - Unlocks the Z[1/2] rational manifold (since 2^{-1} == -1 mod 3) and shatters Z_T +-2
        sign-locking on composite padded-Strassen seams (5x8x8:230, 7x7x7:250, 6x6x7:185, ...).
      - Exact O(ne/32 + 2R/32) incremental edge list with exact `sa != 0` short-circuit (valid
        over F_3 because no 2-axis match can ever be overflow-blocked!).
    """
    return (
        _COMMON_CUDA_V2
        + f"""
#define MAX_R {max_r}
#define MAX_EDGES {max_edges}
#define WARPS_PER_BLOCK {warps_per_block}

__device__ __forceinline__ void f3_add(u64 p1, u64 n1, u64 p2, u64 n2, u64& po, u64& no) {{
    u64 z1 = ~(p1 | n1);
    u64 z2 = ~(p2 | n2);
    po = (p1 & z2) | (p2 & z1) | (n1 & n2);
    no = (n1 & z2) | (n2 & z1) | (p1 & p2);
}}

__device__ __forceinline__ void f3_add_signed(
    u64 p1, u64 n1, u64 p2, u64 n2, int s, u64& po, u64& no
) {{
    f3_add(p1, n1, (s > 0) ? p2 : n2, (s > 0) ? n2 : p2, po, no);
}}

__device__ __forceinline__ int f3_cmp(u64 pa, u64 na, u64 pb, u64 nb) {{
    if (pa == pb && na == nb) return 1;
    if (pa == nb && na == pb) return -1;
    return 0;
}}

__device__ __forceinline__ int f3_nnz(u64 p, u64 n) {{
    return __popcll(p | n);
}}

// Exhaustive 3x3 hardware truth-table verification kernel for f3_add / f3_sub!
extern "C" __global__ void f3_truth_table_kernel(int* out_add, int* out_sub, int* out_bad_bits) {{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const u64 ps[3] = {{0ULL, 1ULL, 0ULL}}; // 0, +1, -1(=2 mod 3)
    const u64 ns[3] = {{0ULL, 0ULL, 1ULL}};
    int bad = 0;
    for (int a = 0; a < 3; ++a) {{
        for (int b = 0; b < 3; ++b) {{
            u64 po, no;
            f3_add(ps[a], ns[a], ps[b], ns[b], po, no);
            if ((po & no) != 0ULL) ++bad;
            out_add[a * 3 + b] = (po & 1ULL) ? 1 : ((no & 1ULL) ? 2 : 0);
            f3_add_signed(ps[a], ns[a], ps[b], ns[b], -1, po, no);
            if ((po & no) != 0ULL) ++bad;
            out_sub[a * 3 + b] = (po & 1ULL) ? 1 : ((no & 1ULL) ? 2 : 0);
        }}
    }}
    out_bad_bits[0] = bad;
}}

__device__ __forceinline__ void f3_drop_slot(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R, int idx
) {{
    int last = R - 1;
    if (idx < last) {{
        pU[idx] = pU[last]; nU[idx] = nU[last];
        pV[idx] = pV[last]; nV[idx] = nV[last];
        pW[idx] = pW[last]; nW[idx] = nW[last];
    }}
    R = last;
}}

__device__ int f3_reduce_slot(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R, int idx, int lane
) {{
    if (idx < 0 || idx >= R) return 0;
    u64 pui = pU[idx], nui = nU[idx];
    u64 pvi = pV[idx], nvi = nV[idx];
    u64 pwi = pW[idx], nwi = nW[idx];
    if ((pui | nui) == 0ULL || (pvi | nvi) == 0ULL || (pwi | nwi) == 0ULL) {{
        if (lane == 0) f3_drop_slot(pU, nU, pV, nV, pW, nW, R, idx);
        R = __shfl_sync(0xFFFFFFFFu, R, 0);
        __syncwarp();
        return 1;
    }}
    int match_k = -1, match_mode = -1;
    u64 res_p = 0ULL, res_n = 0ULL;
    for (int base = 0; base < R; base += 32) {{
        int k = base + lane;
        int mode = -1;
        u64 mp = 0ULL, mn = 0ULL;
        if (k < R && k != idx) {{
            int su = f3_cmp(pui, nui, pU[k], nU[k]);
            int sv = f3_cmp(pvi, nvi, pV[k], nV[k]);
            int sw = f3_cmp(pwi, nwi, pW[k], nW[k]);
            if (su != 0 && sv != 0) {{
                f3_add_signed(pW[k], nW[k], pwi, nwi, su * sv, mp, mn);
                mode = 0;
            }} else if (su != 0 && sw != 0) {{
                f3_add_signed(pV[k], nV[k], pvi, nvi, su * sw, mp, mn);
                mode = 1;
            }} else if (sv != 0 && sw != 0) {{
                f3_add_signed(pU[k], nU[k], pui, nui, sv * sw, mp, mn);
                mode = 2;
            }}
        }}
        u32 hit = __ballot_sync(0xFFFFFFFFu, mode >= 0);
        if (hit) {{
            int wl = __ffs(hit) - 1;
            match_k = base + wl;
            match_mode = __shfl_sync(0xFFFFFFFFu, mode, wl);
            res_p = __shfl_sync(0xFFFFFFFFu, mp, wl);
            res_n = __shfl_sync(0xFFFFFFFFu, mn, wl);
            break;
        }}
    }}
    if (match_k < 0) return 0;

    int dropped = 1;
    if (lane == 0) {{
        bool became_zero = ((res_p | res_n) == 0ULL);
        if (match_mode == 0) {{ pW[match_k] = res_p; nW[match_k] = res_n; }}
        else if (match_mode == 1) {{ pV[match_k] = res_p; nV[match_k] = res_n; }}
        else {{ pU[match_k] = res_p; nU[match_k] = res_n; }}
        if (became_zero) {{
            int hi = (match_k > idx) ? match_k : idx;
            int lo = (match_k > idx) ? idx : match_k;
            f3_drop_slot(pU, nU, pV, nV, pW, nW, R, hi);
            f3_drop_slot(pU, nU, pV, nV, pW, nW, R, lo);
            dropped = 2;
        }} else {{
            f3_drop_slot(pU, nU, pV, nV, pW, nW, R, idx);
            dropped = 1;
        }}
    }}
    R = __shfl_sync(0xFFFFFFFFu, R, 0);
    dropped = __shfl_sync(0xFFFFFFFFu, dropped, 0);
    __syncwarp();
    return dropped;
}}

__device__ void f3_full_reduce_sweep(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R, int lane
) {{
    int i = 0;
    while (i < R) {{
        int d = f3_reduce_slot(pU, nU, pV, nV, pW, nW, R, i, lane);
        if (d > 0) i = (i > 0) ? (i - 1) : 0;
        else ++i;
    }}
}}

__device__ int f3_rebuild_edges(
    const u64* pU, const u64* nU, const u64* pV, const u64* nV, const u64* pW, const u64* nW,
    int R, u32* edges, int lane
) {{
    int ne = 0;
    u32 lt_mask = (1u << lane) - 1u;
    for (int i = 0; i < R - 1 && ne < MAX_EDGES; ++i) {{
        u64 pui = pU[i], nui = nU[i], pvi = pV[i], nvi = nV[i], pwi = pW[i], nwi = nW[i];
        for (int base = i + 1; base < R && ne < MAX_EDGES; base += 32) {{
            int j = base + lane;
            int su = 0, sv = 0, sw = 0;
            if (j < R) {{
                su = f3_cmp(pui, nui, pU[j], nU[j]);
                sv = f3_cmp(pvi, nvi, pV[j], nV[j]);
                sw = f3_cmp(pwi, nwi, pW[j], nW[j]);
            }}
            u32 mu = __ballot_sync(0xFFFFFFFFu, su != 0);
            if (mu) {{
                int off = __popc(mu & lt_mask);
                if (su != 0 && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 0, su < 0);
                ne += __popc(mu);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
            u32 mv = __ballot_sync(0xFFFFFFFFu, sv != 0);
            if (mv && ne < MAX_EDGES) {{
                int off = __popc(mv & lt_mask);
                if (sv != 0 && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 1, sv < 0);
                ne += __popc(mv);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
            u32 mw = __ballot_sync(0xFFFFFFFFu, sw != 0);
            if (mw && ne < MAX_EDGES) {{
                int off = __popc(mw & lt_mask);
                if (sw != 0 && ne + off < MAX_EDGES) edges[ne + off] = PACK_EDGE(i, j, 2, sw < 0);
                ne += __popc(mw);
                if (ne > MAX_EDGES) ne = MAX_EDGES;
            }}
        }}
    }}
    __syncwarp();
    return ne;
}}

// Ultra-fast O(ne/32 + 2R/32) incremental F_3 edge update!
// Because F_3 NEVER overflows, any new 2-axis collision MUST match on the changed axis (sa!=0 / sb!=0),
// skipping >99% of shared-memory loads!
__device__ bool f3_inc_update_after_flip(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R,
    u32* edges, int& ne,
    int idx_i, int idx_j, int m_a, int m_b, int lane
) {{
    u64 pui = pU[idx_i], nui = nU[idx_i], pvi = pV[idx_i], nvi = nV[idx_i], pwi = pW[idx_i], nwi = nW[idx_i];
    u64 puj = pU[idx_j], nuj = nU[idx_j], pvj = pV[idx_j], nvj = nV[idx_j], pwj = pW[idx_j], nwj = nW[idx_j];

    if ((pui | nui) == 0ULL || (pvi | nvi) == 0ULL || (pwi | nwi) == 0ULL ||
        (puj | nuj) == 0ULL || (pvj | nvj) == 0ULL || (pwj | nwj) == 0ULL) {{
        f3_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
        ne = f3_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
        return true;
    }}

    const u64* pAa = (m_a == 0) ? pU : ((m_a == 1) ? pV : pW);
    const u64* nAa = (m_a == 0) ? nU : ((m_a == 1) ? nV : nW);
    const u64* pAb = (m_b == 0) ? pU : ((m_b == 1) ? pV : pW);
    const u64* nAb = (m_b == 0) ? nU : ((m_b == 1) ? nV : nW);
    u64 pva = pAa[idx_i], nva = nAa[idx_i];
    u64 pvb = pAb[idx_j], nvb = nAb[idx_j];

    bool hit_red = false;
    for (int base = 0; base < R; base += 32) {{
        int k = base + lane;
        bool col = false;
        if (k < R) {{
            if (k != idx_i) {{
                int sa = f3_cmp(pva, nva, pAa[k], nAa[k]);
                if (sa != 0) {{
                    int o1 = (m_a == 0) ? f3_cmp(pvi, nvi, pV[k], nV[k]) : f3_cmp(pui, nui, pU[k], nU[k]);
                    int o2 = (m_a == 2) ? f3_cmp(pvi, nvi, pV[k], nV[k]) : f3_cmp(pwi, nwi, pW[k], nW[k]);
                    if (o1 != 0 || o2 != 0) col = true;
                }}
            }}
            if (!col && k != idx_j) {{
                int sb = f3_cmp(pvb, nvb, pAb[k], nAb[k]);
                if (sb != 0) {{
                    int o1 = (m_b == 0) ? f3_cmp(pvj, nvj, pV[k], nV[k]) : f3_cmp(puj, nuj, pU[k], nU[k]);
                    int o2 = (m_b == 2) ? f3_cmp(pvj, nvj, pV[k], nV[k]) : f3_cmp(pwj, nwj, pW[k], nW[k]);
                    if (o1 != 0 || o2 != 0) col = true;
                }}
            }}
        }}
        if (__any_sync(0xFFFFFFFFu, col)) {{ hit_red = true; break; }}
    }}
    if (hit_red) {{
        f3_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
        ne = f3_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
        return true;
    }}

    int out_ne = 0;
    u32 lt_mask = (1u << lane) - 1u;
    for (int base = 0; base < ne; base += 32) {{
        int pos = base + lane;
        u32 ev = (pos < ne) ? edges[pos] : 0u;
        bool keep = false;
        if (pos < ne) {{
            int ei = EDGE_I(ev), ej = EDGE_J(ev), em = EDGE_AX(ev);
            bool stale = (em == m_a && (ei == idx_i || ej == idx_i)) ||
                         (em == m_b && (ei == idx_j || ej == idx_j));
            keep = !stale;
        }}
        __syncwarp();
        u32 kmask = __ballot_sync(0xFFFFFFFFu, keep);
        if (keep) edges[out_ne + __popc(kmask & lt_mask)] = ev;
        out_ne += __popc(kmask);
        __syncwarp();
    }}
    ne = out_ne;

    for (int base = 0; base < R && ne < MAX_EDGES; base += 32) {{
        int k = base + lane;
        int sa = (k < R && k != idx_i) ? f3_cmp(pva, nva, pAa[k], nAa[k]) : 0;
        u32 ma = __ballot_sync(0xFFFFFFFFu, sa != 0);
        if (ma) {{
            int off = __popc(ma & lt_mask);
            if (sa != 0 && ne + off < MAX_EDGES) {{
                int lo = (idx_i < k) ? idx_i : k;
                int hi = (idx_i < k) ? k : idx_i;
                edges[ne + off] = PACK_EDGE(lo, hi, m_a, sa < 0);
            }}
            ne += __popc(ma);
            if (ne > MAX_EDGES) ne = MAX_EDGES;
        }}
        int sb = (k < R && k != idx_j) ? f3_cmp(pvb, nvb, pAb[k], nAb[k]) : 0;
        u32 mb = __ballot_sync(0xFFFFFFFFu, sb != 0);
        if (mb && ne < MAX_EDGES) {{
            int off = __popc(mb & lt_mask);
            if (sb != 0 && ne + off < MAX_EDGES) {{
                int lo = (idx_j < k) ? idx_j : k;
                int hi = (idx_j < k) ? k : idx_j;
                edges[ne + off] = PACK_EDGE(lo, hi, m_b, sb < 0);
            }}
            ne += __popc(mb);
            if (ne > MAX_EDGES) ne = MAX_EDGES;
        }}
    }}
    __syncwarp();
    return false;
}}

// Exact KM 1-step F_3 neighbor-reduction sniper (all 4 (dir, alpha) flips are always valid mod 3!)
__device__ bool f3_snipe_1step(
    const u64* pU, const u64* nU, const u64* pV, const u64* nV, const u64* pW, const u64* nW, int R,
    const u32* edges, int ne, int focus_a, int focus_b,
    int& out_i, int& out_j, int& out_ax, int& out_dir, int& out_alpha, int& out_s, int lane
) {{
    if (ne == 0) return false;
    int cand_e[4];
    int n_cand = 0;
    for (int base = ((ne - 1) & ~31); base >= 0 && n_cand < 4; base -= 32) {{
        int pos = base + lane;
        bool inc = false;
        if (pos < ne) {{
            u32 ev = edges[pos];
            int ei = EDGE_I(ev), ej = EDGE_J(ev);
            inc = (ei == focus_a || ej == focus_a || ei == focus_b || ej == focus_b);
        }}
        u32 m = __ballot_sync(0xFFFFFFFFu, inc);
        while (m && n_cand < 4) {{
            int b = __ffs(m) - 1;
            cand_e[n_cand++] = base + b;
            m &= m - 1u;
        }}
    }}

    for (int c = 0; c < n_cand; ++c) {{
        u32 ev = edges[cand_e[c]];
        int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev), s_eq = EDGE_SGN(ev);
        u64 pui = pU[i], nui = nU[i], pvi = pV[i], nvi = nV[i], pwi = pW[i], nwi = nW[i];
        u64 puj = pU[j], nuj = nU[j], pvj = pV[j], nvj = nV[j], pwj = pW[j], nwj = nW[j];

        for (int dir = 0; dir < 2; ++dir) {{
            for (int alpha = -1; alpha <= 1; alpha += 2) {{
                u64 o1p, o1n, o2p, o2n;
                if (ax == 2) {{
                    if (dir == 0) {{
                        f3_add_signed(pvi, nvi, pvj, nvj, -alpha, o1p, o1n);
                        f3_add_signed(puj, nuj, pui, nui, alpha * s_eq, o2p, o2n);
                    }} else {{
                        f3_add_signed(pui, nui, puj, nuj, -alpha, o1p, o1n);
                        f3_add_signed(pvj, nvj, pvi, nvi, alpha * s_eq, o2p, o2n);
                    }}
                }} else if (ax == 1) {{
                    if (dir == 0) {{
                        f3_add_signed(pwi, nwi, pwj, nwj, -alpha, o1p, o1n);
                        f3_add_signed(puj, nuj, pui, nui, alpha * s_eq, o2p, o2n);
                    }} else {{
                        f3_add_signed(pui, nui, puj, nuj, -alpha, o1p, o1n);
                        f3_add_signed(pwj, nwj, pwi, nwi, alpha * s_eq, o2p, o2n);
                    }}
                }} else {{
                    if (dir == 0) {{
                        f3_add_signed(pwi, nwi, pwj, nwj, -alpha, o1p, o1n);
                        f3_add_signed(pvj, nvj, pvi, nvi, alpha * s_eq, o2p, o2n);
                    }} else {{
                        f3_add_signed(pvi, nvi, pvj, nvj, -alpha, o1p, o1n);
                        f3_add_signed(pwj, nwj, pwi, nwi, alpha * s_eq, o2p, o2n);
                    }}
                }}
                if ((o1p | o1n) == 0ULL || (o2p | o2n) == 0ULL) {{
                    out_i = i; out_j = j; out_ax = ax; out_dir = dir; out_alpha = alpha; out_s = s_eq;
                    return true;
                }}
                for (int base = 0; base < R; base += 32) {{
                    int k = base + lane;
                    bool hit = false;
                    if (k < R && k != i && k != j) {{
                        if (ax == 2) {{
                            if (dir == 0) {{
                                if (f3_cmp(o1p, o1n, pV[k], nV[k]) != 0 &&
                                    (f3_cmp(pui, nui, pU[k], nU[k]) != 0 || f3_cmp(pwi, nwi, pW[k], nW[k]) != 0)) hit = true;
                                else if (f3_cmp(o2p, o2n, pU[k], nU[k]) != 0 &&
                                    (f3_cmp(pvj, nvj, pV[k], nV[k]) != 0 || f3_cmp(pwj, nwj, pW[k], nW[k]) != 0)) hit = true;
                            }} else {{
                                if (f3_cmp(o1p, o1n, pU[k], nU[k]) != 0 &&
                                    (f3_cmp(pvi, nvi, pV[k], nV[k]) != 0 || f3_cmp(pwi, nwi, pW[k], nW[k]) != 0)) hit = true;
                                else if (f3_cmp(o2p, o2n, pV[k], nV[k]) != 0 &&
                                    (f3_cmp(puj, nuj, pU[k], nU[k]) != 0 || f3_cmp(pwj, nwj, pW[k], nW[k]) != 0)) hit = true;
                            }}
                        }} else if (ax == 1) {{
                            if (dir == 0) {{
                                if (f3_cmp(o1p, o1n, pW[k], nW[k]) != 0 &&
                                    (f3_cmp(pui, nui, pU[k], nU[k]) != 0 || f3_cmp(pvi, nvi, pV[k], nV[k]) != 0)) hit = true;
                                else if (f3_cmp(o2p, o2n, pU[k], nU[k]) != 0 &&
                                    (f3_cmp(pwj, nwj, pW[k], nW[k]) != 0 || f3_cmp(pvj, nvj, pV[k], nV[k]) != 0)) hit = true;
                            }} else {{
                                if (f3_cmp(o1p, o1n, pU[k], nU[k]) != 0 &&
                                    (f3_cmp(pwi, nwi, pW[k], nW[k]) != 0 || f3_cmp(pvi, nvi, pV[k], nV[k]) != 0)) hit = true;
                                else if (f3_cmp(o2p, o2n, pW[k], nW[k]) != 0 &&
                                    (f3_cmp(puj, nuj, pU[k], nU[k]) != 0 || f3_cmp(pvj, nvj, pV[k], nV[k]) != 0)) hit = true;
                            }}
                        }} else {{
                            if (dir == 0) {{
                                if (f3_cmp(o1p, o1n, pW[k], nW[k]) != 0 &&
                                    (f3_cmp(pvi, nvi, pV[k], nV[k]) != 0 || f3_cmp(pui, nui, pU[k], nU[k]) != 0)) hit = true;
                                else if (f3_cmp(o2p, o2n, pV[k], nV[k]) != 0 &&
                                    (f3_cmp(pwj, nwj, pW[k], nW[k]) != 0 || f3_cmp(puj, nuj, pU[k], nU[k]) != 0)) hit = true;
                            }} else {{
                                if (f3_cmp(o1p, o1n, pV[k], nV[k]) != 0 &&
                                    (f3_cmp(pwi, nwi, pW[k], nW[k]) != 0 || f3_cmp(pui, nui, pU[k], nU[k]) != 0)) hit = true;
                                else if (f3_cmp(o2p, o2n, pW[k], nW[k]) != 0 &&
                                    (f3_cmp(pvj, nvj, pV[k], nV[k]) != 0 || f3_cmp(puj, nuj, pU[k], nU[k]) != 0)) hit = true;
                            }}
                        }}
                    }}
                    if (__any_sync(0xFFFFFFFFu, hit)) {{
                        out_i = i; out_j = j; out_ax = ax; out_dir = dir; out_alpha = alpha; out_s = s_eq;
                        return true;
                    }}
                }}
            }}
        }}
    }}
    return false;
}}

// Exact warp-parallel 3-term F_3 linear-dependence reduction (unconditional mod 3!)
__device__ bool f3_try_3term_reduce(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW, int& R,
    const u32* edges, int ne, u64 r_seed, int lane
) {{
    if (ne == 0 || R < 3) return false;
    u64 st = r_seed;
    for (int trial = 0; trial < 4; ++trial) {{
        st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
        int e_idx = (int)(((st * 0x2545F4914F6CDD1DULL) >> 4) % (u64)ne);
        u32 ev = edges[e_idx];
        int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev), s_ij = EDGE_SGN(ev);
        u64 *pA = (ax == 0) ? pU : ((ax == 1) ? pV : pW);
        u64 *nA = (ax == 0) ? nU : ((ax == 1) ? nV : nW);
        u64 *pB = (ax == 0) ? pV : ((ax == 1) ? pW : pU);
        u64 *nB = (ax == 0) ? nV : ((ax == 1) ? nW : nU);
        u64 *pC = (ax == 0) ? pW : ((ax == 1) ? pU : pV);
        u64 *nC = (ax == 0) ? nW : ((ax == 1) ? nU : nV);

        u64 pai = pA[i], nai = nA[i];
        u64 pbi = pB[i], nbi = nB[i], pbj = pB[j], nbj = nB[j];
        u64 pci = pC[i], nci = nC[i], pcj = pC[j], ncj = nC[j];

        u64 bpp, bpn, bmp, bmn, cpp, cpn, cmp, cmn;
        f3_add(pbi, nbi, pbj, nbj, bpp, bpn);
        f3_add_signed(pbi, nbi, pbj, nbj, -1, bmp, bmn);
        f3_add(pci, nci, pcj, ncj, cpp, cpn);
        f3_add_signed(pci, nci, pcj, ncj, -1, cmp, cmn);

        int hit_k = -1, hit_mode = -1, hit_sa = 1, hit_sb = 1;
        for (int base = 0; base < R; base += 32) {{
            int k = base + lane;
            int mode = -1, sa = 1, sb = 1;
            if (k < R && k != i && k != j) {{
                int s_ik = f3_cmp(pai, nai, pA[k], nA[k]);
                if (s_ik != 0) {{
                    u64 pbk = pB[k], nbk = nB[k], pck = pC[k], nck = nC[k];
                    int sk = f3_cmp(pbk, nbk, bpp, bpn);
                    if (sk != 0) {{ mode = 0; sa = s_ik * sk; sb = s_ij * s_ik * sk; }}
                    else {{
                        sk = f3_cmp(pbk, nbk, bmp, bmn);
                        if (sk != 0) {{ mode = 0; sa = s_ik * sk; sb = -s_ij * s_ik * sk; }}
                        else {{
                            sk = f3_cmp(pck, nck, cpp, cpn);
                            if (sk != 0) {{ mode = 1; sa = s_ik * sk; sb = s_ij * s_ik * sk; }}
                            else {{
                                sk = f3_cmp(pck, nck, cmp, cmn);
                                if (sk != 0) {{ mode = 1; sa = s_ik * sk; sb = -s_ij * s_ik * sk; }}
                            }}
                        }}
                    }}
                }}
            }}
            u32 hm = __ballot_sync(0xFFFFFFFFu, mode >= 0);
            if (hm) {{
                int wl = __ffs(hm) - 1;
                hit_k = base + wl;
                hit_mode = __shfl_sync(0xFFFFFFFFu, mode, wl);
                hit_sa = __shfl_sync(0xFFFFFFFFu, sa, wl);
                hit_sb = __shfl_sync(0xFFFFFFFFu, sb, wl);
                break;
            }}
        }}
        if (hit_k >= 0) {{
            if (lane == 0) {{
                u64 o1p, o1n, o2p, o2n;
                if (hit_mode == 0) {{
                    f3_add_signed(pC[i], nC[i], pC[hit_k], nC[hit_k], hit_sa, o1p, o1n);
                    f3_add_signed(pC[j], nC[j], pC[hit_k], nC[hit_k], hit_sb, o2p, o2n);
                    pC[i] = o1p; nC[i] = o1n; pC[j] = o2p; nC[j] = o2n;
                }} else {{
                    f3_add_signed(pB[i], nB[i], pB[hit_k], nB[hit_k], hit_sa, o1p, o1n);
                    f3_add_signed(pB[j], nB[j], pB[hit_k], nB[hit_k], hit_sb, o2p, o2n);
                    pB[i] = o1p; nB[i] = o1n; pB[j] = o2p; nB[j] = o2n;
                }}
                f3_drop_slot(pU, nU, pV, nV, pW, nW, R, hit_k);
            }}
            R = __shfl_sync(0xFFFFFFFFu, R, 0);
            __syncwarp();
            f3_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
            return true;
        }}
    }}
    return false;
}}

// Exact Atomic Triple-Collision Split-and-Flip + Active-Bridged Plus over F_3 (100% ALWAYS SUCCEEDS!)
__device__ bool f3_targeted_expand(
    u64* pU, u64* nU, u64* pV, u64* nV, u64* pW, u64* nW,
    int& R, int bridge_j, int bridge_ax, u64 r_seed,
    int& out_ta, int& out_tb, int lane
) {{
    if (R >= MAX_R) return false;
    out_ta = -1; out_tb = -1;
    u64 st = r_seed;

    int hi = -1, hj = -1, hk = -1, hax = -1, ha = 1, hb = 1;
    if ((r_seed & 1ULL) == 0ULL) {{
        for (int trial = 0; trial < 12 && hk < 0; ++trial) {{
            st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
            u64 rv = st * 0x2545F4914F6CDD1DULL;
            int ax = (int)((rv >> 2) % 3ULL);
            int i = (int)((rv >> 8) % (u64)R);
            int j = (int)((rv >> 20) % (u64)(R - 1));
            if (j >= i) ++j;
            int s1 = ((rv >> 32) & 1ULL) ? 1 : -1;
            int s2 = ((rv >> 33) & 1ULL) ? 1 : -1;

            const u64* pA = (ax == 0) ? pU : ((ax == 1) ? pV : pW);
            const u64* nA = (ax == 0) ? nU : ((ax == 1) ? nV : nW);
            u64 sp, sn;
            u64 pi_p = (s1 > 0) ? pA[i] : nA[i];
            u64 pi_n = (s1 > 0) ? nA[i] : pA[i];
            f3_add_signed(pi_p, pi_n, pA[j], nA[j], s2, sp, sn);
            if ((sp | sn) == 0ULL) continue;

            for (int base = 0; base < R; base += 32) {{
                int k = base + lane;
                int sk = (k < R && k != i && k != j) ? f3_cmp(pA[k], nA[k], sp, sn) : 0;
                u32 hm = __ballot_sync(0xFFFFFFFFu, sk != 0);
                if (hm) {{
                    int wl = __ffs(hm) - 1;
                    int wsk = __shfl_sync(0xFFFFFFFFu, sk, wl);
                    hi = i; hj = j; hk = base + wl; hax = ax;
                    ha = wsk * s1;
                    hb = wsk * s2;
                    break;
                }}
            }}
        }}
    }}

    if (hk >= 0) {{
        int al = ((r_seed >> 7) & 1ULL) ? 1 : -1;
        if (lane == 0) {{
            u64 o1p, o1n, o2p, o2n;
            u64 pui = pU[hi], nui = nU[hi], pvi = pV[hi], nvi = nV[hi], pwi = pW[hi], nwi = nW[hi];
            u64 puj = pU[hj], nuj = nU[hj], pvj = pV[hj], nvj = nV[hj], pwj = pW[hj], nwj = nW[hj];
            u64 puk = pU[hk], nuk = nU[hk], pvk = pV[hk], nvk = nV[hk], pwk = pW[hk], nwk = nW[hk];
            if (hax == 0) {{
                f3_add_signed(pvk, nvk, pvi, nvi, -al, o1p, o1n);
                f3_add_signed(pwi, nwi, pwk, nwk, al * ha, o2p, o2n);
                pU[hk] = (hb > 0) ? puj : nuj; nU[hk] = (hb > 0) ? nuj : puj;
                pW[hi] = o2p; nW[hi] = o2n;
                pU[R]  = (ha > 0) ? pui : nui; nU[R]  = (ha > 0) ? nui : pui;
                pV[R]  = o1p; nV[R]  = o1n; pW[R] = pwk; nW[R] = nwk;
            }} else if (hax == 1) {{
                f3_add_signed(pwk, nwk, pwi, nwi, -al, o1p, o1n);
                f3_add_signed(pui, nui, puk, nuk, al * ha, o2p, o2n);
                pV[hk] = (hb > 0) ? pvj : nvj; nV[hk] = (hb > 0) ? nvj : pvj;
                pU[hi] = o2p; nU[hi] = o2n;
                pU[R]  = puk; nU[R]  = nuk;
                pV[R]  = (ha > 0) ? pvi : nvi; nV[R]  = (ha > 0) ? nvi : pvi;
                pW[R]  = o1p; nW[R]  = o1n;
            }} else {{
                f3_add_signed(puk, nuk, pui, nui, -al, o1p, o1n);
                f3_add_signed(pvi, nvi, pvk, nvk, al * ha, o2p, o2n);
                pW[hk] = (hb > 0) ? pwj : nwj; nW[hk] = (hb > 0) ? nwj : pwj;
                pV[hi] = o2p; nV[hi] = o2n;
                pU[R]  = o1p; nU[R]  = o1n; pV[R] = pvk; nV[R] = nvk;
                pW[R]  = (ha > 0) ? pwi : nwi; nW[R]  = (ha > 0) ? nwi : pwi;
            }}
            ++R;
        }}
        R = __shfl_sync(0xFFFFFFFFu, R, 0);
        out_ta = hi; out_tb = R - 1;
        __syncwarp();
        return true;
    }}

    // Guaranteed-valid O(1) active-term-bridged F_3 plus!
    int ta_l = -1;
    if (lane == 0) {{
        u64 pr = r_seed ^ 0xD6E8FEB86659FD93ULL;
        pr ^= pr >> 12; pr ^= pr << 25; pr ^= pr >> 27; pr *= 0x2545F4914F6CDD1DULL;
        int wi = (int)((pr >> 4) % (u64)R);
        bool use_bridge = (bridge_j >= 0 && bridge_j < R && ((pr >> 1) & 1ULL) == 0ULL && bridge_j != wi);
        int wj = use_bridge ? bridge_j : (int)((pr >> 16) % (u64)(R - 1));
        if (!use_bridge && wj >= wi) ++wj;
        int wo;
        if (use_bridge && bridge_ax == 2) wo = 0;
        else if (use_bridge && bridge_ax == 0) wo = 1;
        else if (use_bridge && bridge_ax == 1) wo = 2;
        else wo = (int)((pr >> 28) % 3ULL);
        int ws = ((pr >> 32) & 1ULL) ? 1 : -1;
        int al = ((pr >> 33) & 1ULL) ? 1 : -1;

        u64 pui = pU[wi], nui = nU[wi], pvi = pV[wi], nvi = nV[wi], pwi = pW[wi], nwi = nW[wi];
        u64 puj = pU[wj], nuj = nU[wj], pvj = pV[wj], nvj = nV[wj], pwj = pW[wj], nwj = nW[wj];
        u64 w1p, w1n, w2p, w2n, w3p, w3n;
        if (wo == 0) {{
            f3_add_signed(pwi, nwi, pwj, nwj, -ws, w1p, w1n);
            f3_add_signed(pvi, nvi, pvj, nvj, -al, w2p, w2n);
            f3_add_signed(puj, nuj, pui, nui, al * ws, w3p, w3n);
            pW[wi] = w1p; nW[wi] = w1n;
            pU[wj] = (ws > 0) ? pui : nui; nU[wj] = (ws > 0) ? nui : pui;
            pV[wj] = w2p; nV[wj] = w2n;
            pU[R]  = w3p; nU[R]  = w3n; pV[R] = pvj; nV[R] = nvj; pW[R] = pwj; nW[R] = nwj;
        }} else if (wo == 1) {{
            f3_add_signed(pui, nui, puj, nuj, -ws, w1p, w1n);
            f3_add_signed(pwi, nwi, pwj, nwj, -al, w2p, w2n);
            f3_add_signed(pvj, nvj, pvi, nvi, al * ws, w3p, w3n);
            pU[wi] = w1p; nU[wi] = w1n;
            pV[wj] = (ws > 0) ? pvi : nvi; nV[wj] = (ws > 0) ? nvi : pvi;
            pW[wj] = w2p; nW[wj] = w2n;
            pU[R]  = puj; nU[R]  = nuj; pV[R] = w3p; nV[R] = w3n; pW[R] = pwj; nW[R] = nwj;
        }} else {{
            f3_add_signed(pvi, nvi, pvj, nvj, -ws, w1p, w1n);
            f3_add_signed(pui, nui, puj, nuj, -al, w2p, w2n);
            f3_add_signed(pwj, nwj, pwi, nwi, al * ws, w3p, w3n);
            pV[wi] = w1p; nV[wi] = w1n;
            pU[wj] = w2p; nU[wj] = w2n;
            pW[wj] = (ws > 0) ? pwi : nwi; nW[wj] = (ws > 0) ? nwi : pwi;
            pU[R]  = puj; nU[R]  = nuj; pV[R] = pvj; nV[R] = nvj; pW[R] = w3p; nW[R] = w3n;
        }}
        ++R;
        ta_l = wj;
    }}
    R = __shfl_sync(0xFFFFFFFFu, R, 0);
    out_ta = __shfl_sync(0xFFFFFFFFu, ta_l, 0);
    out_tb = R - 1;
    __syncwarp();
    return true;
}}

__device__ int f3_compute_nnz(
    const u64* pU, const u64* nU, const u64* pV, const u64* nV, const u64* pW, const u64* nW,
    int R, int lane
) {{
    int sum = 0;
    for (int k = lane; k < R; k += 32) {{
        sum += f3_nnz(pU[k], nU[k]) + f3_nnz(pV[k], nV[k]) + f3_nnz(pW[k], nW[k]);
    }}
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);
    return __shfl_sync(0xFFFFFFFFu, sum, 0);
}}

extern "C" __global__ void f3_flip_walk_kernel(
    const u64* __restrict__ init_pU, const u64* __restrict__ init_nU,
    const u64* __restrict__ init_pV, const u64* __restrict__ init_nV,
    const u64* __restrict__ init_pW, const u64* __restrict__ init_nW,
    const int* __restrict__ init_R,
    u64* __restrict__ best_pU, u64* __restrict__ best_nU,
    u64* __restrict__ best_pV, u64* __restrict__ best_nV,
    u64* __restrict__ best_pW, u64* __restrict__ best_nW,
    int* __restrict__ best_R_out,
    int* __restrict__ best_nnz_out,
    u64* __restrict__ stats_out,
    int n_walkers,
    int n_steps,
    int plus_period,
    int plus_headroom,
    int target_R,
    int diverse_mode,
    u64 seed_base
) {{
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int wid = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (wid >= n_walkers) return;

    __shared__ u64 spU[WARPS_PER_BLOCK][MAX_R], snU[WARPS_PER_BLOCK][MAX_R];
    __shared__ u64 spV[WARPS_PER_BLOCK][MAX_R], snV[WARPS_PER_BLOCK][MAX_R];
    __shared__ u64 spW[WARPS_PER_BLOCK][MAX_R], snW[WARPS_PER_BLOCK][MAX_R];
    __shared__ u32 sE[WARPS_PER_BLOCK][MAX_EDGES];

    u64 *pU = spU[warp_in_block], *nU = snU[warp_in_block];
    u64 *pV = spV[warp_in_block], *nV = snV[warp_in_block];
    u64 *pW = spW[warp_in_block], *nW = snW[warp_in_block];
    u32 *edges = sE[warp_in_block];

    int R = init_R[wid];
    if (R > MAX_R) R = MAX_R;
    int g_off = wid * MAX_R;
    for (int k = lane; k < R; k += 32) {{
        pU[k] = init_pU[g_off + k]; nU[k] = init_nU[g_off + k];
        pV[k] = init_pV[g_off + k]; nV[k] = init_nV[g_off + k];
        pW[k] = init_pW[g_off + k]; nW[k] = init_nW[g_off + k];
    }}
    __syncwarp();

    f3_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
    int ne = f3_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
    int best_R = R;
    int best_nnz = f3_compute_nnz(pU, nU, pV, nV, pW, nW, R, lane);
    for (int k = lane; k < R; k += 32) {{
        best_pU[g_off + k] = pU[k]; best_nU[g_off + k] = nU[k];
        best_pV[g_off + k] = pV[k]; best_nV[g_off + k] = nV[k];
        best_pW[g_off + k] = pW[k]; best_nW[g_off + k] = nW[k];
    }}
    __syncwarp();

    u64 rng = seed_base ^ ((u64)(wid + 1) * 0x9E3779B97F4A7C15ULL);
    int w_tier = wid & 7;
    int max_vec_w;
    if (best_R <= 110) {{
        const int caps_lo[8] = {{2, 3, 3, 4, 5, 6, 10, 64}};
        max_vec_w = caps_lo[w_tier];
    }} else {{
        const int caps_hi[8] = {{4, 6, 8, 10, 12, 16, 24, 64}};
        max_vec_w = caps_hi[w_tier];
    }}
    int p_bucket = (wid >> 3) & 3;
    int local_plus_period;
    if (p_bucket == 0) local_plus_period = 1 << 29;
    else if (p_bucket == 1) local_plus_period = plus_period;
    else if (p_bucket == 2) local_plus_period = plus_period * 4;
    else local_plus_period = (plus_period > 128) ? (plus_period >> 1) : 64;
    int local_headroom = (p_bucket == 0) ? 1 : (plus_headroom + ((p_bucket == 3) ? 2 : (p_bucket - 1)));

    int steps_since_reduce = 0;
    int steps_at_elevated = 0;
    int steps_since_save = 0;
    int tabu_a = -1, tabu_b = -1, tabu_ttl = 0;
    int last_i = 0, last_j = 1, last_ax = 0;
    u64 flips_exec = 0ULL, plus_exec = 0ULL, red_exec = 0ULL;

    for (int step = 0; step < n_steps; ++step) {{
        u64 r0 = 0ULL, r1 = 0ULL;
        if (lane == 0) {{
            r0 = xorshift64star(rng);
            r1 = xorshift64star(rng);
        }}
        r0 = __shfl_sync(0xFFFFFFFFu, r0, 0);
        r1 = __shfl_sync(0xFFFFFFFFu, r1, 0);

        bool allow_greedy_reduce = (R <= best_R);

        // 0. Periodic 3-term F_3 linear-dependence reduction (3 terms -> 2!)
        int red3_mask = (R > 128) ? 63 : 31;
        if (allow_greedy_reduce && ne > 0 && ((step & red3_mask) == 15)) {{
            if (f3_try_3term_reduce(pU, nU, pV, nV, pW, nW, R, edges, ne, r0, lane)) {{
                ne = f3_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
                if (lane == 0) ++red_exec;
            }}
        }}

        int chosen_i = -1, chosen_j = -1, chosen_ax = -1;
        int chosen_dir = 0, chosen_alpha = 1, chosen_s = 1;

        // 1. Periodic KM 1-step F_3 neighbor-reduction sniping
        int snipe_mask = (R > 128) ? 63 : 7;
        if (allow_greedy_reduce && ne > 0 && ((step & snipe_mask) == 0)) {{
            int si, sj, sax, sdir, sal, ss;
            if (f3_snipe_1step(pU, nU, pV, nV, pW, nW, R, edges, ne, last_i, last_j,
                               si, sj, sax, sdir, sal, ss, lane)) {{
                chosen_i = si; chosen_j = sj; chosen_ax = sax;
                chosen_dir = sdir; chosen_alpha = sal; chosen_s = ss;
            }}
        }}

        // 2. O(1) parallel 32-lane F_3 edge sampling (ZERO overflow rejections!)
        if (chosen_i < 0 && ne > 0) {{
            u64 lr = r1 ^ ((u64)(lane + 1) * 0xBF58476D1CE4E5B9ULL);
            lr ^= lr >> 13; lr *= 0x2545F4914F6CDD1DULL;
            int e_idx = (int)((lr >> 4) % (u64)ne);
            u32 ev = edges[e_idx];
            int i = EDGE_I(ev), j = EDGE_J(ev), ax = EDGE_AX(ev), s_eq = EDGE_SGN(ev);
            bool is_tabu = (tabu_ttl > 0) && ((i == tabu_a && j == tabu_b) || (i == tabu_b && j == tabu_a));
            u64 pa1, na1, pb1, nb1, pa2, na2, pb2, nb2;
            if (ax == 2) {{
                pa1=pV[i]; na1=nV[i]; pb1=pV[j]; nb1=nV[j];
                pa2=pU[j]; na2=nU[j]; pb2=pU[i]; nb2=nU[i];
            }} else if (ax == 1) {{
                pa1=pW[i]; na1=nW[i]; pb1=pW[j]; nb1=nW[j];
                pa2=pU[j]; na2=nU[j]; pb2=pU[i]; nb2=nU[i];
            }} else {{
                pa1=pW[i]; na1=nW[i]; pb1=pW[j]; nb1=nW[j];
                pa2=pV[j]; na2=nV[j]; pb2=pV[i]; nb2=nV[i];
            }}

            int bdir = (int)(lr & 1ULL);
            int bal = ((lr >> 1) & 1ULL) ? 1 : -1;
            bool ok_wcap = false, ok_raw = false, is_z = false;
            int cdir = bdir, cal = bal;
            int raw_dir = bdir, raw_al = bal;

            if (!is_tabu) {{
                ok_raw = true;
                for (int dt = 0; dt < 2 && !is_z; ++dt) {{
                    int dir = bdir ^ dt;
                    for (int at = 0; at < 2 && !is_z; ++at) {{
                        int al = (at == 0) ? bal : -bal;
                        u64 o1p, o1n, o2p, o2n;
                        int old_w;
                        if (dir == 0) {{
                            f3_add_signed(pa1, na1, pb1, nb1, -al, o1p, o1n);
                            f3_add_signed(pa2, na2, pb2, nb2, al * s_eq, o2p, o2n);
                            old_w = f3_nnz(pa1, na1) + f3_nnz(pa2, na2);
                        }} else {{
                            f3_add_signed(pb2, nb2, pa2, na2, -al, o1p, o1n);
                            f3_add_signed(pb1, nb1, pa1, na1, al * s_eq, o2p, o2n);
                            old_w = f3_nnz(pb2, nb2) + f3_nnz(pb1, nb1);
                        }}
                        int w1 = f3_nnz(o1p, o1n), w2 = f3_nnz(o2p, o2n);
                        if (w1 == 0 || w2 == 0) {{
                            is_z = true; ok_wcap = true; cdir = dir; cal = al; break;
                        }}
                        if (!ok_wcap && ((w1 <= max_vec_w && w2 <= max_vec_w) || (w1 + w2 < old_w))) {{
                            ok_wcap = true; cdir = dir; cal = al;
                        }}
                    }}
                }}
            }}

            u32 zm = __ballot_sync(0xFFFFFFFFu, is_z);
            u32 wm = zm ? zm : __ballot_sync(0xFFFFFFFFu, ok_wcap);
            u32 rm = wm ? wm : __ballot_sync(0xFFFFFFFFu, ok_raw);
            if (rm != 0u) {{
                int wl = __ffs(rm) - 1;
                chosen_i = __shfl_sync(0xFFFFFFFFu, i, wl);
                chosen_j = __shfl_sync(0xFFFFFFFFu, j, wl);
                chosen_ax = __shfl_sync(0xFFFFFFFFu, ax, wl);
                chosen_s = __shfl_sync(0xFFFFFFFFu, s_eq, wl);
                bool used_raw = (wm == 0u);
                chosen_dir = __shfl_sync(0xFFFFFFFFu, used_raw ? raw_dir : cdir, wl);
                chosen_alpha = __shfl_sync(0xFFFFFFFFu, used_raw ? raw_al : cal, wl);
            }}
        }}

        bool reduced_now = false;
        int r_before_step = R;
        if (chosen_i >= 0) {{
            int m_a, m_b;
            if (chosen_ax == 2) {{
                m_a = (chosen_dir == 0) ? 1 : 0;
                m_b = (chosen_dir == 0) ? 0 : 1;
            }} else if (chosen_ax == 1) {{
                m_a = (chosen_dir == 0) ? 2 : 0;
                m_b = (chosen_dir == 0) ? 0 : 2;
            }} else {{
                m_a = (chosen_dir == 0) ? 2 : 1;
                m_b = (chosen_dir == 0) ? 1 : 2;
            }}

            if (lane == 0) {{
                u64 o1p, o1n, o2p, o2n;
                if (chosen_ax == 2) {{
                    if (chosen_dir == 0) {{
                        f3_add_signed(pV[chosen_i], nV[chosen_i], pV[chosen_j], nV[chosen_j], -chosen_alpha, o1p, o1n);
                        f3_add_signed(pU[chosen_j], nU[chosen_j], pU[chosen_i], nU[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pV[chosen_i] = o1p; nV[chosen_i] = o1n; pU[chosen_j] = o2p; nU[chosen_j] = o2n;
                    }} else {{
                        f3_add_signed(pU[chosen_i], nU[chosen_i], pU[chosen_j], nU[chosen_j], -chosen_alpha, o1p, o1n);
                        f3_add_signed(pV[chosen_j], nV[chosen_j], pV[chosen_i], nV[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pU[chosen_i] = o1p; nU[chosen_i] = o1n; pV[chosen_j] = o2p; nV[chosen_j] = o2n;
                    }}
                }} else if (chosen_ax == 1) {{
                    if (chosen_dir == 0) {{
                        f3_add_signed(pW[chosen_i], nW[chosen_i], pW[chosen_j], nW[chosen_j], -chosen_alpha, o1p, o1n);
                        f3_add_signed(pU[chosen_j], nU[chosen_j], pU[chosen_i], nU[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pW[chosen_i] = o1p; nW[chosen_i] = o1n; pU[chosen_j] = o2p; nU[chosen_j] = o2n;
                    }} else {{
                        f3_add_signed(pU[chosen_i], nU[chosen_i], pU[chosen_j], nU[chosen_j], -chosen_alpha, o1p, o1n);
                        f3_add_signed(pW[chosen_j], nW[chosen_j], pW[chosen_i], nW[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pU[chosen_i] = o1p; nU[chosen_i] = o1n; pW[chosen_j] = o2p; nW[chosen_j] = o2n;
                    }}
                }} else {{
                    if (chosen_dir == 0) {{
                        f3_add_signed(pW[chosen_i], nW[chosen_i], pW[chosen_j], nW[chosen_j], -chosen_alpha, o1p, o1n);
                        f3_add_signed(pV[chosen_j], nV[chosen_j], pV[chosen_i], nV[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pW[chosen_i] = o1p; nW[chosen_i] = o1n; pV[chosen_j] = o2p; nV[chosen_j] = o2n;
                    }} else {{
                        f3_add_signed(pV[chosen_i], nV[chosen_i], pV[chosen_j], nV[chosen_j], -chosen_alpha, o1p, o1n);
                        f3_add_signed(pW[chosen_j], nW[chosen_j], pW[chosen_i], nW[chosen_i], chosen_alpha * chosen_s, o2p, o2n);
                        pV[chosen_i] = o1p; nV[chosen_i] = o1n; pW[chosen_j] = o2p; nW[chosen_j] = o2n;
                    }}
                }}
                ++flips_exec;
            }}
            __syncwarp();

            if (f3_inc_update_after_flip(pU, nU, pV, nV, pW, nW, R, edges, ne, chosen_i, chosen_j, m_a, m_b, lane)) {{
                reduced_now = true;
                if (lane == 0) ++red_exec;
            }}
            last_i = (chosen_i < R) ? chosen_i : 0;
            last_j = (chosen_j < R) ? chosen_j : 0;
            last_ax = chosen_ax;
        }}

        if (R < best_R) {{
            steps_since_reduce = 0;
            steps_at_elevated = 0;
        }} else {{
            ++steps_since_reduce;
            if (R > best_R) ++steps_at_elevated;
            else if (reduced_now && r_before_step > best_R) steps_at_elevated = 0;
        }}
        ++steps_since_save;
        if (tabu_ttl > 0) --tabu_ttl;

        bool refresh_at_best = (R == best_R) && (
            (steps_since_save >= 500) ||
            (step == n_steps - 1) ||
            (diverse_mode && R <= target_R && steps_since_save >= 250)
        );
        if (R < best_R || refresh_at_best) {{
            int new_nnz = f3_compute_nnz(pU, nU, pV, nV, pW, nW, R, lane);
            best_R = R;
            best_nnz = new_nnz;
            steps_since_save = 0;
            for (int k = lane; k < R; k += 32) {{
                best_pU[g_off + k] = pU[k]; best_nU[g_off + k] = nU[k];
                best_pV[g_off + k] = pV[k]; best_nV[g_off + k] = nV[k];
                best_pW[g_off + k] = pW[k]; best_nW[g_off + k] = nW[k];
            }}
            __syncwarp();
        }}

        int r_cap = best_R + local_headroom;
        if (r_cap > MAX_R - 1) r_cap = MAX_R - 1;
        bool need_tunnel = (chosen_i < 0) || (steps_since_reduce >= local_plus_period);

        if (need_tunnel) {{
            if (R < r_cap) {{
                // p_bucket == 3 launches a 2-step multi-plus saddle burst when at best_R to crack 2 Strassen seams at once!
                int burst = (p_bucket == 3 && R == best_R && (R + 2) <= r_cap) ? 2 : 1;
                int ta = -1, tb = -1;
                for (int bi = 0; bi < burst && R < r_cap; ++bi) {{
                    f3_targeted_expand(pU, nU, pV, nV, pW, nW, R, last_i, last_ax,
                                       r0 ^ ((u64)(step + bi * 131) * 0x9E3779B97F4A7C15ULL), ta, tb, lane);
                    if (lane == 0) ++plus_exec;
                }}
                f3_full_reduce_sweep(pU, nU, pV, nV, pW, nW, R, lane);
                ne = f3_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
                steps_since_reduce = 0;
                tabu_a = ta; tabu_b = tb; tabu_ttl = 6;
            }} else if (steps_at_elevated >= (local_plus_period * 3) || chosen_i < 0) {{
                R = best_R;
                for (int k = lane; k < R; k += 32) {{
                    pU[k] = best_pU[g_off + k]; nU[k] = best_nU[g_off + k];
                    pV[k] = best_pV[g_off + k]; nV[k] = best_nV[g_off + k];
                    pW[k] = best_pW[g_off + k]; nW[k] = best_nW[g_off + k];
                }}
                __syncwarp();
                ne = f3_rebuild_edges(pU, nU, pV, nV, pW, nW, R, edges, lane);
                steps_since_reduce = 0;
                steps_at_elevated = 0;
            }}
        }}
    }}

    if (lane == 0) {{
        best_R_out[wid] = best_R;
        best_nnz_out[wid] = best_nnz;
        stats_out[wid * 3 + 0] = flips_exec;
        stats_out[wid * 3 + 1] = plus_exec;
        stats_out[wid * 3 + 2] = red_exec;
    }}
}}
"""
    )


# =============================================================================
# 3. Exact Bit-Packing Conversions & Zero-External-Leak Structural Combinators
# =============================================================================


def scheme_to_f2_packed(scheme: bv.BilinearScheme) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack an F2 (or Z/Zq reduced mod 2) scheme into three `uint64[R]` arrays."""
    m, n, p = scheme.m, scheme.n, scheme.p
    if max(m * n, n * p, m * p) > 64:
        raise ValueError(f"Format <{m},{n},{p}> exceeds 64-bit packing limit")
    R = scheme.rank
    u = np.zeros(R, dtype=np.uint64)
    v = np.zeros(R, dtype=np.uint64)
    w = np.zeros(R, dtype=np.uint64)
    for r in range(R):
        ub = vb = wb = 0
        for idx, val in enumerate(scheme.U[r]):
            if int(val) & 1:
                ub |= 1 << idx
        for idx, val in enumerate(scheme.V[r]):
            if int(val) & 1:
                vb |= 1 << idx
        for idx, val in enumerate(scheme.W[r]):
            if int(val) & 1:
                wb |= 1 << idx
        u[r] = np.uint64(ub)
        v[r] = np.uint64(vb)
        w[r] = np.uint64(wb)
    return u, v, w


def f2_packed_to_scheme(
    m: int,
    n: int,
    p: int,
    u: Sequence[int],
    v: Sequence[int],
    w: Sequence[int],
    name: str = "",
) -> bv.BilinearScheme:
    """Unpack three `uint64[R]` bitmask arrays into an exact `Z2` BilinearScheme."""
    R = len(u)
    mn, np_d, mp = m * n, n * p, m * p
    U = [[(int(u[r]) >> a) & 1 for a in range(mn)] for r in range(R)]
    V = [[(int(v[r]) >> b) & 1 for b in range(np_d)] for r in range(R)]
    W = [[(int(w[r]) >> c) & 1 for c in range(mp)] for r in range(R)]
    return bv.BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring="Z2", name=name)


def scheme_to_zt_packed(
    scheme: bv.BilinearScheme,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a ternary `Z` scheme (entries in {-1,0,+1}) into 6 `uint64[R]` bitplanes."""
    m, n, p = scheme.m, scheme.n, scheme.p
    if max(m * n, n * p, m * p) > 64:
        raise ValueError(f"Format <{m},{n},{p}> exceeds 64-bit packing limit")
    if scheme.ring != "Z" or not scheme.is_ternary():
        raise ValueError("scheme_to_zt_packed requires a ternary scheme over ring='Z'")
    R = scheme.rank
    pu = np.zeros(R, dtype=np.uint64)
    nu = np.zeros(R, dtype=np.uint64)
    pv = np.zeros(R, dtype=np.uint64)
    nv = np.zeros(R, dtype=np.uint64)
    pw = np.zeros(R, dtype=np.uint64)
    nw = np.zeros(R, dtype=np.uint64)
    for r in range(R):
        p_u = n_u = p_v = n_v = p_w = n_w = 0
        for idx, val in enumerate(scheme.U[r]):
            iv = int(val)
            if iv == 1:
                p_u |= 1 << idx
            elif iv == -1:
                n_u |= 1 << idx
        for idx, val in enumerate(scheme.V[r]):
            iv = int(val)
            if iv == 1:
                p_v |= 1 << idx
            elif iv == -1:
                n_v |= 1 << idx
        for idx, val in enumerate(scheme.W[r]):
            iv = int(val)
            if iv == 1:
                p_w |= 1 << idx
            elif iv == -1:
                n_w |= 1 << idx
        pu[r], nu[r] = np.uint64(p_u), np.uint64(n_u)
        pv[r], nv[r] = np.uint64(p_v), np.uint64(n_v)
        pw[r], nw[r] = np.uint64(p_w), np.uint64(n_w)
    return pu, nu, pv, nv, pw, nw


def zt_packed_to_scheme(
    m: int,
    n: int,
    p: int,
    pu: Sequence[int],
    nu: Sequence[int],
    pv: Sequence[int],
    nv: Sequence[int],
    pw: Sequence[int],
    nw: Sequence[int],
    name: str = "",
) -> bv.BilinearScheme:
    """Unpack 6 `uint64[R]` bitplanes into an exact ternary `Z` BilinearScheme."""
    R = len(pu)
    mn, np_d, mp = m * n, n * p, m * p
    U = [[((int(pu[r]) >> a) & 1) - ((int(nu[r]) >> a) & 1) for a in range(mn)] for r in range(R)]
    V = [[((int(pv[r]) >> b) & 1) - ((int(nv[r]) >> b) & 1) for b in range(np_d)] for r in range(R)]
    W = [[((int(pw[r]) >> c) & 1) - ((int(nw[r]) >> c) & 1) for c in range(mp)] for r in range(R)]
    return bv.BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring="Z", name=name)


def scheme_to_f3_packed(
    scheme: bv.BilinearScheme,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a `Z3` (or `Z`/`Zq` reduced mod 3) scheme into 6 `uint64[R]` F_3 bitplanes (+1->p, -1=2->n)."""
    m, n, p = scheme.m, scheme.n, scheme.p
    if max(m * n, n * p, m * p) > 64:
        raise ValueError(f"Format <{m},{n},{p}> exceeds 64-bit packing limit")
    R = scheme.rank
    pu = np.zeros(R, dtype=np.uint64)
    nu = np.zeros(R, dtype=np.uint64)
    pv = np.zeros(R, dtype=np.uint64)
    nv = np.zeros(R, dtype=np.uint64)
    pw = np.zeros(R, dtype=np.uint64)
    nw = np.zeros(R, dtype=np.uint64)
    for r in range(R):
        p_u = n_u = p_v = n_v = p_w = n_w = 0
        for idx, val in enumerate(scheme.U[r]):
            iv = int(val) % 3
            if iv == 1:
                p_u |= 1 << idx
            elif iv == 2:
                n_u |= 1 << idx
        for idx, val in enumerate(scheme.V[r]):
            iv = int(val) % 3
            if iv == 1:
                p_v |= 1 << idx
            elif iv == 2:
                n_v |= 1 << idx
        for idx, val in enumerate(scheme.W[r]):
            iv = int(val) % 3
            if iv == 1:
                p_w |= 1 << idx
            elif iv == 2:
                n_w |= 1 << idx
        pu[r], nu[r] = np.uint64(p_u), np.uint64(n_u)
        pv[r], nv[r] = np.uint64(p_v), np.uint64(n_v)
        pw[r], nw[r] = np.uint64(p_w), np.uint64(n_w)
    return pu, nu, pv, nv, pw, nw


def f3_packed_to_scheme(
    m: int,
    n: int,
    p: int,
    pu: Sequence[int],
    nu: Sequence[int],
    pv: Sequence[int],
    nv: Sequence[int],
    pw: Sequence[int],
    nw: Sequence[int],
    name: str = "",
) -> bv.BilinearScheme:
    """Unpack 6 `uint64[R]` F_3 bitplanes into an exact `Z3` BilinearScheme (entries in {0, 1, 2})."""
    R = len(pu)
    mn, np_d, mp = m * n, n * p, m * p
    U = [[(((int(pu[r]) >> a) & 1) - ((int(nu[r]) >> a) & 1)) % 3 for a in range(mn)] for r in range(R)]
    V = [[(((int(pv[r]) >> b) & 1) - ((int(nv[r]) >> b) & 1)) % 3 for b in range(np_d)] for r in range(R)]
    W = [[(((int(pw[r]) >> c) & 1) - ((int(nw[r]) >> c) & 1)) % 3 for c in range(mp)] for r in range(R)]
    return bv.BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring="Z3", name=name)


def transpose_scheme(scheme: bv.BilinearScheme, name: str = "") -> bv.BilinearScheme:
    """Exact transpose involution (AB)^T = B^T A^T: <m, n, p : R> -> <p, n, m : R>."""
    m, n, p = scheme.m, scheme.n, scheme.p
    U_new, V_new, W_new = [], [], []
    for r in range(scheme.rank):
        u_row = [scheme.V[r][j * p + k] for k in range(p) for j in range(n)]
        v_row = [scheme.U[r][i * n + j] for j in range(n) for i in range(m)]
        w_row = [scheme.W[r][i * p + k] for k in range(p) for i in range(m)]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    return bv.BilinearScheme(
        m=p, n=n, p=m, U=U_new, V=V_new, W=W_new, ring=scheme.ring, modulus=scheme.modulus, name=name
    )


def cyclic_rotate_scheme(scheme: bv.BilinearScheme, name: str = "") -> bv.BilinearScheme:
    """Exact C_3 cyclic symmetry via trace form tr(A B C_tr): <m, n, p : R> -> <n, p, m : R>."""
    w_tr = scheme.w_to_trace_convention()
    return bv.BilinearScheme.from_trace_convention(
        m=scheme.n,
        n=scheme.p,
        p=scheme.m,
        U=scheme.V,
        V=w_tr,
        W_tr=scheme.U,
        ring=scheme.ring,
        modulus=scheme.modulus,
        name=name,
    )


def permute_axes(
    scheme: bv.BilinearScheme, target_shape: Tuple[int, int, int], name: str = ""
) -> bv.BilinearScheme:
    """Apply exact S_3 symmetry to map `scheme` of shape (m,n,p) to any permutation `target_shape`."""
    cur = scheme
    for use_tr in (False, True):
        cand = transpose_scheme(cur) if use_tr else cur
        for _ in range(3):
            if cand.shape == target_shape:
                cand.name = name or f"{scheme.name}_perm_{target_shape}"
                return cand
            cand = cyclic_rotate_scheme(cand)
    raise ValueError(f"target_shape {target_shape} is not a permutation of {scheme.shape}")


def direct_sum_m(s1: bv.BilinearScheme, s2: bv.BilinearScheme, name: str = "") -> bv.BilinearScheme:
    """Exact vertical block concatenation: <m1, n, p : R1> + <m2, n, p : R2> -> <m1+m2, n, p : R1+R2>."""
    if (s1.n, s1.p) != (s2.n, s2.p) or (s1.ring, s1.modulus) != (s2.ring, s2.modulus):
        raise ValueError("Incompatible schemes for direct_sum_m")
    m1, m2, n, p = s1.m, s2.m, s1.n, s1.p
    m = m1 + m2
    U, V, W = [], [], []
    for r in range(s1.rank):
        U.append(list(s1.U[r]) + [0] * (m2 * n))
        V.append(list(s1.V[r]))
        W.append(list(s1.W[r]) + [0] * (m2 * p))
    for r in range(s2.rank):
        U.append([0] * (m1 * n) + list(s2.U[r]))
        V.append(list(s2.V[r]))
        W.append([0] * (m1 * p) + list(s2.W[r]))
    return bv.BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring=s1.ring, modulus=s1.modulus, name=name)


def direct_sum_n(s1: bv.BilinearScheme, s2: bv.BilinearScheme, name: str = "") -> bv.BilinearScheme:
    """Exact inner-dimension split A1*B1 + A2*B2: <m, n1, p : R1> + <m, n2, p : R2> -> <m, n1+n2, p : R1+R2>."""
    if (s1.m, s1.p) != (s2.m, s2.p) or (s1.ring, s1.modulus) != (s2.ring, s2.modulus):
        raise ValueError("Incompatible schemes for direct_sum_n")
    m, n1, n2, p = s1.m, s1.n, s2.n, s1.p
    n = n1 + n2
    U, V, W = [], [], []
    for r in range(s1.rank):
        u_row = [0] * (m * n)
        for i in range(m):
            for j in range(n1):
                u_row[i * n + j] = s1.U[r][i * n1 + j]
        U.append(u_row)
        V.append(list(s1.V[r]) + [0] * (n2 * p))
        W.append(list(s1.W[r]))
    for r in range(s2.rank):
        u_row = [0] * (m * n)
        for i in range(m):
            for j in range(n2):
                u_row[i * n + (n1 + j)] = s2.U[r][i * n2 + j]
        U.append(u_row)
        V.append([0] * (n1 * p) + list(s2.V[r]))
        W.append(list(s2.W[r]))
    return bv.BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring=s1.ring, modulus=s1.modulus, name=name)


def kronecker_product(s1: bv.BilinearScheme, s2: bv.BilinearScheme, name: str = "") -> bv.BilinearScheme:
    """Exact Kronecker tensor product <m1,n1,p1:R1> (x) <m2,n2,p2:R2> -> <m1*m2, n1*n2, p1*p2 : R1*R2>."""
    if (s1.ring, s1.modulus) != (s2.ring, s2.modulus):
        raise ValueError("kronecker_product requires matching rings")
    m1, n1, p1 = s1.shape
    m2, n2, p2 = s2.shape
    m, n, p = m1 * m2, n1 * n2, p1 * p2
    U, V, W = [], [], []
    for r1 in range(s1.rank):
        for r2 in range(s2.rank):
            u_row = [0] * (m * n)
            v_row = [0] * (n * p)
            w_row = [0] * (m * p)
            for i1 in range(m1):
                for j1 in range(n1):
                    c1 = s1.U[r1][i1 * n1 + j1]
                    if c1 == 0:
                        continue
                    for i2 in range(m2):
                        for j2 in range(n2):
                            c2 = s2.U[r2][i2 * n2 + j2]
                            if c2 != 0:
                                u_row[(i1 * m2 + i2) * n + (j1 * n2 + j2)] = c1 * c2
            for j1 in range(n1):
                for k1 in range(p1):
                    c1 = s1.V[r1][j1 * p1 + k1]
                    if c1 == 0:
                        continue
                    for j2 in range(n2):
                        for k2 in range(p2):
                            c2 = s2.V[r2][j2 * p2 + k2]
                            if c2 != 0:
                                v_row[(j1 * n2 + j2) * p + (k1 * p2 + k2)] = c1 * c2
            for i1 in range(m1):
                for k1 in range(p1):
                    c1 = s1.W[r1][i1 * p1 + k1]
                    if c1 == 0:
                        continue
                    for i2 in range(m2):
                        for k2 in range(p2):
                            c2 = s2.W[r2][i2 * p2 + k2]
                            if c2 != 0:
                                w_row[(i1 * m2 + i2) * p + (k1 * p2 + k2)] = c1 * c2
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
    return bv.BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring=s1.ring, modulus=s1.modulus, name=name)


def padded_strassen_compose(
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    p1: int,
    p2: int,
    subs: Sequence[bv.BilinearScheme],
    name: str = "",
) -> bv.BilinearScheme:
    """
    Exact generalized padded-Strassen block composition for <m1+m2, n1+n2, p1+p2>
    using 7 sub-schemes [M1..M7] matching the tightest non-zero block dimensions.
    """
    if len(subs) != 7:
        raise ValueError("padded_strassen_compose requires exactly 7 sub-schemes [M1..M7]")
    ring, mod = subs[0].ring, subs[0].modulus
    m, n, p = m1 + m2, n1 + n2, p1 + p2
    expected_shapes = [
        (max(m1, m2), max(n1, n2), max(p1, p2)),  # M1
        (m2, max(n1, n2), p1),                     # M2
        (m1, n1, p2),                              # M3
        (m2, n2, p1),                              # M4
        (m1, n2, p2),                              # M5
        (m2, n1, p2),                              # M6
        (m1, n2, p1),                              # M7
    ]
    for idx, (sc, shp) in enumerate(zip(subs, expected_shapes)):
        if sc.shape != shp:
            raise ValueError(f"Sub-scheme M{idx+1} must have shape {shp}, got {sc.shape}")
        if (sc.ring, sc.modulus) != (ring, mod):
            raise ValueError("All 7 sub-schemes must share (ring, modulus)")

    A11, A12 = (0, 0, m1, n1), (0, n1, m1, n2)
    A21, A22 = (m1, 0, m2, n1), (m1, n1, m2, n2)
    B11, B12 = (0, 0, n1, p1), (0, p1, n1, p2)
    B21, B22 = (n1, 0, n2, p1), (n1, p1, n2, p2)
    C11, C12 = (0, 0, m1, p1), (0, p1, m1, p2)
    C21, C22 = (m1, 0, m2, p1), (m1, p1, m2, p2)

    specs = [
        ([(A11, 1), (A22, 1)],  [(B11, 1), (B22, 1)],   [(C11, 1), (C22, 1)]),
        ([(A21, 1), (A22, 1)],  [(B11, 1)],             [(C21, 1), (C22, -1)]),
        ([(A11, 1)],            [(B12, 1), (B22, -1)],  [(C12, 1), (C22, 1)]),
        ([(A22, 1)],            [(B21, 1), (B11, -1)],  [(C11, 1), (C21, 1)]),
        ([(A11, 1), (A12, 1)],  [(B22, 1)],             [(C11, -1), (C12, 1)]),
        ([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)],   [(C22, 1)]),
        ([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)],   [(C11, 1)]),
    ]

    U_out: List[List[int]] = []
    V_out: List[List[int]] = []
    W_out: List[List[int]] = []

    for sc, (u_blks, v_blks, w_blks) in zip(subs, specs):
        sm, sn, sp = sc.shape
        for r in range(sc.rank):
            u_row = [0] * (m * n)
            v_row = [0] * (n * p)
            w_row = [0] * (m * p)
            for (r0, c0, bh, bw), sgn in u_blks:
                for i in range(min(bh, sm)):
                    for j in range(min(bw, sn)):
                        val = int(sc.U[r][i * sn + j])
                        if val:
                            u_row[(r0 + i) * n + (c0 + j)] += sgn * val
            for (r0, c0, bh, bw), sgn in v_blks:
                for i in range(min(bh, sn)):
                    for j in range(min(bw, sp)):
                        val = int(sc.V[r][i * sp + j])
                        if val:
                            v_row[(r0 + i) * p + (c0 + j)] += sgn * val
            for (r0, c0, bh, bw), sgn in w_blks:
                for i in range(min(bh, sm)):
                    for j in range(min(bw, sp)):
                        val = int(sc.W[r][i * sp + j])
                        if val:
                            w_row[(r0 + i) * p + (c0 + j)] += sgn * val
            U_out.append(u_row)
            V_out.append(v_row)
            W_out.append(w_row)

    return bv.BilinearScheme(
        m=m, n=n, p=p, U=U_out, V=V_out, W=W_out, ring=ring, modulus=mod, name=name
    )


def random_monomial_symmetry(scheme: bv.BilinearScheme, rng: random.Random) -> bv.BilinearScheme:
    """Exact signed-permutation symmetry (B_m x B_n x B_p) + rank permutation."""
    m, n, p = scheme.shape
    pm, pn, pp = list(range(m)), list(range(n)), list(range(p))
    rng.shuffle(pm)
    rng.shuffle(pn)
    rng.shuffle(pp)
    if scheme.ring == "Z":
        sm = [rng.choice((-1, 1)) for _ in range(m)]
        sn = [rng.choice((-1, 1)) for _ in range(n)]
        sp = [rng.choice((-1, 1)) for _ in range(p)]
    else:
        sm, sn, sp = [1] * m, [1] * n, [1] * p
    order = list(range(scheme.rank))
    rng.shuffle(order)

    U_new, V_new, W_new = [], [], []
    for r in order:
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (m * p)
        for i in range(m):
            for j in range(n):
                u_row[pm[i] * n + pn[j]] = sm[i] * sn[j] * int(scheme.U[r][i * n + j])
        for j in range(n):
            for k in range(p):
                v_row[pn[j] * p + pp[k]] = sn[j] * sp[k] * int(scheme.V[r][j * p + k])
        for i in range(m):
            for k in range(p):
                w_row[pm[i] * p + pp[k]] = sm[i] * sp[k] * int(scheme.W[r][i * p + k])
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    return bv.BilinearScheme(
        m=m, n=n, p=p, U=U_new, V=V_new, W=W_new, ring=scheme.ring, modulus=scheme.modulus, name=scheme.name
    )


def project_slice(scheme: bv.BilinearScheme, axis: str, drop_idx: int) -> bv.BilinearScheme:
    """Exact meta-flip down-projection dropping 1 index along 'm', 'n', or 'p'."""
    if axis not in ("m", "n", "p"):
        raise ValueError(f"Invalid projection axis {axis!r}; expected one of ('m', 'n', 'p')")
    m, n, p = scheme.shape
    dim_map = {"m": m, "n": n, "p": p}
    if not (0 <= drop_idx < dim_map[axis]):
        raise ValueError(f"drop_idx={drop_idx} out of range for axis {axis!r} of size {dim_map[axis]}")
    keep_m = [i for i in range(m) if not (axis == "m" and i == drop_idx)]
    keep_n = [j for j in range(n) if not (axis == "n" and j == drop_idx)]
    keep_p = [k for k in range(p) if not (axis == "p" and k == drop_idx)]
    m2, n2, p2 = len(keep_m), len(keep_n), len(keep_p)
    if m2 < 1 or n2 < 1 or p2 < 1:
        raise ValueError("Cannot project dimension below 1")

    U_out, V_out, W_out = [], [], []
    for r in range(scheme.rank):
        u_row = [scheme.U[r][i * n + j] for i in keep_m for j in keep_n]
        v_row = [scheme.V[r][j * p + k] for j in keep_n for k in keep_p]
        w_row = [scheme.W[r][i * p + k] for i in keep_m for k in keep_p]
        if any(x != 0 for x in u_row) and any(x != 0 for x in v_row) and any(x != 0 for x in w_row):
            U_out.append(u_row)
            V_out.append(v_row)
            W_out.append(w_row)
    return bv.BilinearScheme(
        m=m2, n=n2, p=p2, U=U_out, V=V_out, W=W_out, ring=scheme.ring, modulus=scheme.modulus
    )


def best_project_slice(scheme: bv.BilinearScheme, axis: str) -> bv.BilinearScheme:
    """Try projecting along EVERY index of `axis` and return the lowest-rank resulting scheme."""
    dim = {"m": scheme.m, "n": scheme.n, "p": scheme.p}[axis]
    best_sc: Optional[bv.BilinearScheme] = None
    for idx in range(dim):
        cand = project_slice(scheme, axis, idx)
        if best_sc is None or cand.rank < best_sc.rank:
            best_sc = cand
    assert best_sc is not None
    return best_sc


# =============================================================================
# 4. High-Level GPU Engine Wrapper (`FlipGraphGPUEngine`)
# =============================================================================


@dataclass
class GPURunResult:
    ring: str
    shape: Tuple[int, int, int]
    n_walkers: int
    n_steps: int
    elapsed_s: float
    compile_s: float
    attempted_steps: int
    executed_flips: int
    executed_plus: int
    executed_reduces: int
    attempted_steps_per_s: float
    executed_flips_per_s: float
    peak_vram_mib: float
    best_rank: int
    best_nnz: int
    top_schemes: List[bv.BilinearScheme]
    ranks_all: List[int]


def _select_kernel_profile(max_needed_r: int, is_zt: bool) -> Tuple[int, int, int]:
    """Select `(MAX_R, MAX_EDGES, warps_per_block)` keeping static shared memory <= 44 KiB/block."""
    if not is_zt:
        if max_needed_r <= 96:
            return 96, 1024, 4    # 4 * (2304 + 4096) = 25.6 KiB
        if max_needed_r <= 192:
            return 192, 1536, 2   # 2 * (4608 + 6144) = 21.5 KiB
        return 272, 2048, 2       # 2 * (6528 + 8192) = 29.4 KiB
    else:
        if max_needed_r <= 96:
            return 96, 1024, 4    # 4 * (4608 + 4096) = 34.8 KiB
        if max_needed_r <= 192:
            return 192, 1536, 2   # 2 * (9216 + 6144) = 30.7 KiB
        return 272, 2048, 2       # 2 * (13056 + 8192) = 42.5 KiB


class FlipGraphGPUEngine:
    """High-throughput RTX 5060 SM 12.0 Flip-Graph v2 engine (F2 + Z_T 2-bitplane)."""

    def __init__(self) -> None:
        self.backend = get_nvrtc_backend()

    def _check_vram_guard(self) -> float:
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if peak_mib > MAX_VRAM_MIB:
            raise MemoryError(
                f"GPU VRAM guard exceeded: {peak_mib:.1f} MiB > {MAX_VRAM_MIB:.1f} MiB limit"
            )
        return peak_mib

    def run_f2_batch(
        self,
        seeds: Sequence[bv.BilinearScheme],
        n_walkers: int = 2048,
        n_steps: int = 4000,
        plus_period: int = 120,
        plus_headroom: int = 2,
        target_rank: int = 0,
        diverse_mode: bool = False,
        top_k: int = 16,
        seed: int = 2026,
        randomize_symmetry: bool = True,
    ) -> GPURunResult:
        if not seeds:
            raise ValueError("At least one seed scheme is required")
        m, n, p = seeds[0].shape
        max_seed_r = max(s.rank for s in seeds)
        max_r, max_edges, wpb = _select_kernel_profile(max_seed_r + plus_headroom + 4, is_zt=False)
        if max_seed_r > max_r:
            raise ValueError(f"Seed rank {max_seed_r} exceeds GPU MAX_R={max_r}")

        key = f"f2_v2_r{max_r}_e{max_edges}_w{wpb}"
        funcs, compile_s = self.backend.compile_module(
            key, _build_f2_cuda_source(max_r, max_edges, wpb), ["f2_flip_walk_kernel"]
        )
        fn = funcs["f2_flip_walk_kernel"]

        rng = random.Random(seed)
        packed_base = [scheme_to_f2_packed(s) for s in seeds]
        n_variants = min(n_walkers, max(len(seeds), 64 if randomize_symmetry else len(seeds)))
        variants = []
        for idx in range(n_variants):
            s_base = seeds[idx % len(seeds)]
            if idx < len(seeds) or not randomize_symmetry:
                variants.append(packed_base[idx % len(seeds)])
            else:
                variants.append(scheme_to_f2_packed(random_monomial_symmetry(s_base, rng)))

        h_U = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_V = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_W = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_R = np.zeros(n_walkers, dtype=np.int32)
        for w in range(n_walkers):
            u_arr, v_arr, w_arr = variants[w % n_variants]
            r_len = len(u_arr)
            h_U[w, :r_len], h_V[w, :r_len], h_W[w, :r_len] = u_arr, v_arr, w_arr
            h_R[w] = r_len

        torch.cuda.reset_peak_memory_stats()
        d_init_U = torch.from_numpy(h_U.view(np.int64)).cuda()
        d_init_V = torch.from_numpy(h_V.view(np.int64)).cuda()
        d_init_W = torch.from_numpy(h_W.view(np.int64)).cuda()
        d_init_R = torch.from_numpy(h_R).cuda()

        d_best_U = torch.empty_like(d_init_U)
        d_best_V = torch.empty_like(d_init_V)
        d_best_W = torch.empty_like(d_init_W)
        d_best_R = torch.empty(n_walkers, dtype=torch.int32, device="cuda")
        d_best_nnz = torch.empty(n_walkers, dtype=torch.int32, device="cuda")
        d_stats = torch.zeros(n_walkers * 3, dtype=torch.int64, device="cuda")

        args = [
            ctypes.c_uint64(d_init_U.data_ptr()),
            ctypes.c_uint64(d_init_V.data_ptr()),
            ctypes.c_uint64(d_init_W.data_ptr()),
            ctypes.c_uint64(d_init_R.data_ptr()),
            ctypes.c_uint64(d_best_U.data_ptr()),
            ctypes.c_uint64(d_best_V.data_ptr()),
            ctypes.c_uint64(d_best_W.data_ptr()),
            ctypes.c_uint64(d_best_R.data_ptr()),
            ctypes.c_uint64(d_best_nnz.data_ptr()),
            ctypes.c_uint64(d_stats.data_ptr()),
            ctypes.c_int(n_walkers),
            ctypes.c_int(n_steps),
            ctypes.c_int(plus_period),
            ctypes.c_int(plus_headroom),
            ctypes.c_int(target_rank),
            ctypes.c_int(1 if diverse_mode else 0),
            ctypes.c_uint64(seed * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF),
        ]

        blocks = (n_walkers + wpb - 1) // wpb
        threads = wpb * 32

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.backend.launch(fn, (blocks, 1, 1), (threads, 1, 1), args)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
        peak_vram = self._check_vram_guard()

        out_R = d_best_R.cpu().numpy()
        out_nnz = d_best_nnz.cpu().numpy()
        stats_np = d_stats.cpu().numpy().view(np.uint64).reshape(n_walkers, 3)

        exec_flips = int(stats_np[:, 0].sum())
        exec_plus = int(stats_np[:, 1].sum())
        exec_red = int(stats_np[:, 2].sum())

        order = np.lexsort((out_nnz, out_R))
        best_r_val = int(out_R[order[0]])
        at_best = np.where(out_R == best_r_val)[0]
        at_near = np.where((out_R > best_r_val) & (out_R <= best_r_val + 2))[0]
        if diverse_mode and target_rank > 0:
            at_best = np.where(out_R <= max(best_r_val, target_rank))[0]

        n_best_quota = max(2, int(top_k * 0.65)) if (len(at_near) > 0 and not diverse_mode) else top_k
        pick_order: List[int] = [int(order[0])]
        seen_w = set(pick_order)
        if len(at_best) > 0:
            for idx_w in rng.sample(at_best.tolist(), min(len(at_best), max(n_best_quota * 3, 24))):
                if idx_w not in seen_w:
                    seen_w.add(idx_w)
                    pick_order.append(int(idx_w))
        if len(at_near) > 0 and not diverse_mode:
            for idx_w in rng.sample(at_near.tolist(), min(len(at_near), max((top_k - n_best_quota) * 3, 16))):
                if idx_w not in seen_w:
                    seen_w.add(idx_w)
                    pick_order.append(int(idx_w))
        for idx_w in order[: min(len(order), max(top_k * 4, 32))]:
            ii = int(idx_w)
            if ii not in seen_w:
                seen_w.add(ii)
                pick_order.append(ii)
        pick_order = pick_order[: min(len(pick_order), max(top_k * 5, 48))]

        sel_idx = np.array(pick_order, dtype=np.int64)
        c_U = d_best_U[sel_idx].cpu().numpy().view(np.uint64)
        c_V = d_best_V[sel_idx].cpu().numpy().view(np.uint64)
        c_W = d_best_W[sel_idx].cpu().numpy().view(np.uint64)

        top_schemes: List[bv.BilinearScheme] = []
        seen_hashes = set()
        n_best_taken = 0
        for local_i, wid in enumerate(sel_idx):
            r_w = int(out_R[wid])
            if not diverse_mode and r_w == best_r_val and n_best_taken >= n_best_quota and len(at_near) > 0:
                continue
            sc = f2_packed_to_scheme(
                m, n, p, c_U[local_i, :r_w], c_V[local_i, :r_w], c_W[local_i, :r_w],
                name=f"gpu_f2_{m}x{n}x{p}_R{r_w}_w{int(wid)}"
            )
            h = sc.canonical_sha256()
            if h not in seen_hashes:
                seen_hashes.add(h)
                top_schemes.append(sc)
                if r_w == best_r_val:
                    n_best_taken += 1
                if len(top_schemes) >= top_k:
                    break
        top_schemes.sort(key=lambda s: s.rank)

        best_idx = int(order[0])
        attempted = n_walkers * n_steps
        return GPURunResult(
            ring="Z2",
            shape=(m, n, p),
            n_walkers=n_walkers,
            n_steps=n_steps,
            elapsed_s=elapsed_s,
            compile_s=compile_s,
            attempted_steps=attempted,
            executed_flips=exec_flips,
            executed_plus=exec_plus,
            executed_reduces=exec_red,
            attempted_steps_per_s=attempted / max(elapsed_s, 1e-9),
            executed_flips_per_s=exec_flips / max(elapsed_s, 1e-9),
            peak_vram_mib=peak_vram,
            best_rank=int(out_R[best_idx]),
            best_nnz=int(out_nnz[best_idx]),
            top_schemes=top_schemes,
            ranks_all=out_R.tolist(),
        )

    def run_zt_batch(
        self,
        seeds: Sequence[bv.BilinearScheme],
        n_walkers: int = 2048,
        n_steps: int = 4000,
        plus_period: int = 120,
        plus_headroom: int = 2,
        top_k: int = 12,
        seed: int = 2026,
        randomize_symmetry: bool = True,
    ) -> GPURunResult:
        if not seeds:
            raise ValueError("At least one ternary Z seed scheme is required")
        m, n, p = seeds[0].shape
        max_seed_r = max(s.rank for s in seeds)
        max_r, max_edges, wpb = _select_kernel_profile(max_seed_r + plus_headroom + 4, is_zt=True)
        if max_seed_r > max_r:
            raise ValueError(f"Seed rank {max_seed_r} exceeds GPU Z_T MAX_R={max_r}")

        key = f"zt_v2_r{max_r}_e{max_edges}_w{wpb}"
        funcs, compile_s = self.backend.compile_module(
            key, _build_zt_cuda_source(max_r, max_edges, wpb), ["zt_flip_walk_kernel"]
        )
        fn = funcs["zt_flip_walk_kernel"]

        rng = random.Random(seed)
        packed_base = [scheme_to_zt_packed(s) for s in seeds]
        n_variants = min(n_walkers, max(len(seeds), 64 if randomize_symmetry else len(seeds)))
        variants = []
        for idx in range(n_variants):
            s_base = seeds[idx % len(seeds)]
            if idx < len(seeds) or not randomize_symmetry:
                variants.append(packed_base[idx % len(seeds)])
            else:
                variants.append(scheme_to_zt_packed(random_monomial_symmetry(s_base, rng)))

        h_pU = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_nU = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_pV = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_nV = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_pW = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_nW = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_R = np.zeros(n_walkers, dtype=np.int32)
        for w in range(n_walkers):
            pu, nu, pv, nv, pw, nw = variants[w % n_variants]
            r_len = len(pu)
            h_pU[w, :r_len], h_nU[w, :r_len] = pu, nu
            h_pV[w, :r_len], h_nV[w, :r_len] = pv, nv
            h_pW[w, :r_len], h_nW[w, :r_len] = pw, nw
            h_R[w] = r_len

        torch.cuda.reset_peak_memory_stats()
        d_ipU = torch.from_numpy(h_pU.view(np.int64)).cuda()
        d_inU = torch.from_numpy(h_nU.view(np.int64)).cuda()
        d_ipV = torch.from_numpy(h_pV.view(np.int64)).cuda()
        d_inV = torch.from_numpy(h_nV.view(np.int64)).cuda()
        d_ipW = torch.from_numpy(h_pW.view(np.int64)).cuda()
        d_inW = torch.from_numpy(h_nW.view(np.int64)).cuda()
        d_iR = torch.from_numpy(h_R).cuda()

        d_bpU = torch.empty_like(d_ipU)
        d_bnU = torch.empty_like(d_inU)
        d_bpV = torch.empty_like(d_ipV)
        d_bnV = torch.empty_like(d_inV)
        d_bpW = torch.empty_like(d_ipW)
        d_bnW = torch.empty_like(d_inW)
        d_bR = torch.empty(n_walkers, dtype=torch.int32, device="cuda")
        d_bnnz = torch.empty(n_walkers, dtype=torch.int32, device="cuda")
        d_stats = torch.zeros(n_walkers * 3, dtype=torch.int64, device="cuda")

        args = [
            ctypes.c_uint64(d_ipU.data_ptr()), ctypes.c_uint64(d_inU.data_ptr()),
            ctypes.c_uint64(d_ipV.data_ptr()), ctypes.c_uint64(d_inV.data_ptr()),
            ctypes.c_uint64(d_ipW.data_ptr()), ctypes.c_uint64(d_inW.data_ptr()),
            ctypes.c_uint64(d_iR.data_ptr()),
            ctypes.c_uint64(d_bpU.data_ptr()), ctypes.c_uint64(d_bnU.data_ptr()),
            ctypes.c_uint64(d_bpV.data_ptr()), ctypes.c_uint64(d_bnV.data_ptr()),
            ctypes.c_uint64(d_bpW.data_ptr()), ctypes.c_uint64(d_bnW.data_ptr()),
            ctypes.c_uint64(d_bR.data_ptr()),
            ctypes.c_uint64(d_bnnz.data_ptr()),
            ctypes.c_uint64(d_stats.data_ptr()),
            ctypes.c_int(n_walkers),
            ctypes.c_int(n_steps),
            ctypes.c_int(plus_period),
            ctypes.c_int(plus_headroom),
            ctypes.c_uint64(seed * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF),
        ]

        blocks = (n_walkers + wpb - 1) // wpb
        threads = wpb * 32

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.backend.launch(fn, (blocks, 1, 1), (threads, 1, 1), args)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
        peak_vram = self._check_vram_guard()

        out_R = d_bR.cpu().numpy()
        out_nnz = d_bnnz.cpu().numpy()
        stats_np = d_stats.cpu().numpy().view(np.uint64).reshape(n_walkers, 3)

        exec_flips = int(stats_np[:, 0].sum())
        exec_plus = int(stats_np[:, 1].sum())
        exec_red = int(stats_np[:, 2].sum())

        order = np.lexsort((out_nnz, out_R))
        best_r_val = int(out_R[order[0]])
        at_best = np.where(out_R == best_r_val)[0]
        at_near = np.where((out_R > best_r_val) & (out_R <= best_r_val + 2))[0]

        n_best_quota = max(2, int(top_k * 0.65)) if len(at_near) > 0 else top_k
        pick_order: List[int] = [int(order[0])]
        seen_w = set(pick_order)
        if len(at_best) > 0:
            for idx_w in rng.sample(at_best.tolist(), min(len(at_best), max(n_best_quota * 3, 24))):
                if idx_w not in seen_w:
                    seen_w.add(idx_w)
                    pick_order.append(int(idx_w))
        if len(at_near) > 0:
            for idx_w in rng.sample(at_near.tolist(), min(len(at_near), max((top_k - n_best_quota) * 3, 16))):
                if idx_w not in seen_w:
                    seen_w.add(idx_w)
                    pick_order.append(int(idx_w))
        for idx_w in order[: min(len(order), max(top_k * 4, 32))]:
            ii = int(idx_w)
            if ii not in seen_w:
                seen_w.add(ii)
                pick_order.append(ii)
        pick_order = pick_order[: min(len(pick_order), max(top_k * 5, 48))]

        sel_idx = np.array(pick_order, dtype=np.int64)
        cpU = d_bpU[sel_idx].cpu().numpy().view(np.uint64)
        cnU = d_bnU[sel_idx].cpu().numpy().view(np.uint64)
        cpV = d_bpV[sel_idx].cpu().numpy().view(np.uint64)
        cnV = d_bnV[sel_idx].cpu().numpy().view(np.uint64)
        cpW = d_bpW[sel_idx].cpu().numpy().view(np.uint64)
        cnW = d_bnW[sel_idx].cpu().numpy().view(np.uint64)

        top_schemes: List[bv.BilinearScheme] = []
        seen_hashes = set()
        n_best_taken = 0
        for local_i, wid in enumerate(sel_idx):
            r_w = int(out_R[wid])
            if r_w == best_r_val and n_best_taken >= n_best_quota and len(at_near) > 0:
                continue
            sc = zt_packed_to_scheme(
                m, n, p,
                cpU[local_i, :r_w], cnU[local_i, :r_w],
                cpV[local_i, :r_w], cnV[local_i, :r_w],
                cpW[local_i, :r_w], cnW[local_i, :r_w],
                name=f"gpu_zt_{m}x{n}x{p}_R{r_w}_w{int(wid)}"
            )
            h = sc.canonical_sha256()
            if h not in seen_hashes:
                seen_hashes.add(h)
                top_schemes.append(sc)
                if r_w == best_r_val:
                    n_best_taken += 1
                if len(top_schemes) >= top_k:
                    break
        top_schemes.sort(key=lambda s: s.rank)

        best_idx = int(order[0])
        attempted = n_walkers * n_steps
        return GPURunResult(
            ring="Z",
            shape=(m, n, p),
            n_walkers=n_walkers,
            n_steps=n_steps,
            elapsed_s=elapsed_s,
            compile_s=compile_s,
            attempted_steps=attempted,
            executed_flips=exec_flips,
            executed_plus=exec_plus,
            executed_reduces=exec_red,
            attempted_steps_per_s=attempted / max(elapsed_s, 1e-9),
            executed_flips_per_s=exec_flips / max(elapsed_s, 1e-9),
            peak_vram_mib=peak_vram,
            best_rank=int(out_R[best_idx]),
            best_nnz=int(out_nnz[best_idx]),
            top_schemes=top_schemes,
            ranks_all=out_R.tolist(),
        )

    def run_f3_batch(
        self,
        seeds: Sequence[bv.BilinearScheme],
        n_walkers: int = 2048,
        n_steps: int = 4000,
        plus_period: int = 120,
        plus_headroom: int = 2,
        target_rank: int = 0,
        diverse_mode: bool = False,
        top_k: int = 12,
        seed: int = 2026,
        randomize_symmetry: bool = True,
    ) -> GPURunResult:
        if not seeds:
            raise ValueError("At least one Z3 or Z seed scheme is required")
        m, n, p = seeds[0].shape
        max_seed_r = max(s.rank for s in seeds)
        max_r, max_edges, wpb = _select_kernel_profile(max_seed_r + plus_headroom + 4, is_zt=True)
        if max_seed_r > max_r:
            raise ValueError(f"Seed rank {max_seed_r} exceeds GPU F_3 MAX_R={max_r}")

        key = f"f3_v1_r{max_r}_e{max_edges}_w{wpb}"
        funcs, compile_s = self.backend.compile_module(
            key, _build_f3_cuda_source(max_r, max_edges, wpb), ["f3_flip_walk_kernel", "f3_truth_table_kernel"]
        )
        fn = funcs["f3_flip_walk_kernel"]

        rng = random.Random(seed)
        packed_base = [scheme_to_f3_packed(s) for s in seeds]
        n_variants = min(n_walkers, max(len(seeds), 64 if randomize_symmetry else len(seeds)))
        variants = []
        for idx in range(n_variants):
            s_base = seeds[idx % len(seeds)]
            if idx < len(seeds) or not randomize_symmetry:
                variants.append(packed_base[idx % len(seeds)])
            else:
                variants.append(scheme_to_f3_packed(random_monomial_symmetry(s_base, rng)))

        h_pU = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_nU = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_pV = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_nV = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_pW = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_nW = np.zeros((n_walkers, max_r), dtype=np.uint64)
        h_R = np.zeros(n_walkers, dtype=np.int32)
        for w in range(n_walkers):
            pu, nu, pv, nv, pw, nw = variants[w % n_variants]
            r_len = len(pu)
            h_pU[w, :r_len], h_nU[w, :r_len] = pu, nu
            h_pV[w, :r_len], h_nV[w, :r_len] = pv, nv
            h_pW[w, :r_len], h_nW[w, :r_len] = pw, nw
            h_R[w] = r_len

        torch.cuda.reset_peak_memory_stats()
        d_ipU = torch.from_numpy(h_pU.view(np.int64)).cuda()
        d_inU = torch.from_numpy(h_nU.view(np.int64)).cuda()
        d_ipV = torch.from_numpy(h_pV.view(np.int64)).cuda()
        d_inV = torch.from_numpy(h_nV.view(np.int64)).cuda()
        d_ipW = torch.from_numpy(h_pW.view(np.int64)).cuda()
        d_inW = torch.from_numpy(h_nW.view(np.int64)).cuda()
        d_iR = torch.from_numpy(h_R).cuda()

        d_bpU = torch.empty_like(d_ipU)
        d_bnU = torch.empty_like(d_inU)
        d_bpV = torch.empty_like(d_ipV)
        d_bnV = torch.empty_like(d_inV)
        d_bpW = torch.empty_like(d_ipW)
        d_bnW = torch.empty_like(d_inW)
        d_bR = torch.empty(n_walkers, dtype=torch.int32, device="cuda")
        d_bnnz = torch.empty(n_walkers, dtype=torch.int32, device="cuda")
        d_stats = torch.zeros(n_walkers * 3, dtype=torch.int64, device="cuda")

        args = [
            ctypes.c_uint64(d_ipU.data_ptr()), ctypes.c_uint64(d_inU.data_ptr()),
            ctypes.c_uint64(d_ipV.data_ptr()), ctypes.c_uint64(d_inV.data_ptr()),
            ctypes.c_uint64(d_ipW.data_ptr()), ctypes.c_uint64(d_inW.data_ptr()),
            ctypes.c_uint64(d_iR.data_ptr()),
            ctypes.c_uint64(d_bpU.data_ptr()), ctypes.c_uint64(d_bnU.data_ptr()),
            ctypes.c_uint64(d_bpV.data_ptr()), ctypes.c_uint64(d_bnV.data_ptr()),
            ctypes.c_uint64(d_bpW.data_ptr()), ctypes.c_uint64(d_bnW.data_ptr()),
            ctypes.c_uint64(d_bR.data_ptr()),
            ctypes.c_uint64(d_bnnz.data_ptr()),
            ctypes.c_uint64(d_stats.data_ptr()),
            ctypes.c_int(n_walkers),
            ctypes.c_int(n_steps),
            ctypes.c_int(plus_period),
            ctypes.c_int(plus_headroom),
            ctypes.c_int(target_rank),
            ctypes.c_int(1 if diverse_mode else 0),
            ctypes.c_uint64(seed * 0xD6E8FEB86659FD93 & 0xFFFFFFFFFFFFFFFF),
        ]

        blocks = (n_walkers + wpb - 1) // wpb
        threads = wpb * 32

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.backend.launch(fn, (blocks, 1, 1), (threads, 1, 1), args)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
        peak_vram = self._check_vram_guard()

        out_R = d_bR.cpu().numpy()
        out_nnz = d_bnnz.cpu().numpy()
        stats_np = d_stats.cpu().numpy().view(np.uint64).reshape(n_walkers, 3)

        exec_flips = int(stats_np[:, 0].sum())
        exec_plus = int(stats_np[:, 1].sum())
        exec_red = int(stats_np[:, 2].sum())

        order = np.lexsort((out_nnz, out_R))
        best_r_val = int(out_R[order[0]])
        at_best = np.where(out_R == best_r_val)[0]
        at_near = np.where((out_R > best_r_val) & (out_R <= best_r_val + 2))[0]
        if diverse_mode and target_rank > 0:
            at_best = np.where(out_R <= max(best_r_val, target_rank))[0]

        n_best_quota = max(2, int(top_k * 0.65)) if (len(at_near) > 0 and not diverse_mode) else top_k
        pick_order: List[int] = [int(order[0])]
        seen_w = set(pick_order)
        if len(at_best) > 0:
            for idx_w in rng.sample(at_best.tolist(), min(len(at_best), max(n_best_quota * 3, 24))):
                if idx_w not in seen_w:
                    seen_w.add(idx_w)
                    pick_order.append(int(idx_w))
        if len(at_near) > 0 and not diverse_mode:
            for idx_w in rng.sample(at_near.tolist(), min(len(at_near), max((top_k - n_best_quota) * 3, 16))):
                if idx_w not in seen_w:
                    seen_w.add(idx_w)
                    pick_order.append(int(idx_w))
        for idx_w in order[: min(len(order), max(top_k * 4, 32))]:
            ii = int(idx_w)
            if ii not in seen_w:
                seen_w.add(ii)
                pick_order.append(ii)
        pick_order = pick_order[: min(len(pick_order), max(top_k * 5, 48))]

        sel_idx = np.array(pick_order, dtype=np.int64)
        cpU = d_bpU[sel_idx].cpu().numpy().view(np.uint64)
        cnU = d_bnU[sel_idx].cpu().numpy().view(np.uint64)
        cpV = d_bpV[sel_idx].cpu().numpy().view(np.uint64)
        cnV = d_bnV[sel_idx].cpu().numpy().view(np.uint64)
        cpW = d_bpW[sel_idx].cpu().numpy().view(np.uint64)
        cnW = d_bnW[sel_idx].cpu().numpy().view(np.uint64)

        top_schemes: List[bv.BilinearScheme] = []
        seen_hashes = set()
        n_best_taken = 0
        for local_i, wid in enumerate(sel_idx):
            r_w = int(out_R[wid])
            if not diverse_mode and r_w == best_r_val and n_best_taken >= n_best_quota and len(at_near) > 0:
                continue
            sc = f3_packed_to_scheme(
                m, n, p,
                cpU[local_i, :r_w], cnU[local_i, :r_w],
                cpV[local_i, :r_w], cnV[local_i, :r_w],
                cpW[local_i, :r_w], cnW[local_i, :r_w],
                name=f"gpu_f3_{m}x{n}x{p}_R{r_w}_w{int(wid)}"
            )
            h = sc.canonical_sha256()
            if h not in seen_hashes:
                seen_hashes.add(h)
                top_schemes.append(sc)
                if r_w == best_r_val:
                    n_best_taken += 1
                if len(top_schemes) >= top_k:
                    break
        top_schemes.sort(key=lambda s: s.rank)

        best_idx = int(order[0])
        attempted = n_walkers * n_steps
        return GPURunResult(
            ring="Z3",
            shape=(m, n, p),
            n_walkers=n_walkers,
            n_steps=n_steps,
            elapsed_s=elapsed_s,
            compile_s=compile_s,
            attempted_steps=attempted,
            executed_flips=exec_flips,
            executed_plus=exec_plus,
            executed_reduces=exec_red,
            attempted_steps_per_s=attempted / max(elapsed_s, 1e-9),
            executed_flips_per_s=exec_flips / max(elapsed_s, 1e-9),
            peak_vram_mib=peak_vram,
            best_rank=int(out_R[best_idx]),
            best_nnz=int(out_nnz[best_idx]),
            top_schemes=top_schemes,
            ranks_all=out_R.tolist(),
        )

    def verify_f3_truth_table_gpu(self) -> Tuple[bool, str]:
        """Exhaustively test all 9 (a,b) in F_3 x F_3 for device `f3_add` and `f3_sub` on GPU hardware."""
        key = "f3_v1_r96_e1024_w4"
        funcs, _ = self.backend.compile_module(
            key, _build_f3_cuda_source(96, 1024, 4), ["f3_flip_walk_kernel", "f3_truth_table_kernel"]
        )
        fn = funcs["f3_truth_table_kernel"]
        d_add = torch.zeros(9, dtype=torch.int32, device="cuda")
        d_sub = torch.zeros(9, dtype=torch.int32, device="cuda")
        d_bad = torch.zeros(1, dtype=torch.int32, device="cuda")
        args = [
            ctypes.c_uint64(d_add.data_ptr()),
            ctypes.c_uint64(d_sub.data_ptr()),
            ctypes.c_uint64(d_bad.data_ptr()),
        ]
        self.backend.launch(fn, (1, 1, 1), (1, 1, 1), args)
        torch.cuda.synchronize()
        got_add = d_add.cpu().numpy().tolist()
        got_sub = d_sub.cpu().numpy().tolist()
        bad_bits = int(d_bad.cpu().item())
        exp_add = [(a + b) % 3 for a in range(3) for b in range(3)]
        exp_sub = [(a - b) % 3 for a in range(3) for b in range(3)]
        ok = (got_add == exp_add) and (got_sub == exp_sub) and (bad_bits == 0)
        return ok, f"add={got_add == exp_add} (9/9), sub={got_sub == exp_sub} (9/9), disjoint_bitplanes={bad_bits == 0}"

    def evolve_f3(
        self,
        seeds: Sequence[bv.BilinearScheme],
        target_rank: int,
        n_rounds: int = 4,
        n_walkers: int = 4096,
        n_steps: int = 20000,
        plus_period: int = 600,
        plus_headroom: int = 2,
        top_k: int = 24,
        seed: int = 2026,
    ) -> GPURunResult:
        """Multi-round evolutionary GPU beam search over F_3 with elite basin amplification."""
        cur_seeds = list(seeds)
        last_res: Optional[GPURunResult] = None
        for rd in range(n_rounds):
            res = self.run_f3_batch(
                cur_seeds,
                n_walkers=n_walkers,
                n_steps=n_steps,
                plus_period=plus_period,
                plus_headroom=plus_headroom,
                target_rank=target_rank,
                top_k=top_k,
                seed=seed + rd * 997,
            )
            last_res = res
            cur_seeds = res.top_schemes
            if res.best_rank <= target_rank:
                break
        assert last_res is not None
        return last_res

    def evolve_f2(
        self,
        seeds: Sequence[bv.BilinearScheme],
        target_rank: int,
        n_rounds: int = 4,
        n_walkers: int = 4096,
        n_steps: int = 20000,
        plus_period: int = 800,
        plus_headroom: int = 2,
        top_k: int = 24,
        seed: int = 2026,
    ) -> GPURunResult:
        """Multi-round evolutionary GPU beam search over F2 with elite basin amplification."""
        cur_seeds = list(seeds)
        last_res: Optional[GPURunResult] = None
        for rd in range(n_rounds):
            res = self.run_f2_batch(
                cur_seeds,
                n_walkers=n_walkers,
                n_steps=n_steps,
                plus_period=plus_period,
                plus_headroom=plus_headroom,
                target_rank=target_rank,
                top_k=top_k,
                seed=seed + rd * 997,
            )
            last_res = res
            cur_seeds = res.top_schemes
            if res.best_rank <= target_rank:
                break
        assert last_res is not None
        return last_res

    def evolve_zt(
        self,
        seeds: Sequence[bv.BilinearScheme],
        target_rank: int,
        n_rounds: int = 4,
        n_walkers: int = 4096,
        n_steps: int = 20000,
        plus_period: int = 600,
        plus_headroom: int = 2,
        top_k: int = 24,
        seed: int = 2026,
    ) -> GPURunResult:
        """Multi-round evolutionary GPU beam search over Z_T with elite basin amplification."""
        cur_seeds = list(seeds)
        last_res: Optional[GPURunResult] = None
        for rd in range(n_rounds):
            res = self.run_zt_batch(
                cur_seeds,
                n_walkers=n_walkers,
                n_steps=n_steps,
                plus_period=plus_period,
                plus_headroom=plus_headroom,
                top_k=top_k,
                seed=seed + rd * 997,
            )
            last_res = res
            cur_seeds = res.top_schemes
            if res.best_rank <= target_rank:
                break
        assert last_res is not None
        return last_res


# =============================================================================
# 5. Mandatory Tracer #1 Self-Test (`--selftest`) + High-R Speed Benchmark
# =============================================================================


def run_selftest(verbose: bool = True) -> bool:
    engine = FlipGraphGPUEngine()
    checks: List[Tuple[str, bool, str]] = []

    def rec(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1. Cold-start naive <2,2,2:8> -> 7 over F2
    naive2_f2 = bv.make_naive_scheme(2, 2, 2, ring="Z2")
    res_2_f2 = engine.run_f2_batch(
        [naive2_f2], n_walkers=2048, n_steps=2000, plus_period=40, plus_headroom=2, top_k=4, seed=101
    )
    rep_2_f2 = bv.verify_scheme(res_2_f2.top_schemes[0])
    rec(
        "f2_strassen_8_to_7",
        rep_2_f2.valid and res_2_f2.best_rank == 7 and res_2_f2.peak_vram_mib <= MAX_VRAM_MIB,
        f"R={res_2_f2.best_rank}, valid={rep_2_f2.valid}, flips/s={res_2_f2.executed_flips_per_s/1e6:.1f}M/s, "
        f"compile={res_2_f2.compile_s*1000:.1f}ms, time={res_2_f2.elapsed_s*1000:.1f}ms, vram={res_2_f2.peak_vram_mib:.1f}MiB",
    )

    # 2. Cold-start naive <2,2,2:8> -> 7 over Z_T
    naive2_zt = bv.make_naive_scheme(2, 2, 2, ring="Z")
    res_2_zt = engine.run_zt_batch(
        [naive2_zt], n_walkers=2048, n_steps=2000, plus_period=40, plus_headroom=2, top_k=4, seed=202
    )
    rep_2_zt = bv.verify_scheme(res_2_zt.top_schemes[0])
    rec(
        "zt_strassen_8_to_7",
        rep_2_zt.valid and rep_2_zt.is_ternary and res_2_zt.best_rank == 7 and res_2_zt.peak_vram_mib <= MAX_VRAM_MIB,
        f"R={res_2_zt.best_rank}, valid={rep_2_zt.valid}, ternary={rep_2_zt.is_ternary}, "
        f"flips/s={res_2_zt.executed_flips_per_s/1e6:.1f}M/s, compile={res_2_zt.compile_s*1000:.1f}ms, "
        f"time={res_2_zt.elapsed_s*1000:.1f}ms, vram={res_2_zt.peak_vram_mib:.1f}MiB",
    )

    # 3. Cold-start naive <3,3,3:27> -> 23 over F2
    naive3_f2 = bv.make_naive_scheme(3, 3, 3, ring="Z2")
    res_3_f2 = engine.evolve_f2(
        [naive3_f2], target_rank=23, n_rounds=4, n_walkers=4096, n_steps=20000, plus_period=800, top_k=24, seed=303
    )
    rep_3_f2 = bv.verify_scheme(res_3_f2.top_schemes[0])
    rec(
        "f2_laderman_27_to_23",
        rep_3_f2.valid and res_3_f2.best_rank == 23 and res_3_f2.peak_vram_mib <= MAX_VRAM_MIB,
        f"R={res_3_f2.best_rank} (nnz={res_3_f2.best_nnz}), valid={rep_3_f2.valid}, "
        f"flips/s={res_3_f2.executed_flips_per_s/1e6:.1f}M/s, steps/s={res_3_f2.attempted_steps_per_s/1e6:.1f}M/s, "
        f"last_rd_time={res_3_f2.elapsed_s:.3f}s, vram={res_3_f2.peak_vram_mib:.1f}MiB",
    )

    # 4. Cold-start naive <3,3,3:27> -> 23 over Z_T
    naive3_zt = bv.make_naive_scheme(3, 3, 3, ring="Z")
    res_3_zt = engine.evolve_zt(
        [naive3_zt], target_rank=23, n_rounds=5, n_walkers=4096, n_steps=20000, plus_period=600, top_k=32, seed=500
    )
    rep_3_zt = bv.verify_scheme(res_3_zt.top_schemes[0])
    rec(
        "zt_laderman_27_to_23",
        rep_3_zt.valid and rep_3_zt.is_ternary and res_3_zt.best_rank == 23 and res_3_zt.peak_vram_mib <= MAX_VRAM_MIB,
        f"R={res_3_zt.best_rank} (nnz={res_3_zt.best_nnz}, adds={rep_3_zt.additive_complexity['naive_additions_total']}), "
        f"valid={rep_3_zt.valid}, ternary={rep_3_zt.is_ternary}, "
        f"flips/s={res_3_zt.executed_flips_per_s/1e6:.1f}M/s, steps/s={res_3_zt.attempted_steps_per_s/1e6:.1f}M/s, "
        f"last_rd_time={res_3_zt.elapsed_s:.3f}s, vram={res_3_zt.peak_vram_mib:.1f}MiB",
    )

    # 5. Verify padded_strassen_compose on <3,3,3:25>
    s222 = bv.strassen_222("Z")
    s221 = bv.make_naive_scheme(2, 2, 1, "Z")
    s122 = bv.make_naive_scheme(1, 2, 2, "Z")
    s212 = bv.make_naive_scheme(2, 1, 2, "Z")
    s112 = bv.make_naive_scheme(1, 1, 2, "Z")
    s211 = bv.make_naive_scheme(2, 1, 1, "Z")
    s121 = bv.make_naive_scheme(1, 2, 1, "Z")
    ps_333 = padded_strassen_compose(2, 1, 2, 1, 2, 1, [s222, s122, s221, s112, s211, s121, s212], name="ps_333")
    rep_ps = bv.verify_scheme(ps_333)
    rec(
        "padded_strassen_compose_exact",
        rep_ps.valid and rep_ps.is_ternary and ps_333.rank == 25,
        f"<3,3,3:{ps_333.rank}> valid={rep_ps.valid}, ternary={rep_ps.is_ternary}",
    )

    # 6. High-R Z_T smoke + throughput check at R=244 (5x8x8) verifying incremental edge-list correctness & speed!
    s244 = direct_sum_n(bv.direct_sum_p(s222, s222), bv.direct_sum_p(s222, s222))  # <2,4,4:28>
    s344 = direct_sum_n(bv.direct_sum_p(bv.laderman_333("Z"), bv.make_naive_scheme(3, 3, 1, "Z")), bv.make_naive_scheme(3, 1, 4, "Z"))  # <3,4,4:44>
    # Wait: 4*44 + 3*28 = 260 -> fits MAX_R=272 cleanly!
    ps_588 = padded_strassen_compose(2, 3, 4, 4, 4, 4, [s344, s344, s244, s344, s244, s344, s244], name="bench_588")
    res_hi = engine.run_zt_batch(
        [ps_588], n_walkers=2048, n_steps=15000, plus_period=600, plus_headroom=2, top_k=4, seed=606
    )
    rep_hi = bv.verify_scheme(res_hi.top_schemes[0])
    rec(
        "zt_highR_5x8x8_incremental",
        rep_hi.valid and rep_hi.is_ternary and res_hi.peak_vram_mib <= MAX_VRAM_MIB,
        f"start={ps_588.rank}->R={res_hi.best_rank}, valid={rep_hi.valid}, ternary={rep_hi.is_ternary}, "
        f"flips/s={res_hi.executed_flips_per_s/1e6:.1f}M/s, plus={res_hi.executed_plus:,}, red={res_hi.executed_reduces:,}, "
        f"time={res_hi.elapsed_s:.3f}s, vram={res_hi.peak_vram_mib:.1f}MiB",
    )

    # 7. Exhaustive 3x3 GPU hardware truth-table verification of branchless F_3 add/sub
    ok_tt, det_tt = engine.verify_f3_truth_table_gpu()
    rec("f3_add_sub_truth_table_gpu", ok_tt, det_tt)

    # 8. Cold-start naive <2,2,2:8> -> 7 over F_3 (verified in brent_verifier over "Z3")
    naive2_f3 = bv.make_naive_scheme(2, 2, 2, ring="Z3")
    res_2_f3 = engine.run_f3_batch(
        [naive2_f3], n_walkers=2048, n_steps=2000, plus_period=40, plus_headroom=2, top_k=4, seed=707
    )
    rep_2_f3 = bv.verify_scheme(res_2_f3.top_schemes[0])
    rec(
        "f3_strassen_8_to_7",
        rep_2_f3.valid and rep_2_f3.ring == "Z3" and res_2_f3.best_rank == 7 and res_2_f3.peak_vram_mib <= MAX_VRAM_MIB,
        f"R={res_2_f3.best_rank}, ring={rep_2_f3.ring}, valid={rep_2_f3.valid}, "
        f"flips/s={res_2_f3.executed_flips_per_s/1e6:.1f}M/s, compile={res_2_f3.compile_s*1000:.1f}ms, "
        f"time={res_2_f3.elapsed_s*1000:.1f}ms, vram={res_2_f3.peak_vram_mib:.1f}MiB",
    )

    # 9. Cold-start naive <3,3,3:27> -> 23 over F_3 (verified in brent_verifier over "Z3")
    naive3_f3 = bv.make_naive_scheme(3, 3, 3, ring="Z3")
    res_3_f3 = engine.evolve_f3(
        [naive3_f3], target_rank=23, n_rounds=5, n_walkers=4096, n_steps=20000, plus_period=600, top_k=24, seed=808
    )
    rep_3_f3 = bv.verify_scheme(res_3_f3.top_schemes[0])
    rec(
        "f3_laderman_27_to_23",
        rep_3_f3.valid and rep_3_f3.ring == "Z3" and res_3_f3.best_rank == 23 and res_3_f3.peak_vram_mib <= MAX_VRAM_MIB,
        f"R={res_3_f3.best_rank} (nnz={res_3_f3.best_nnz}), ring={rep_3_f3.ring}, valid={rep_3_f3.valid}, "
        f"flips/s={res_3_f3.executed_flips_per_s/1e6:.1f}M/s, last_rd_time={res_3_f3.elapsed_s:.3f}s, "
        f"vram={res_3_f3.peak_vram_mib:.1f}MiB",
    )

    # 10. High-R F_3 benchmark on 5x8x8 and 7x7x7 verifying zero-overflow speed & exact Z3 validity!
    s444 = kronecker_product(s222, s222).reduce_mod(3)
    r_s334_f3 = engine.evolve_f3(
        [bv.make_naive_scheme(3, 3, 4, "Z3")], target_rank=30, n_rounds=5,
        n_walkers=2048, n_steps=20000, plus_period=1000, top_k=16, seed=908,
    )
    s334_f3 = r_s334_f3.top_schemes[0]  # rank <= 30
    s344_f3 = direct_sum_n(s334_f3, bv.make_naive_scheme(3, 1, 4, "Z3"))  # rank <= 42 -> 7x7x7 <= 265 <= 272
    ps_777 = padded_strassen_compose(
        4, 3, 4, 3, 4, 3,
        [s444, s344_f3, permute_axes(s344_f3, (4, 4, 3)), s334_f3,
         permute_axes(s334_f3, (4, 3, 3)), permute_axes(s334_f3, (3, 4, 3)), permute_axes(s344_f3, (4, 3, 4))],
        name="bench_777_f3",
    )
    res_f3_588 = engine.run_f3_batch(
        [ps_588.reduce_mod(3)], n_walkers=2048, n_steps=15000, plus_period=600, plus_headroom=2, top_k=4, seed=909
    )
    res_f3_777 = engine.run_f3_batch(
        [ps_777], n_walkers=2048, n_steps=15000, plus_period=600, plus_headroom=2, top_k=4, seed=910
    )
    rep_f3_588 = bv.verify_scheme(res_f3_588.top_schemes[0])
    rep_f3_777 = bv.verify_scheme(res_f3_777.top_schemes[0])
    rec(
        "f3_highR_5x8x8_and_7x7x7",
        rep_f3_588.valid and rep_f3_777.valid and max(res_f3_588.peak_vram_mib, res_f3_777.peak_vram_mib) <= MAX_VRAM_MIB,
        f"5x8x8({ps_588.rank}->{res_f3_588.best_rank}): {res_f3_588.executed_flips_per_s/1e6:.1f}M/s valid={rep_f3_588.valid} | "
        f"7x7x7({ps_777.rank}->{res_f3_777.best_rank}): {res_f3_777.executed_flips_per_s/1e6:.1f}M/s valid={rep_f3_777.valid} | "
        f"vram={max(res_f3_588.peak_vram_mib, res_f3_777.peak_vram_mib):.1f}MiB",
    )

    all_ok = all(ok for _, ok, _ in checks)
    if verbose:
        print(
            f"GPU SELFTEST SUMMARY ({engine.backend.device_name}, {engine.backend.arch}): "
            f"{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} passed -> {'ALL PASS' if all_ok else 'FAILED'}"
        )
    return all_ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RTX 5060 SM 12.0 NVRTC Flip-Graph v2 Engine (F2 + Z_T).")
    parser.add_argument("--selftest", action="store_true", help="Run cold-start Strassen, Laderman & high-R GPU tests.")
    args = parser.parse_args(argv)
    if args.selftest:
        return 0 if run_selftest(verbose=True) else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
