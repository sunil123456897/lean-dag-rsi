"""
Fetch, parse, normalize (to `C_eq_AB_row_major`), and strictly certify public
2023-2026 F2 Fast Matrix Multiplication world-record schemes with full provenance
(raw URL, pinned commit SHA / arXiv version, raw SHA-256, canonical SHA-256).

All downloaded schemes are verified via `brent_verifier.verify_scheme()` over `"Z2"`
and saved to `fmm_discovery/artifacts/public_f2_seeds/` alongside `manifest.json`.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
import zlib
from typing import Any, Dict, List, Tuple

import numpy as np

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_FMM_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from brent_verifier import BilinearScheme, direct_sum_p, verify_scheme  # noqa: E402
from hensel_lifter import (  # noqa: E402
    fast_verify_f2_u64,
    mod4_oracle_u64,
    scheme_to_u64_f2,
)


def _http_get_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "antigravity-fmm-phase2-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _detect_and_verify_f2(
    m: int,
    n: int,
    p: int,
    U: List[List[int]],
    V: List[List[int]],
    W_raw: List[List[int]],
    w_is_pxm: bool,
    name: str,
) -> Tuple[BilinearScheme, str]:
    """
    Build BilinearScheme over Z2 and auto-detect whether W is in C=AB (m x p)
    or trace tr(ABC^T) (p x m) convention, returning ONLY a 100% brent_verifier-certified scheme!
    """
    R = len(U)
    # Candidate 1: direct C = AB row-major (m x p)
    if not w_is_pxm:
        sc_direct = BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W_raw, ring="Z2", modulus=2, name=name)
        if verify_scheme(sc_direct).valid:
            return sc_direct, "C_eq_AB_row_major"

    # Candidate 2: W_raw is in trace convention tr(A B C_tr) where C_tr = C^T (p x m)
    # Entry (k2, k1) in C_tr corresponds to (k1, k2) in C (m x p)
    W_ab = [[0] * (m * p) for _ in range(R)]
    for r in range(R):
        for k2 in range(p):
            for k1 in range(m):
                W_ab[r][k1 * p + k2] = W_raw[r][k2 * m + k1]
    sc_tr = BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W_ab, ring="Z2", modulus=2, name=name)
    if verify_scheme(sc_tr).valid:
        return sc_tr, "trace_tr_ABCt_transposed_to_C_eq_AB"

    # Candidate 3: if m == p and w_is_pxm was False, also try transposing W
    if m == p:
        sc_direct2 = BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W_raw, ring="Z2", modulus=2, name=name)
        if verify_scheme(sc_direct2).valid:
            return sc_direct2, "C_eq_AB_row_major"

    raise ValueError(f"Scheme {name} ({m}x{n}x{p}, R={R}) failed brent_verifier in BOTH C=AB and trace conventions!")


def parse_exp_f2(text: str, m: int, n: int, p: int, name: str) -> Tuple[BilinearScheme, str]:
    """
    Parse Kauers/Moosbauer `.exp` format:
      (a12+a13)*(b32+b35)*(c21+c53)
    1-based indices: a_{i,j} (1<=i<=m, 1<=j<=n), b_{j,k} (1<=j<=n, 1<=k<=p), c_{r,c}.
    Supports single-digit (m,n,p <= 9) indices as standard in mkauers/jakobmoosbauer repos.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    U: List[List[int]] = []
    V: List[List[int]] = []
    W_tr: List[List[int]] = []
    W_ab: List[List[int]] = []
    max_c_row = 0
    max_c_col = 0

    term_re = re.compile(r"\(([^)]+)\)\s*\*\s*\(([^)]+)\)\s*\*\s*\(([^)]+)\)")
    tok_re = re.compile(r"([abc])(\d)(\d)")

    raw_terms: List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]] = []
    for ln in lines:
        mt = term_re.search(ln)
        if not mt:
            continue
        pa, pb, pc = mt.group(1), mt.group(2), mt.group(3)
        a_idx = [(int(r) - 1, int(c) - 1) for ch, r, c in tok_re.findall(pa) if ch == "a"]
        b_idx = [(int(r) - 1, int(c) - 1) for ch, r, c in tok_re.findall(pb) if ch == "b"]
        c_idx = [(int(r) - 1, int(c) - 1) for ch, r, c in tok_re.findall(pc) if ch == "c"]
        for r, c in c_idx:
            max_c_row = max(max_c_row, r)
            max_c_col = max(max_c_col, c)
        raw_terms.append((a_idx, b_idx, c_idx))

    # Determine whether c_{r,c} is indexed as p x m (trace) or m x p (direct)
    c_is_pxm = (max_c_row >= m and max_c_row < p) or (max_c_col < m and max_c_row >= m)
    for a_idx, b_idx, c_idx in raw_terms:
        u_row = [0] * (m * n)
        for r, c in a_idx:
            u_row[r * n + c] ^= 1
        v_row = [0] * (n * p)
        for r, c in b_idx:
            v_row[r * p + c] ^= 1
        U.append(u_row)
        V.append(v_row)
        if c_is_pxm:
            w_row = [0] * (p * m)
            for r, c in c_idx:
                w_row[r * m + c] ^= 1
            W_tr.append(w_row)
        else:
            w_row = [0] * (m * p)
            for r, c in c_idx:
                w_row[r * p + c] ^= 1
            W_ab.append(w_row)

    if c_is_pxm:
        return _detect_and_verify_f2(m, n, p, U, V, W_tr, w_is_pxm=True, name=name)
    return _detect_and_verify_f2(m, n, p, U, V, W_ab, w_is_pxm=False, name=name)


def parse_mathematica_m_f2(text: str, m: int, n: int, p: int, name: str) -> Tuple[BilinearScheme, str]:
    """
    Parse Mathematica `.m` nested-brace tensor format (e.g. yamato-arai/adap/data/455_73.m):
      {{{{...}, ...}, {{...}, ...}, {{...}, ...}}, ...}
    """
    py_lit = text.strip().replace("{", "[").replace("}", "]")
    # Strip any trailing semicolon or variable assignment
    if "=" in py_lit.split("\n")[0]:
        py_lit = py_lit.split("=", 1)[1].strip()
    py_lit = py_lit.rstrip(";").strip()
    data = ast.literal_eval(py_lit)

    U: List[List[int]] = []
    V: List[List[int]] = []
    W_raw: List[List[int]] = []
    w_is_pxm = False

    for term in data:
        mat_a, mat_b, mat_c = term[0], term[1], term[2]
        assert len(mat_a) == m and len(mat_a[0]) == n, f"A shape mismatch: {len(mat_a)}x{len(mat_a[0])}"
        assert len(mat_b) == n and len(mat_b[0]) == p, f"B shape mismatch: {len(mat_b)}x{len(mat_b[0])}"
        U.append([int(x) % 2 for row in mat_a for x in row])
        V.append([int(x) % 2 for row in mat_b for x in row])
        if len(mat_c) == p and len(mat_c[0]) == m and m != p:
            w_is_pxm = True
        W_raw.append([int(x) % 2 for row in mat_c for x in row])

    return _detect_and_verify_f2(m, n, p, U, V, W_raw, w_is_pxm=w_is_pxm, name=name)


def parse_medley_arxiv_777(tar_bytes: bytes, name: str) -> Tuple[BilinearScheme, str, str]:
    """
    Extract and decode Zeke Medley et al. (arXiv:2609.17533v2) Appendix A rank-245 7x7x7 F2 scheme
    from `BEGIN-RANK245-7X7X7-ZLIB-BASE32 ... END-RANK245-7X7X7-ZLIB-BASE32`.
    Returns (verified BilinearScheme, convention, raw_payload_block).
    """
    tf = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*")
    tex_txt = ""
    for mem in tf.getmembers():
        if mem.isfile() and mem.name.endswith(".tex"):
            candidate = tf.extractfile(mem).read().decode("utf-8", errors="ignore")
            if "BEGIN-RANK245-7X7X7-ZLIB-BASE32" in candidate:
                tex_txt = candidate
                break
    if not tex_txt:
        raise ValueError("Could not locate BEGIN-RANK245-7X7X7-ZLIB-BASE32 inside arXiv tarball")

    start_tag = "BEGIN-RANK245-7X7X7-ZLIB-BASE32"
    end_tag = "END-RANK245-7X7X7-ZLIB-BASE32"
    s_idx = tex_txt.index(start_tag) + len(start_tag)
    e_idx = tex_txt.index(end_tag, s_idx)
    payload_raw = tex_txt[s_idx:e_idx]
    b32 = "".join(ch for ch in payload_raw if ch.isalnum() or ch == "=")
    # Pad base32 to multiple of 8 if needed
    pad = (-len(b32)) % 8
    raw_zlib = base64.b32decode(b32 + ("=" * pad))
    blob = zlib.decompress(raw_zlib)
    # 245 terms * 3 matrices * 7 little-endian bytes = 5145 bytes
    assert len(blob) % 21 == 0, f"Unexpected decoded length {len(blob)} (expected multiple of 21)"
    R = len(blob) // 21

    U: List[List[int]] = []
    V: List[List[int]] = []
    W_raw: List[List[int]] = []
    for r in range(R):
        off = r * 21
        ma = int.from_bytes(blob[off : off + 7], "little")
        mb = int.from_bytes(blob[off + 7 : off + 14], "little")
        mc = int.from_bytes(blob[off + 14 : off + 21], "little")
        U.append([(ma >> b) & 1 for b in range(49)])
        V.append([(mb >> b) & 1 for b in range(49)])
        W_raw.append([(mc >> b) & 1 for b in range(49)])

    sc, conv = _detect_and_verify_f2(7, 7, 7, U, V, W_raw, w_is_pxm=False, name=name)
    return sc, conv, payload_raw.strip()


def _count_f2_edges(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> int:
    R = len(u)
    ne = 0
    for i in range(R):
        ui, vi, wi = u[i], v[i], w[i]
        for j in range(i + 1, R):
            if ui == u[j]:
                ne += 1
            if vi == v[j]:
                ne += 1
            if wi == w[j]:
                ne += 1
    return ne


def fetch_all_public_seeds(out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    t0 = time.perf_counter()
    manifest_entries: List[Dict[str, Any]] = []

    def _finalize_and_save(
        seed_id: str,
        sc: BilinearScheme,
        provenance: Dict[str, Any],
        check_unrestricted: bool = True,
    ) -> Dict[str, Any]:
        rep = verify_scheme(sc)
        assert rep.valid, f"CRITICAL: {seed_id} failed brent_verifier!"
        u, v, w = scheme_to_u64_f2(sc)
        assert fast_verify_f2_u64(sc.m, sc.n, sc.p, u, v, w)

        o_supp = mod4_oracle_u64(sc.m, sc.n, sc.p, u, v, w, support_restricted=True)
        unres_info: Dict[str, Any] = {"coker_viol_4": None, "free_bits": None, "rank_J1": None, "elapsed_ms": None}
        if check_unrestricted and sc.m * sc.n * sc.p <= 128:
            o_unres = mod4_oracle_u64(sc.m, sc.n, sc.p, u, v, w, support_restricted=False)
            unres_info = {
                "coker_viol_4": o_unres.coker_viol_4,
                "free_bits": o_unres.free_bits,
                "rank_J1": o_unres.rank_J1,
                "elapsed_ms": o_unres.elapsed_ms,
            }

        ne = _count_f2_edges(u, v, w)
        provenance = dict(provenance)
        provenance["canonical_sha256"] = rep.canonical_sha256
        provenance["verified_brent_Z2"] = True
        sc.metadata.update({"provenance": provenance, "seed_id": seed_id})

        json_path = os.path.join(out_dir, f"{seed_id}.json")
        sc.save_json(json_path)

        entry = {
            "seed_id": seed_id,
            "file": json_path,
            "format": [sc.m, sc.n, sc.p],
            "rank": sc.rank,
            "nnz_total": o_supp.nnz_total,
            "flip_edges": ne,
            "supp_coker_4": o_supp.coker_viol_4,
            "supp_free_bits": o_supp.free_bits,
            "supp_rank_J1": o_supp.rank_J1,
            "supp_oracle_ms": o_supp.elapsed_ms,
            "unres_coker_4": unres_info["coker_viol_4"],
            "unres_free_bits": unres_info["free_bits"],
            "unres_rank_J1": unres_info["rank_J1"],
            "unres_oracle_ms": unres_info["elapsed_ms"],
            "provenance": provenance,
        }
        manifest_entries.append(entry)
        print(
            f"  [VERIFIED Z2] {seed_id:24s} <{sc.m},{sc.n},{sc.p}:R={sc.rank:3d}> "
            f"nnz={o_supp.nnz_total:4d} ne={ne:3d} "
            f"supp_coker={o_supp.coker_viol_4:3d} (free={o_supp.free_bits:3d}) "
            f"unres_coker={unres_info['coker_viol_4']} (t_u={unres_info['elapsed_ms']}ms) "
            f"sha={rep.canonical_sha256[:12]}",
            flush=True,
        )
        return entry

    # 1. Priority 1: 4x5x5 R=73 (Arai-Ichikawa-Hukushima, ISSAC 2024)
    url_455 = "https://raw.githubusercontent.com/yamato-arai/adap/fe7b20408a32/data/455_73.m"
    raw_455 = _http_get_bytes(url_455)
    with open(os.path.join(raw_dir, "455_73.m"), "wb") as fh:
        fh.write(raw_455)
    sc_455, conv_455 = parse_mathematica_m_f2(raw_455.decode("utf-8"), 4, 5, 5, "arai_4x5x5_R73_F2")
    _finalize_and_save(
        "4x5x5_R73_arai",
        sc_455,
        {
            "source_repo": "yamato-arai/adap",
            "paper": "Arai, Ichikawa, Hukushima (ISSAC 2024, arXiv:2312.16960)",
            "url": url_455,
            "commit_sha": "fe7b20408a32",
            "raw_sha256": _sha256_hex(raw_455),
            "convention_detected": conv_455,
        },
    )

    # 2. Priority 2: 4x5x6 R=89 (TWO distinct schemes from Kauers-Wood, Oct 2025)
    for tag, fname in [("a", "k08dcabe7aa6634b.mod2.exp"), ("b", "k0e35aeb54b662ba.mod2.exp")]:
        url_456 = f"https://raw.githubusercontent.com/mkauers/matrix-multiplication/12c26b29a545/456/{fname}"
        raw_456 = _http_get_bytes(url_456)
        with open(os.path.join(raw_dir, fname), "wb") as fh:
            fh.write(raw_456)
        sc_456, conv_456 = parse_exp_f2(raw_456.decode("utf-8"), 4, 5, 6, f"kauers_4x5x6_R89_{tag}_F2")
        _finalize_and_save(
            f"4x5x6_R89_kauers_{tag}",
            sc_456,
            {
                "source_repo": "mkauers/matrix-multiplication",
                "paper": "Kauers & Wood (2025), Meta-Flip Graphs",
                "url": url_456,
                "commit_sha": "12c26b29a545",
                "raw_sha256": _sha256_hex(raw_456),
                "convention_detected": conv_456,
            },
        )

    # 3. Priority 3 (+ Bonus 4x4x4:47): 4x4x4 R=47 -> direct sum along p -> 4x4x8 R=94!
    url_444 = "https://raw.githubusercontent.com/jakobmoosbauer/flips/e31a0a0f0d25/solutions/444-47-mod2.exp"
    raw_444 = _http_get_bytes(url_444)
    with open(os.path.join(raw_dir, "444-47-mod2.exp"), "wb") as fh:
        fh.write(raw_444)
    sc_444, conv_444 = parse_exp_f2(raw_444.decode("utf-8"), 4, 4, 4, "km_4x4x4_R47_F2")
    _finalize_and_save(
        "4x4x4_R47_km",
        sc_444,
        {
            "source_repo": "jakobmoosbauer/flips",
            "paper": "Kauers & Moosbauer (ISSAC 2023, arXiv:2212.01175)",
            "url": url_444,
            "commit_sha": "e31a0a0f0d25",
            "raw_sha256": _sha256_hex(raw_444),
            "convention_detected": conv_444,
        },
    )

    sc_448 = direct_sum_p(sc_444, sc_444, name="km_4x4x8_R94_direct_sum_F2")
    _finalize_and_save(
        "4x4x8_R94_km_ds",
        sc_448,
        {
            "source_repo": "jakobmoosbauer/flips (direct_sum_p of 444-47-mod2.exp x2)",
            "paper": "Kauers & Moosbauer (ISSAC 2023) 4x4x4:47 direct sum along p",
            "url": url_444,
            "commit_sha": "e31a0a0f0d25",
            "raw_sha256": _sha256_hex(raw_444),
            "convention_detected": f"{conv_444} -> direct_sum_p",
        },
    )

    # 4. Priority 4: 7x7x7 R=245 (Zeke Medley et al., arXiv:2609.17533v2, Sep 2026)
    url_777 = "https://arxiv.org/e-print/2609.17533v2"
    raw_777 = _http_get_bytes(url_777, timeout=45)
    sc_777, conv_777, payload_777 = parse_medley_arxiv_777(raw_777, "medley_7x7x7_R245_F2")
    with open(os.path.join(raw_dir, "medley_777_245_base32.txt"), "w", encoding="utf-8") as fh:
        fh.write(payload_777)
    _finalize_and_save(
        "7x7x7_R245_medley",
        sc_777,
        {
            "source_repo": "arXiv:2609.17533v2 (Appendix A BEGIN-RANK245-7X7X7-ZLIB-BASE32)",
            "paper": "Medley, Gokul, Luu, Manolios (Sep 2026 / ALENEX 2027)",
            "url": url_777,
            "commit_sha": "arXiv:2609.17533v2",
            "raw_sha256": _sha256_hex(payload_777.encode("utf-8")),
            "tarball_sha256": _sha256_hex(raw_777),
            "convention_detected": conv_777,
        },
        check_unrestricted=False,  # 7x7x7 unrestricted Jacobian is 117,649 x 36,015 -> use support oracle!
    )

    # 5. Priority 5: 4x4x5 R=60 (Kauers & Moosbauer, ISSAC 2023)
    url_445 = "https://raw.githubusercontent.com/jakobmoosbauer/flips/e31a0a0f0d25/solutions/445-60-mod2.exp"
    raw_445 = _http_get_bytes(url_445)
    with open(os.path.join(raw_dir, "445-60-mod2.exp"), "wb") as fh:
        fh.write(raw_445)
    sc_445, conv_445 = parse_exp_f2(raw_445.decode("utf-8"), 4, 4, 5, "km_4x4x5_R60_F2")
    _finalize_and_save(
        "4x4x5_R60_km",
        sc_445,
        {
            "source_repo": "jakobmoosbauer/flips",
            "paper": "Kauers & Moosbauer (ISSAC 2023, arXiv:2212.01175)",
            "url": url_445,
            "commit_sha": "e31a0a0f0d25",
            "raw_sha256": _sha256_hex(raw_445),
            "convention_detected": conv_445,
        },
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "convention": "C_eq_AB_row_major",
        "total_seeds": len(manifest_entries),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "seeds": manifest_entries,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(manifest_entries)} brent_verifier-certified public F2 seeds -> {manifest_path}", flush=True)
    return manifest


def main() -> int:
    out_dir = os.path.join(_FMM_ROOT, "artifacts", "public_f2_seeds")
    fetch_all_public_seeds(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
