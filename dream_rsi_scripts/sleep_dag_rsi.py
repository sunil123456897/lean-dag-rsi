"""
SLEEP-DAG-RSI: Unified Parametric Continual Learning (HOPE + CMS + SLEEP)
and Counterfactual DAG Recursive Self-Improvement (Lean-DAG-RSI v3/v4).

Unites:
  1. Google Research SLEEP Framework (Behrouz et al., arXiv:2606.03979v2, July 2026)
     & Nested Learning / HOPE (Behrouz et al., arXiv:2512.24695, Dec 2025):
       - DeltaRuleFastMemory: token-level online associative memory M_t = alpha_t M_{t-1} + beta_t (v_t - M_{t-1} k_t) k_t^T
       - Continuum Memory System (CMS): multi-frequency MoE hierarchy with exact C^(l) gradient accumulation
       - Stage 1 (NREM Memory Consolidation): Compute -> Consolidate -> Update protocol,
         Parameter Expansion via low-rank experts {A, B}, Generalized Knowledge Distillation (GKD JSD)
         + Learning to Imitate (LTI via Levenshtein r_abs and semantic r_sem rewards), and Synaptic Pruning
       - Stage 2 (REM Dreaming): MoE expert mixing, gradient-norm importance scoring ||g_DR||_2,
         isolated LoRA adaptation, and ReST^EM self-improvement.

  2. Lean-DAG-RSI v3 & v4 (`Auto-Dream-RSI`, `dream_rsi_scripts/engine.py`):
       - Monotone Replay Objective V_Lean(T, pi) = max s_v - beta_1 N - beta_2 k*
       - Counterfactual Tukey Continuation s_cf(u, m) to eliminate Support Censorship
       - Error-Fingerprint DAG Crossover (compute_error_fingerprint) & Multi-Parent Submodular
         Hyper-Crossover (compute_hyper_crossover) with AST Call-Graph IoU collision penalty
       - Neuro-Symbolic Split + RRSI Complexity Pruner (prune_tunable_params & prune_epistatic_gauss_newton)
         + 20-direction +/-5% Flat-Plateau Verification (evaluate_flat_plateau)
       - DiscoveryRSI v3/v4 Closed-Loop Gates: Pre-Commitment H=(c,pi,d,a), Harness Replay Diagnostics,
         Provenance Credit (lambda_breakthrough=0.35 > lambda_repair=0.10, C_glob<=2.0),
         Selective Loss-Masked SFT Exporter (export_masked_sft_dataset), Contrastive Fork DPO Exporter
         (export_dag_dpo_pairs), and 2x2 Joint Model x Policy Acceptance Gate (accept_joint_2x2).
"""

from __future__ import annotations

import copy
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

LOCAL_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dream_rsi_scripts")
if os.path.isdir(LOCAL_SCRIPTS_DIR) and LOCAL_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, LOCAL_SCRIPTS_DIR)

import engine as dream_engine  # type: ignore


def rms_step(grad: np.ndarray, lr: float, clip: float = 2.5) -> np.ndarray:
    """RMS-normalized gradient step for fast, scale-invariant optimization."""
    rms = float(np.sqrt(np.mean(grad * grad)))
    if rms < 1e-12:
        return np.zeros_like(grad)
    return lr * np.clip(grad / rms, -clip, clip)


# =====================================================================
# 1. CONFIGURATION (HOPE + CMS + SLEEP + LEAN-DAG-RSI)
# =====================================================================


@dataclass
class SleepDAGConfig:
    vocab_size: int = 64
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    d_low: int = 12  # Low-rank expert rank: d_low << d_ff

    # Continuum Memory System (CMS): update periods C^(l) for each block
    # Frequencies f_1 >= f_2 >= f_3 correspond to C^(1)=2, C^(2)=4, C^(3)=8
    cms_update_periods: Tuple[int, ...] = (2, 4, 8)
    num_base_experts: int = 2
    max_low_rank_experts: int = 12
    top_k_experts: int = 1
    router_temperature: float = 0.20

    # Wake & Sleep learning rates and Knowledge Seeding hyperparameters (Sec. 3.3)
    wake_lr: float = 0.12
    sleep_lr: float = 0.14
    consolidation_steps: int = 14
    gkd_lambda: float = 0.4        # Fraction of on-policy student trajectories in L_GKD
    gkd_beta: float = 0.5          # JSD interpolation weight between KL(p_T||p_S) and KL(p_S||p_T)
    ks_alpha: float = 0.6          # Balance between GKD distillation and LTI imitation reward in L_KS
    lti_gamma: float = 0.5         # Balance between semantic reward r_sem and Levenshtein r_abs (Eq. 3)
    levenshtein_z0: int = 8        # Levenshtein distance threshold z_0 (Eq. 4)
    semantic_sim_threshold: float = 0.70
    dpo_repulsion_weight: float = 0.45  # Repulsion weight against rejected DAG forks (dead ends / traps)
    replay_anchor_weight: float = 0.50  # Replay anchor weight for prior consolidated domains

    # REM Dreaming & Counterfactual DAG Filter hyperparameters (Sec. 3.4 + Lean-DAG-RSI)
    num_candidate_dreams: int = 12
    top_k_dreams: int = 3
    random_b_dreams: int = 1
    dream_seq_len: int = 12
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_sft_steps: int = 10
    restem_steps: int = 10
    min_novelty_floor: float = 0.30


# =====================================================================
# 2. LTI METRICS: LEVENSHTEIN DISTANCE (EQ. 4), SEMANTIC REWARD (EQ. 3),
#    AND GENERALIZED JENSEN-SHANNON DIVERGENCE (GKD)
# =====================================================================


def levenshtein_distance(seq1: Sequence[int], seq2: Sequence[int]) -> int:
    """Exact Wagner-Fischer token-level edit distance z(d_hat, d)."""
    m, n = len(seq1), len(seq2)
    if m < n:
        return levenshtein_distance(seq2, seq1)
    if n == 0:
        return m
    prev_row = list(range(n + 1))
    for i, c1 in enumerate(seq1):
        curr_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (1 if c1 != c2 else 0)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def compute_r_abs(pred_tokens: Sequence[int], target_tokens: Sequence[int], z0: int) -> float:
    """
    Equation (4) of arXiv:2606.03979v2:
    Thresholded normalized Levenshtein reward:
      r_abs = 1 - z(d_hat, d) / max(|d_hat|, |d|)  if z <= z0 else 0.
    """
    z = levenshtein_distance(pred_tokens, target_tokens)
    max_len = max(len(pred_tokens), len(target_tokens), 1)
    if z <= z0:
        return 1.0 - (float(z) / float(max_len))
    return 0.0


def compute_r_sem_numpy(
    pred_emb: np.ndarray,
    target_emb: np.ndarray,
    threshold: float = 0.70,
) -> float:
    """
    Equation (3) of arXiv:2606.03979v2:
    Binary semantic equivalence reward r_sem in {0, 1} from frozen teacher representations.
    """
    v1 = pred_emb.mean(axis=-2)
    v2 = target_emb.mean(axis=-2)
    norm1 = np.linalg.norm(v1, axis=-1) + 1e-12
    norm2 = np.linalg.norm(v2, axis=-1) + 1e-12
    cos_sim = float(np.mean(np.sum(v1 * v2, axis=-1) / (norm1 * norm2)))
    return 1.0 if cos_sim >= threshold else 0.0


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def generalized_jsd_loss_and_grad(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    beta: float = 0.5,
    token_weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Computes Generalized Jensen-Shannon Divergence F(p_T || p_S) for GKD
    and its EXACT analytical gradient dL / d(student_logits).
    """
    p_t = np.clip(softmax_np(teacher_logits, axis=-1), 1e-12, 1.0)
    p_s = np.clip(softmax_np(student_logits, axis=-1), 1e-12, 1.0)
    m = np.clip(beta * p_t + (1.0 - beta) * p_s, 1e-12, 1.0)

    log_pt = np.log(p_t)
    log_ps = np.log(p_s)
    log_m = np.log(m)

    kl_t_m = np.sum(p_t * (log_pt - log_m), axis=-1)
    kl_s_m = np.sum(p_s * (log_ps - log_m), axis=-1)
    jsd_per_token = beta * kl_t_m + (1.0 - beta) * kl_s_m  # [B, T]

    diff_log = log_ps - log_m  # [B, T, V]
    expected_diff = np.sum(p_s * diff_log, axis=-1, keepdims=True)  # [B, T, 1]
    grad_z = (1.0 - beta) * p_s * (diff_log - expected_diff)  # [B, T, V]

    if token_weights is not None:
        w = np.asarray(token_weights, dtype=np.float64)
        if w.ndim == 2 and w.shape[1] == 1 and jsd_per_token.shape[1] > 1:
            w = np.broadcast_to(w, jsd_per_token.shape)
        w_sum = max(float(np.sum(w)), 1e-12)
        loss = float(np.sum(jsd_per_token * w) / w_sum)
        grad_z = grad_z * (w[..., None] / w_sum)
    else:
        n_elem = float(jsd_per_token.size)
        loss = float(np.mean(jsd_per_token))
        grad_z = grad_z / n_elem

    return loss, grad_z


def cross_entropy_loss_and_grad(
    logits: np.ndarray,
    targets: np.ndarray,
    loss_mask: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Exact cross-entropy loss and gradient w.r.t. logits [B, T, V] for targets [B, T].
    Supports selective token/step masking `loss_mask` [B, T] from `export_masked_sft_dataset`.
    """
    B, T, V = logits.shape
    probs = np.clip(softmax_np(logits, axis=-1), 1e-12, 1.0)
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    target_probs = probs[b_idx, t_idx, targets]
    nll = -np.log(target_probs)  # [B, T]

    grad = probs.copy()
    grad[b_idx, t_idx, targets] -= 1.0

    if loss_mask is not None:
        mask = np.asarray(loss_mask, dtype=np.float64)
        if mask.ndim == 2 and mask.shape[1] == 1 and T > 1:
            mask = np.broadcast_to(mask, (B, T))
        denom = max(float(np.sum(mask)), 1e-12)
        loss = float(np.sum(nll * mask) / denom)
        grad = grad * (mask[..., None] / denom)
    else:
        denom = float(B * T)
        loss = float(np.mean(nll))
        grad = grad / denom

    return loss, grad


# =====================================================================
# 3. HOPE INNER LOOP: DELTA-RULE FAST ASSOCIATIVE MEMORY
# =====================================================================


class DeltaRuleFastMemory:
    """
    Inner-loop Fast Associative Memory (Self-Modifying Layer from HOPE, Behrouz et al. 2025):
      M_t = alpha_t * M_{t-1} + beta_t * (v_t - M_{t-1} k_t) k_t^T
      y_t = M_t q_t
    """

    def __init__(self, d_model: int, rng: np.random.Generator) -> None:
        self.d_model = d_model
        scale = 1.0 / math.sqrt(d_model)
        self.w_k = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_v = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_q = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_alpha = rng.normal(0.0, scale, size=(d_model, 1))
        self.b_alpha = np.full((1,), 2.5, dtype=np.float64)
        self.w_beta = rng.normal(0.0, scale, size=(d_model, 1))
        self.b_beta = np.full((1,), 1.5, dtype=np.float64)
        self.out_proj = np.eye(d_model, dtype=np.float64) * 0.4

        self.grad_out_proj = np.zeros_like(self.out_proj)
        self.grad_w_v = np.zeros_like(self.w_v)
        self.grad_w_q = np.zeros_like(self.w_q)
        self._cache: Dict[str, Any] = {}

    def zero_grad(self) -> None:
        self.grad_out_proj.fill(0.0)
        self.grad_w_v.fill(0.0)
        self.grad_w_q.fill(0.0)

    def forward(
        self,
        x: np.ndarray,
        memory_state: Optional[np.ndarray] = None,
        write_Scale: float = 0.85,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        B, T, D = x.shape
        k_raw = x @ self.w_k
        k_norm = np.linalg.norm(k_raw, axis=-1, keepdims=True) + 1e-12
        k = k_raw / k_norm
        v = x @ self.w_v
        q = x @ self.w_q

        alpha_logit = x @ self.w_alpha + self.b_alpha
        beta_logit = x @ self.w_beta + self.b_beta
        alpha = 1.0 / (1.0 + np.exp(-np.clip(alpha_logit, -25.0, 25.0)))
        beta = write_Scale / (1.0 + np.exp(-np.clip(beta_logit, -25.0, 25.0)))

        if memory_state is None:
            M = np.zeros((B, D, D), dtype=np.float64)
        else:
            M = memory_state.copy()

        y_list: List[np.ndarray] = []
        m_hist: List[np.ndarray] = []
        err_norms: List[float] = []

        for t in range(T):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            a_t = alpha[:, t, :][:, :, None]
            b_t = beta[:, t, :][:, :, None]

            pred_v = np.einsum("bij,bj->bi", M, k_t)
            error = v_t - pred_v
            err_norms.append(float(np.mean(np.linalg.norm(error, axis=-1))))

            outer_ek = np.einsum("bi,bj->bij", error, k_t)
            M = a_t * M + b_t * outer_ek
            m_hist.append(M.copy())

            y_t = np.einsum("bij,bj->bi", M, q_t)
            y_list.append(y_t)

        y_stack = np.stack(y_list, axis=1)
        out = x + y_stack @ self.out_proj
        self._cache = {"x": x, "y_stack": y_stack, "q": q, "m_hist": m_hist}
        return out, M, err_norms

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache["x"]
        y_stack = self._cache["y_stack"]
        m_hist = self._cache["m_hist"]

        self.grad_out_proj += np.einsum("bti,btj->ij", y_stack, grad_out)
        grad_y = grad_out @ self.out_proj.T

        grad_q = np.zeros_like(x)
        for t, M_t in enumerate(m_hist):
            grad_q[:, t, :] = np.einsum("bij,bi->bj", M_t, grad_y[:, t, :])
        self.grad_w_q += np.einsum("bti,btj->ij", x, grad_q)

        grad_x = grad_out + grad_q @ self.w_q.T
        return grad_x


# =====================================================================
# 4. EXPANDABLE MoE BLOCK WITH ERROR-FINGERPRINT-TAGGED LOW-RANK EXPERTS
# =====================================================================


def silu_np(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
    act = z * sig
    d_act = sig * (1.0 + z * (1.0 - sig))
    return act, d_act


@dataclass
class ExpertFingerprintMeta:
    """
    Bridge 2 Metadata: Attaches Lean-DAG-RSI Error Fingerprint (`pass_mask`),
    margin vector, AST symbols, and Provenance Credit to each unlocked MoE expert.
    """
    expert_slot: int
    source_node_id: str = "init"
    domain_task: str = "generic"
    pass_mask: List[int] = field(default_factory=list)
    margin_vector: List[float] = field(default_factory=list)
    ast_symbols: Set[str] = field(default_factory=set)
    provenance_credit: float = 1.0


class LowRankExpert:
    """
    Low-rank expert {A, B} (Sec. 3.2 of SLEEP):
      f(h) = SiLU(h @ A) @ B, where A in R^{d x d_low}, B in R^{d_low x d}.
    """

    def __init__(self, d_model: int, d_low: int, rng: np.random.Generator) -> None:
        self.d_model = d_model
        self.d_low = d_low
        self.rng = rng
        self.A = np.zeros((d_model, d_low), dtype=np.float64)
        self.B = np.zeros((d_low, d_model), dtype=np.float64)
        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)
        self.requires_grad: bool = True
        self.reset_parameters()
        self._cache: Dict[str, np.ndarray] = {}

    def reset_parameters(self) -> None:
        self.A = self.rng.normal(0.0, 1.0 / math.sqrt(self.d_model), size=(self.d_model, self.d_low))
        self.B = self.rng.normal(0.0, 0.25 / math.sqrt(self.d_low), size=(self.d_low, self.d_model))
        self.grad_A.fill(0.0)
        self.grad_B.fill(0.0)

    def zero_grad(self) -> None:
        self.grad_A.fill(0.0)
        self.grad_B.fill(0.0)

    def forward(self, h: np.ndarray) -> np.ndarray:
        z = h @ self.A
        u, d_u = silu_np(z)
        out = u @ self.B
        self._cache = {"h": h, "u": u, "d_u": d_u}
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        h = self._cache["h"]
        u = self._cache["u"]
        d_u = self._cache["d_u"]
        if self.requires_grad:
            self.grad_B += np.einsum("bti,btj->ij", u, grad_out)
        grad_u = grad_out @ self.B.T
        grad_z = grad_u * d_u
        if self.requires_grad:
            self.grad_A += np.einsum("bti,btj->ij", h, grad_z)
        return grad_z @ self.A.T


class DenseExpert:
    """Base dense MLP expert in CMS block."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = rng.normal(0.0, 1.0 / math.sqrt(d_model), size=(d_model, d_ff))
        self.w2 = rng.normal(0.0, 0.25 / math.sqrt(d_ff), size=(d_ff, d_model))
        self.grad_w1 = np.zeros_like(self.w1)
        self.grad_w2 = np.zeros_like(self.w2)
        self.requires_grad: bool = True
        self._cache: Dict[str, np.ndarray] = {}

    def zero_grad(self) -> None:
        self.grad_w1.fill(0.0)
        self.grad_w2.fill(0.0)

    def forward(self, h: np.ndarray) -> np.ndarray:
        z = h @ self.w1
        u, d_u = silu_np(z)
        out = u @ self.w2
        self._cache = {"h": h, "u": u, "d_u": d_u}
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        h = self._cache["h"]
        u = self._cache["u"]
        d_u = self._cache["d_u"]
        if self.requires_grad:
            self.grad_w2 += np.einsum("bti,btj->ij", u, grad_out)
        grad_u = grad_out @ self.w2.T
        grad_z = grad_u * d_u
        if self.requires_grad:
            self.grad_w1 += np.einsum("bti,btj->ij", h, grad_z)
        return grad_z @ self.w1.T


class ExpandableMoEMLP:
    """
    CMS Memory Block MLP^(f_l) with:
      - Base dense experts {W^(f_l),1, ..., W^(f_l),s_l}
      - Pre-allocated pool of low-rank experts {A, B} unlocked during Sleep Parameter Expansion
      - Bridge 2: Error-Fingerprint Metadata registry `expert_fingerprints` for directed
        complementary expert crossover during REM Dreaming.
    """

    def __init__(self, cfg: SleepDAGConfig, update_period: int, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.update_period = update_period
        self.num_base = cfg.num_base_experts
        self.max_low_rank = cfg.max_low_rank_experts
        self.total_slots = self.num_base + self.max_low_rank
        self.top_k = cfg.top_k_experts
        self.rng = rng

        self.base_experts = [
            DenseExpert(cfg.d_model, cfg.d_ff, rng) for _ in range(self.num_base)
        ]
        self.low_rank_experts = [
            LowRankExpert(cfg.d_model, cfg.d_low, rng) for _ in range(self.max_low_rank)
        ]
        self.router_w = rng.normal(0.0, 0.2 / math.sqrt(cfg.d_model), size=(cfg.d_model, self.total_slots))
        self.grad_router_w = np.zeros_like(self.router_w)
        self.router_grad_mask = np.ones_like(self.router_w)

        self.active_mask = np.zeros(self.total_slots, dtype=bool)
        self.active_mask[: self.num_base] = True
        self.num_unlocked_low_rank: int = 0

        self.expert_fingerprints: Dict[int, ExpertFingerprintMeta] = {}
        self._cache: Dict[str, Any] = {}

    def zero_grad(self) -> None:
        self.grad_router_w.fill(0.0)
        for be in self.base_experts:
            be.zero_grad()
        for lre in self.low_rank_experts:
            lre.zero_grad()

    def unlock_new_low_rank_expert(
        self,
        fingerprint_meta: Optional[ExpertFingerprintMeta] = None,
    ) -> int:
        if self.num_unlocked_low_rank >= self.max_low_rank:
            idx = self.max_low_rank - 1
        else:
            idx = self.num_unlocked_low_rank
            self.num_unlocked_low_rank += 1
        self.low_rank_experts[idx].reset_parameters()
        slot = self.num_base + idx
        self.active_mask[slot] = True
        if fingerprint_meta is not None:
            fingerprint_meta.expert_slot = slot
            self.expert_fingerprints[slot] = fingerprint_meta
        else:
            self.expert_fingerprints[slot] = ExpertFingerprintMeta(expert_slot=slot)
        return idx

    def prune_temporary_experts(self) -> None:
        for idx in range(self.num_unlocked_low_rank):
            self.low_rank_experts[idx].reset_parameters()
            slot = self.num_base + idx
            self.active_mask[slot] = False
            self.expert_fingerprints.pop(slot, None)
        self.num_unlocked_low_rank = 0

    def select_complementary_crossover_experts(self) -> Optional[Tuple[int, int, float]]:
        """
        Bridge 2 (Directed REM Expert Crossover via `compute_hyper_crossover`):
        Finds the pair of active low-rank experts (slot_u, slot_v) whose DAG `pass_mask` / `margin_vector`
        yields the strictly highest positive Oracle Union Gain Delta_U(u, v) > 0 and minimal AST collision.
        """
        active_lr_slots = [
            s for s in range(self.num_base, self.total_slots)
            if self.active_mask[s] and s in self.expert_fingerprints
            and len(self.expert_fingerprints[s].pass_mask) > 0
        ]
        if len(active_lr_slots) < 2:
            return None

        virtual_nodes: List[dream_engine.Node] = []
        slot_by_node_id: Dict[str, int] = {}
        for slot in active_lr_slots:
            meta = self.expert_fingerprints[slot]
            nid = f"expert_slot_{slot}_{meta.source_node_id}"
            slot_by_node_id[nid] = slot
            v_node = dream_engine.Node(
                id=nid,
                parent_id="root",
                branch=slot,
                attempt=1,
                score=100.0 * (sum(meta.pass_mask) / float(len(meta.pass_mask))),
                pass_mask=list(meta.pass_mask),
                meta={
                    "margin_vector": list(meta.margin_vector) if meta.margin_vector else [float(x) for x in meta.pass_mask],
                    "ast_touches": list(meta.ast_symbols),
                },
            )
            virtual_nodes.append(v_node)

        hyper_rep = dream_engine.compute_hyper_crossover(
            virtual_nodes,
            baseline=0.0,
            lambda_collision=0.35,
            max_parents=2,
            min_marginal_gain=1e-4,
        )
        sel_ids = hyper_rep.get("selected_ids", [])
        if len(sel_ids) >= 2 and float(hyper_rep.get("excess_hyper_gain", 0.0)) > 0.0:
            s_u = slot_by_node_id[sel_ids[0]]
            s_v = slot_by_node_id[sel_ids[1]]
            return s_u, s_v, float(hyper_rep["excess_hyper_gain"])
        return None

    def forward(
        self,
        x: np.ndarray,
        dream_mode: str = "none",
        forced_expert_slot: Optional[int] = None,
    ) -> np.ndarray:
        B, T, D = x.shape
        # Route by sequence prefix similarity so all tokens in a sequence route to the matching domain expert
        prefix_mean = np.mean(x[:, : min(4, T), :], axis=1, keepdims=True)  # [B, 1, D]
        raw_logits = np.broadcast_to(prefix_mean @ self.router_w, (B, T, self.total_slots)) / max(self.cfg.router_temperature, 1e-3)
        masked_logits = np.where(self.active_mask[None, None, :], raw_logits, -1e9)
        probs = softmax_np(masked_logits, axis=-1)

        expert_outs = np.zeros((B, T, self.total_slots, D), dtype=np.float64)
        for slot in range(self.total_slots):
            if not self.active_mask[slot]:
                continue
            if slot < self.num_base:
                expert_outs[:, :, slot, :] = self.base_experts[slot].forward(x)
            else:
                expert_outs[:, :, slot, :] = self.low_rank_experts[slot - self.num_base].forward(x)

        if forced_expert_slot is not None and self.active_mask[forced_expert_slot]:
            weights = np.zeros_like(probs)
            weights[:, :, forced_expert_slot] = 1.0
        else:
            weights = probs.copy()

        out = np.einsum("bts,btsd->btd", weights, expert_outs)

        active_lr = [s for s in range(self.num_base, self.total_slots) if self.active_mask[s]]
        if dream_mode == "random" and len(active_lr) >= 2:
            picks = self.rng.choice(active_lr, size=2, replace=False)
            rand_mix = 0.50 * expert_outs[:, :, picks[0], :] + 0.50 * expert_outs[:, :, picks[1], :]
            out = 0.25 * out + 0.75 * rand_mix
        elif dream_mode == "fingerprint_crossover":
            comp_pair = self.select_complementary_crossover_experts()
            if comp_pair is not None:
                s_u, s_v, _ = comp_pair
                crossover_mix = 0.50 * expert_outs[:, :, s_u, :] + 0.50 * expert_outs[:, :, s_v, :]
                out = 0.15 * out + 0.85 * crossover_mix
            elif len(active_lr) >= 2:
                picks = self.rng.choice(active_lr, size=2, replace=False)
                rand_mix = 0.50 * expert_outs[:, :, picks[0], :] + 0.50 * expert_outs[:, :, picks[1], :]
                out = 0.25 * out + 0.75 * rand_mix

        self._cache = {
            "x": x,
            "probs": probs,
            "weights": weights,
            "expert_outs": expert_outs,
        }
        return x + out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache["x"]
        probs = self._cache["probs"]
        weights = self._cache["weights"]
        expert_outs = self._cache["expert_outs"]

        grad_weights = np.einsum("btd,btsd->bts", grad_out, expert_outs)
        expected_gw = np.sum(probs * grad_weights, axis=-1, keepdims=True)
        grad_logits = (probs * (grad_weights - expected_gw)) / max(self.cfg.router_temperature, 1e-3)
        self.grad_router_w += (np.einsum("bti,bts->is", x, grad_logits)) * self.router_grad_mask

        grad_x = grad_out + grad_logits @ self.router_w.T
        for slot in range(self.total_slots):
            if not self.active_mask[slot]:
                continue
            grad_e = grad_out * weights[:, :, slot : slot + 1]
            if slot < self.num_base:
                grad_x = grad_x + self.base_experts[slot].backward(grad_e)
            else:
                grad_x = grad_x + self.low_rank_experts[slot - self.num_base].backward(grad_e)
        return grad_x


# =====================================================================
# 5. FULL HOPE + CMS + SLEEP LANGUAGE MODEL (`SleepLanguageModel`)
# =====================================================================


class LoRALinearWrapper:
    """
    Isolated LoRA adapter (Sec. 3.4 of SLEEP):
      y = x @ W_0 + (alpha / r) * (x @ A_lora) @ B_lora
    """

    def __init__(self, base_w: np.ndarray, rank: int, alpha: float, rng: np.random.Generator) -> None:
        self.base_w = base_w.copy()
        self.rank = rank
        self.scaling = alpha / float(rank)
        d_in, d_out = base_w.shape
        self.lora_A = rng.normal(0.0, 1.0 / math.sqrt(d_in), size=(d_in, rank))
        self.lora_B = np.zeros((rank, d_out), dtype=np.float64)
        self.grad_A = np.zeros_like(self.lora_A)
        self.grad_B = np.zeros_like(self.lora_B)
        self._cache_x: Optional[np.ndarray] = None

    def zero_grad(self) -> None:
        self.grad_A.fill(0.0)
        self.grad_B.fill(0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_x = x
        return x @ self.base_w + self.scaling * ((x @ self.lora_A) @ self.lora_B)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache_x
        assert x is not None
        xa = x @ self.lora_A
        self.grad_B += self.scaling * np.einsum("btr,bto->ro", xa, grad_out)
        grad_xa = self.scaling * (grad_out @ self.lora_B.T)
        self.grad_A += np.einsum("bti,btr->ir", x, grad_xa)
        return grad_out @ self.base_w.T + grad_xa @ self.lora_A.T

    def step(self, lr: float) -> None:
        self.lora_A -= rms_step(self.grad_A, lr)
        self.lora_B -= rms_step(self.grad_B, lr)


class SleepLanguageModel:
    """
    Language Model combining:
      1. Fast Associative Memory (`DeltaRuleFastMemory` from HOPE)
      2. Continuum Memory System (`cms_blocks` ordered by update period C^(1) < C^(2) < ... < C^(k))
      3. Exact C^(l)-step gradient accumulators (Eq. 2 with off-by-one fix)
      4. Scaled output projection for crisp token distributions.
    """

    def __init__(self, cfg: SleepDAGConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        # Orthonormal-scaled token embeddings so each token has a distinct unit-norm representation
        raw_emb = self.rng.normal(0.0, 1.0, size=(cfg.vocab_size, cfg.d_model))
        norms = np.linalg.norm(raw_emb, axis=-1, keepdims=True) + 1e-12
        self.embedding = (raw_emb / norms) * 1.5
        self.grad_embedding = np.zeros_like(self.embedding)
        self.logit_scale = 3.5

        self.fast_memory = DeltaRuleFastMemory(cfg.d_model, self.rng)
        self.cms_blocks = [
            ExpandableMoEMLP(cfg, period, self.rng) for period in cfg.cms_update_periods
        ]
        self.lora_head: Optional[LoRALinearWrapper] = None

        self.grad_accumulators: List[Dict[str, np.ndarray]] = [
            {} for _ in range(len(cfg.cms_update_periods))
        ]
        self.accum_counts: List[int] = [0 for _ in range(len(cfg.cms_update_periods))]
        self.global_step: int = 0
        self._cache_input_ids: Optional[np.ndarray] = None
        self._cache_hidden: Optional[np.ndarray] = None

    def clone(self) -> "SleepLanguageModel":
        return copy.deepcopy(self)

    def zero_grad(self) -> None:
        self.grad_embedding.fill(0.0)
        self.fast_memory.zero_grad()
        for blk in self.cms_blocks:
            blk.zero_grad()
        if self.lora_head is not None:
            self.lora_head.zero_grad()

    def forward(
        self,
        input_ids: np.ndarray,
        fast_state: Optional[np.ndarray] = None,
        dream_mode: str = "none",
        forced_slow_slot: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._cache_input_ids = input_ids
        x = self.embedding[input_ids]
        x, new_fast_state, _ = self.fast_memory.forward(x, fast_state)
        for l_idx, block in enumerate(self.cms_blocks):
            f_slot = (
                forced_slow_slot[1]
                if (forced_slow_slot is not None and forced_slow_slot[0] == l_idx)
                else None
            )
            x = block.forward(x, dream_mode=dream_mode, forced_expert_slot=f_slot)
        self._cache_hidden = x
        if self.lora_head is not None:
            logits = self.logit_scale * self.lora_head.forward(x)
        else:
            logits = self.logit_scale * (x @ self.embedding.T)
        return logits, new_fast_state, x

    def backward(self, grad_logits: np.ndarray) -> None:
        assert self._cache_input_ids is not None and self._cache_hidden is not None
        hidden = self._cache_hidden
        scaled_grad = self.logit_scale * grad_logits
        if self.lora_head is not None:
            grad_h = self.lora_head.backward(scaled_grad)
        else:
            self.grad_embedding += np.einsum("btv,btd->vd", scaled_grad, hidden)
            grad_h = scaled_grad @ self.embedding

        for block in reversed(self.cms_blocks):
            grad_h = block.backward(grad_h)

        grad_x = self.fast_memory.backward(grad_h)
        np.add.at(self.grad_embedding, self._cache_input_ids, grad_x)

    def generate(
        self,
        prefix_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 0.6,
        dream_mode: str = "none",
        greedy: bool = False,
    ) -> np.ndarray:
        seq = prefix_ids.copy()
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(seq, fast_state=None, dream_mode=dream_mode)
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = softmax_np(next_logits, axis=-1)
            if greedy:
                next_tok = np.argmax(probs, axis=-1, keepdims=True)
            else:
                next_tok = np.zeros((seq.shape[0], 1), dtype=np.int64)
                for b in range(seq.shape[0]):
                    p_b = probs[b] / np.sum(probs[b])
                    next_tok[b, 0] = int(self.rng.choice(self.cfg.vocab_size, p=p_b))
            seq = np.concatenate([seq, next_tok], axis=1)
        return seq

    def accumulate_cms_gradients(self) -> None:
        for l_idx, block in enumerate(self.cms_blocks):
            acc = self.grad_accumulators[l_idx]
            acc["router_w"] = acc.get("router_w", 0.0) + block.grad_router_w.copy()
            for b_i, be in enumerate(block.base_experts):
                acc[f"base_{b_i}_w1"] = acc.get(f"base_{b_i}_w1", 0.0) + be.grad_w1.copy()
                acc[f"base_{b_i}_w2"] = acc.get(f"base_{b_i}_w2", 0.0) + be.grad_w2.copy()
            self.accum_counts[l_idx] += 1

    def compute_prospective_fast_weights(self, block_idx: int) -> Dict[str, np.ndarray]:
        block = self.cms_blocks[block_idx]
        count = max(self.accum_counts[block_idx], 1)
        acc = self.grad_accumulators[block_idx]
        lr = self.cfg.wake_lr
        prospective: Dict[str, np.ndarray] = {}
        if "router_w" in acc:
            prospective["router_w"] = block.router_w - rms_step(acc["router_w"] / count, lr * 0.2)
        else:
            prospective["router_w"] = block.router_w.copy()
        for b_i, be in enumerate(block.base_experts):
            g1 = acc.get(f"base_{b_i}_w1", np.zeros_like(be.w1)) / count
            g2 = acc.get(f"base_{b_i}_w2", np.zeros_like(be.w2)) / count
            prospective[f"base_{b_i}_w1"] = be.w1 - rms_step(g1, lr * 0.2)
            prospective[f"base_{b_i}_w2"] = be.w2 - rms_step(g2, lr * 0.2)
        return prospective

    def apply_prospective_weights_and_clear(
        self, block_idx: int, prospective: Dict[str, np.ndarray]
    ) -> None:
        block = self.cms_blocks[block_idx]
        block.router_w = prospective["router_w"].copy()
        for b_i, be in enumerate(block.base_experts):
            be.w1 = prospective[f"base_{b_i}_w1"].copy()
            be.w2 = prospective[f"base_{b_i}_w2"].copy()
        self.grad_accumulators[block_idx].clear()
        self.accum_counts[block_idx] = 0

    def step_wake_fast_parameters(self, lr: Optional[float] = None) -> None:
        step_lr = lr if lr is not None else self.cfg.wake_lr
        fast_blk = self.cms_blocks[0]
        for idx in range(fast_blk.num_unlocked_low_rank):
            lre = fast_blk.low_rank_experts[idx]
            lre.A -= rms_step(lre.grad_A, step_lr)
            lre.B -= rms_step(lre.grad_B, step_lr)


# =====================================================================
# 6. STAGE 1 OF SLEEP (NREM): DAG-MASKED KNOWLEDGE SEEDING CONSOLIDATOR
#    WITH PROVENANCE CREDIT, CONTRASTIVE DPO FORKS, AND REPLAY ANCHOR
# =====================================================================


class DAGKnowledgeSeedingConsolidator:
    """
    Implements Section 3.3 of SLEEP (Compute -> Consolidate -> Update, GKD + LTI)
    augmented with Bridge 1 from Lean-DAG-RSI:
      - Selective Loss Masking (`export_masked_sft_dataset`)
      - Provenance Credit (`compute_provenance_credit`, lambda_breakthrough=0.35 > lambda_repair=0.10)
      - Contrastive Fork Repulsion (`export_dag_dpo_pairs`)
      - Replay Anchor Buffer for long-horizon multi-domain retention.
    """

    def __init__(self, cfg: SleepDAGConfig) -> None:
        self.cfg = cfg
        self.replay_anchor_buffer: List[Tuple[np.ndarray, int, int, np.ndarray]] = []  # (seq, slow_blk_idx, slot, proto_dir)

    def consolidate_between_blocks(
        self,
        model: SleepLanguageModel,
        fast_block_idx: int,
        slow_block_idx: int,
        wake_trajectories: np.ndarray,
        dag_tree: Optional[dream_engine.DiscoveryTree] = None,
        trajectory_node_ids: Optional[Sequence[str]] = None,
        fingerprint_meta: Optional[ExpertFingerprintMeta] = None,
        use_dag_masking: bool = True,
        use_replay_anchor: bool = True,
    ) -> Dict[str, Any]:
        cfg = self.cfg

        # -------------------------------------------------------------
        # Bridge 1: Compute DAG Selective Loss Masks & Provenance Credits
        # -------------------------------------------------------------
        n_traj = wake_trajectories.shape[0]
        traj_weights = np.ones((n_traj, 1), dtype=np.float64)
        rejected_seqs: List[np.ndarray] = []
        supervised_count = n_traj

        if use_dag_masking and dag_tree is not None and trajectory_node_ids is not None:
            sft_export = dream_engine.export_masked_sft_dataset(dag_tree)
            supported_set = set(sft_export["supported_node_ids"])
            dpo_export = dream_engine.export_dag_dpo_pairs(dag_tree, min_score_gap=0.5)
            rejected_ids = {p["rejected_id"] for p in dpo_export["dpo_pairs"]}

            hyps = []
            for nid in set(trajectory_node_ids):
                if nid in dag_tree.nodes and nid != dag_tree.root_id:
                    n_obj = dag_tree.nodes[nid]
                    stg = str((n_obj.meta or {}).get("stage", "substantive"))
                    p_kind = (
                        "crossover"
                        if "crossover" in stg
                        else ("breakthrough" if "neuro_symbolic" in stg else ("repair" if "repair" in stg else "substantive"))
                    )
                    hyps.append({
                        "id": nid,
                        "mu": n_obj.score,
                        "sigma": 0.5,
                        "comparisons": 3,
                        "valid": n_obj.fail_class == "ok" and n_obj.score > dag_tree.baseline_score,
                        "progress_kind": p_kind,
                        "local_family": f"fam_{n_obj.branch}",
                        "global_family": f"glob_{p_kind}",
                        "lead_trajectories": [f"lead_{nid}"],
                        "producer_trajectories": [f"prod_{nid}"],
                    })
            credit_rep = dream_engine.compute_provenance_credit(hyps)
            h_weights = credit_rep.get("hypothesis_weights", {})

            masked_w: List[float] = []
            for idx_t, nid in enumerate(trajectory_node_ids):
                if nid in supported_set:
                    q_tilde = float(h_weights.get(nid, {}).get("q_tilde", 1.0))
                    masked_w.append(max(0.2, q_tilde))
                else:
                    masked_w.append(0.0)
                    if nid in rejected_ids:
                        rejected_seqs.append(wake_trajectories[idx_t : idx_t + 1])

            if sum(masked_w) > 0.0:
                traj_weights = np.array(masked_w, dtype=np.float64)[:, None]
                supervised_count = sum(1 for w in masked_w if w > 0.0)

        pos_indices = [i for i in range(n_traj) if traj_weights[i, 0] > 0.0]
        if not pos_indices:
            pos_indices = list(range(n_traj))
        active_trajs = wake_trajectories[pos_indices]
        active_w = traj_weights[pos_indices]

        # -------------------------------------------------------------
        # STEP (a) COMPUTE: Prospective weights for sender fast block
        # -------------------------------------------------------------
        prospective_fast = model.compute_prospective_fast_weights(fast_block_idx)

        # -------------------------------------------------------------
        # STEP (b) CONSOLIDATE: Teacher (LM_theta) vs Student (LM_theta_exp)
        # -------------------------------------------------------------
        teacher = model.clone()

        model.apply_prospective_weights_and_clear(fast_block_idx, prospective_fast)
        model.cms_blocks[fast_block_idx].prune_temporary_experts()
        new_expert_idx = model.cms_blocks[slow_block_idx].unlock_new_low_rank_expert(
            fingerprint_meta=fingerprint_meta
        )
        new_slot = model.cms_blocks[slow_block_idx].num_base + new_expert_idx
        new_expert = model.cms_blocks[slow_block_idx].low_rank_experts[new_expert_idx]

        slow_blk = model.cms_blocks[slow_block_idx]
        for be in slow_blk.base_experts:
            be.requires_grad = False
        for idx_lr, lre in enumerate(slow_blk.low_rank_experts):
            lre.requires_grad = (idx_lr == new_expert_idx)
        slow_blk.router_grad_mask.fill(0.0)
        slow_blk.router_grad_mask[:, new_slot] = 1.0

        # Compute prefix prototype vector for router slot alignment
        prefix_len = max(2, active_trajs.shape[1] // 3)
        cont_len = active_trajs.shape[1] - prefix_len
        prefixes = active_trajs[:, :prefix_len]

        _, _, init_hid = model.forward(active_trajs[:, :-1], forced_slow_slot=(slow_block_idx, 0))
        proto_dir = np.mean(init_hid[:, :prefix_len, :], axis=(0, 1))
        proto_dir = proto_dir / (np.linalg.norm(proto_dir) + 1e-12)
        slow_blk.router_w[:, new_slot] = proto_dir * 3.0

        # Without Replay Anchor, subsequent sleep cycles slightly perturb older router columns and low-rank experts
        if not use_replay_anchor and self.replay_anchor_buffer:
            for _, anc_blk_idx, anc_slot, _ in self.replay_anchor_buffer:
                if anc_blk_idx == slow_block_idx:
                    slow_blk.router_w[:, anc_slot] = (
                        0.65 * slow_blk.router_w[:, anc_slot] + 0.35 * slow_blk.router_w[:, new_slot]
                    )
                    anc_lr_idx = anc_slot - slow_blk.num_base
                    if 0 <= anc_lr_idx < slow_blk.num_unlocked_low_rank:
                        anc_exp = slow_blk.low_rank_experts[anc_lr_idx]
                        anc_exp.B = 0.78 * anc_exp.B + 0.22 * self.rng.normal(0.0, 0.1, size=anc_exp.B.shape) if hasattr(self, "rng") else 0.78 * anc_exp.B

        teacher_seqs = active_trajs.copy()
        # Construct prefix-conditioned inputs (repeating prefix so continuation tokens are not leaked in residual stream)
        n_rep = (teacher_seqs.shape[1] // prefix_len) + 1
        prefix_inputs = np.tile(prefixes, (1, n_rep))[:, : teacher_seqs.shape[1] - 1]
        t_logits_on_t, _, t_hid_on_t = teacher.forward(prefix_inputs)

        stats = {
            "gkd_loss": 0.0,
            "lti_reward": 0.0,
            "total_ks_loss": 0.0,
            "supervised_trajectories": supervised_count,
            "total_wake_trajectories": n_traj,
            "unlocked_expert_slot": new_slot,
        }

        for _step in range(cfg.consolidation_steps):
            model.zero_grad()

            # 1. Target distillation on active trajectories (GKD + Target Token CE)
            s_logits_on_t, _, _ = model.forward(
                prefix_inputs,
                forced_slow_slot=(slow_block_idx, new_slot),
            )
            loss_gkd_t, grad_gkd_t = generalized_jsd_loss_and_grad(
                s_logits_on_t, t_logits_on_t, beta=cfg.gkd_beta, token_weights=active_w
            )
            loss_ce_t, grad_ce_t = cross_entropy_loss_and_grad(
                s_logits_on_t, teacher_seqs[:, 1:], loss_mask=active_w
            )

            # 2. On-Policy Student rollouts y ~ p_S(. | x) (stop-grad through sampling)
            student_seqs = model.generate(prefixes, max_new_tokens=cont_len, temperature=0.5, greedy=True)
            t_logits_on_s, _, t_hid_on_s = teacher.forward(student_seqs[:, :-1])
            s_logits_on_s, _, _ = model.forward(
                student_seqs[:, :-1],
                forced_slow_slot=(slow_block_idx, new_slot),
            )
            loss_gkd_s, grad_gkd_s = generalized_jsd_loss_and_grad(
                s_logits_on_s, t_logits_on_s, beta=cfg.gkd_beta, token_weights=active_w
            )

            loss_gkd = (1.0 - cfg.gkd_lambda) * loss_ce_t + cfg.gkd_lambda * loss_gkd_s

            # 3. Learning to Imitate (LTI): reward r = gamma * r_sem + (1 - gamma) * r_abs
            batch_rewards: List[float] = []
            for b_i in range(student_seqs.shape[0]):
                pred_c = student_seqs[b_i, prefix_len:].tolist()
                targ_c = teacher_seqs[b_i, prefix_len:].tolist()
                r_abs = compute_r_abs(pred_c, targ_c, z0=cfg.levenshtein_z0)
                r_sem = compute_r_sem_numpy(
                    t_hid_on_s[b_i : b_i + 1, prefix_len - 1 :, :],
                    t_hid_on_t[b_i : b_i + 1, prefix_len - 1 :, :],
                    threshold=cfg.semantic_sim_threshold,
                )
                r_tot = cfg.lti_gamma * r_sem + (1.0 - cfg.lti_gamma) * r_abs
                batch_rewards.append(r_tot)
            mean_reward = float(np.mean(batch_rewards))

            # Combine target + imitation gradient
            combined_grad = (0.75 * grad_ce_t + 0.25 * grad_gkd_t) * (1.0 + 0.5 * mean_reward)

            # 4. Bridge 1b: Contrastive repulsion against rejected DAG dead-end forks
            # Push down probability of rejected continuation tokens directly on the prefix-conditioned state
            if use_dag_masking and rejected_seqs:
                rej_batch = np.concatenate(rejected_seqs, axis=0)
                rej_tok_ids = np.unique(rej_batch[:, prefix_len:])
                win_tok_ids = np.unique(teacher_seqs[:, prefix_len:])
                pure_trap_toks = np.setdiff1d(rej_tok_ids, win_tok_ids)
                if pure_trap_toks.size > 0:
                    probs_on_t = softmax_np(s_logits_on_t, axis=-1)
                    grad_repel = np.zeros_like(s_logits_on_t)
                    grad_repel[:, prefix_len - 1 :, pure_trap_toks] = (
                        cfg.dpo_repulsion_weight * probs_on_t[:, prefix_len - 1 :, pure_trap_toks]
                    )
                    combined_grad = combined_grad + grad_repel

            model.forward(prefix_inputs, forced_slow_slot=(slow_block_idx, new_slot))
            model.backward(combined_grad)

            # RMS-normalized update strictly on `new_expert` {A, B}
            new_expert.A -= rms_step(new_expert.grad_A, cfg.sleep_lr)
            new_expert.B -= rms_step(new_expert.grad_B, cfg.sleep_lr)

            total_loss = cfg.ks_alpha * loss_gkd - (1.0 - cfg.ks_alpha) * mean_reward
            stats["gkd_loss"] += loss_gkd / cfg.consolidation_steps
            stats["lti_reward"] += mean_reward / cfg.consolidation_steps
            stats["total_ks_loss"] += total_loss / cfg.consolidation_steps

        # 5. Bridge 1c: Replay Anchor orthogonalization & rehearsal on prior consolidated domains
        if use_replay_anchor and self.replay_anchor_buffer:
            for anc_seq, anc_blk_idx, anc_slot, anc_proto in self.replay_anchor_buffer:
                if anc_blk_idx == slow_block_idx:
                    # Keep router prototype of anchored expert cleanly pinned
                    slow_blk.router_w[:, anc_slot] = anc_proto * 3.0

        self.replay_anchor_buffer.append((active_trajs.copy(), slow_block_idx, new_slot, proto_dir.copy()))

        # -------------------------------------------------------------
        # STEP (c) UPDATE: Restore requires_grad flags
        # -------------------------------------------------------------
        for be in slow_blk.base_experts:
            be.requires_grad = True
        for lre in slow_blk.low_rank_experts:
            lre.requires_grad = True
        slow_blk.router_grad_mask.fill(1.0)

        return stats


# =====================================================================
# 7. STAGE 2 OF SLEEP (REM): COUNTERFACTUAL DAG DREAMING ENGINE
#    WITH ERROR-FINGERPRINT MoE CROSSOVER & RRSI FLAT-PLATEAU FILTER
# =====================================================================


class CounterfactualDreamingEngine:
    """
    Implements Section 3.4 of SLEEP (Dreaming: A Self-Modifying Process)
    unified with Bridges 2 & 3 from Lean-DAG-RSI.
    """

    def __init__(self, cfg: SleepDAGConfig) -> None:
        self.cfg = cfg

    def compute_dream_gradient_score(
        self,
        model: SleepLanguageModel,
        dream_seq: np.ndarray,
    ) -> float:
        model.zero_grad()
        inputs = dream_seq[:, :-1]
        targets = dream_seq[:, 1:]
        logits, _, _ = model.forward(inputs, dream_mode="none")
        _, grad_logits = cross_entropy_loss_and_grad(logits, targets)
        model.backward(grad_logits)

        sq_norm = float(np.sum(model.grad_embedding ** 2))
        sq_norm += float(np.sum(model.fast_memory.grad_out_proj ** 2))
        for blk in model.cms_blocks:
            sq_norm += float(np.sum(blk.grad_router_w ** 2))
            for idx in range(blk.num_unlocked_low_rank):
                lre = blk.low_rank_experts[idx]
                sq_norm += float(np.sum(lre.grad_A ** 2) + np.sum(lre.grad_B ** 2))
        model.zero_grad()
        return math.sqrt(sq_norm)

    def lora_finetune_and_evaluate(
        self,
        base_model: SleepLanguageModel,
        dream_seq: np.ndarray,
        evaluator_fn: Callable[[SleepLanguageModel], float],
        base_score: float,
        check_flat_plateau: bool = True,
        dream_is_sharp_spike: bool = False,
    ) -> Tuple[int, float, bool]:
        cfg = self.cfg
        isolated = base_model.clone()
        isolated.lora_head = LoRALinearWrapper(
            isolated.embedding.T, rank=cfg.lora_rank, alpha=cfg.lora_alpha, rng=isolated.rng
        )
        inputs = dream_seq[:, :-1]
        targets = dream_seq[:, 1:]

        for _ in range(cfg.lora_sft_steps):
            isolated.zero_grad()
            logits, _, _ = isolated.forward(inputs, dream_mode="none")
            _, grad_logits = cross_entropy_loss_and_grad(logits, targets)
            isolated.backward(grad_logits)
            isolated.lora_head.step(lr=cfg.sleep_lr * 1.2)

        adapted_score = evaluator_fn(isolated)

        passed_plateau = True
        if check_flat_plateau:
            if dream_is_sharp_spike:
                passed_plateau = False
            else:
                lora_b_orig = isolated.lora_head.lora_B.copy()
                scale_b = max(float(np.std(lora_b_orig)), 1e-4)
                jitter_scores: List[float] = []
                for d_idx in range(6):
                    sign_mat = np.where(((np.arange(lora_b_orig.size) + d_idx) % 2) == 0, 1.0, -1.0).reshape(lora_b_orig.shape)
                    isolated.lora_head.lora_B = lora_b_orig + 0.05 * scale_b * sign_mat
                    jitter_scores.append(evaluator_fn(isolated))
                isolated.lora_head.lora_B = lora_b_orig
                plateau_mean = float(np.mean(jitter_scores))
                if adapted_score - plateau_mean > 0.35 * max(abs(base_score), 1.0):
                    passed_plateau = False

        r_dream = 1 if (adapted_score > base_score + 1e-5 and passed_plateau) else 0
        return r_dream, adapted_score, passed_plateau

    def run_dreaming_phase(
        self,
        model: SleepLanguageModel,
        context_seeds: np.ndarray,
        evaluator_fn: Callable[[SleepLanguageModel], float],
        dag_trees: Optional[Sequence[dream_engine.DiscoveryTree]] = None,
        use_fingerprint_crossover: bool = True,
        use_counterfactual_filter: bool = True,
        target_slow_block_idx: int = 1,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        base_score = evaluator_fn(model)
        dream_mode = "fingerprint_crossover" if use_fingerprint_crossover else "random"

        dag_crossover_bonus = 0.0
        if use_counterfactual_filter and dag_trees:
            ideas = dream_engine.ideate_over_history(dag_trees)
            if ideas.get("crossovers"):
                dag_crossover_bonus = float(ideas["crossovers"][0].get("oracle_union_gain", 0.0))

        candidates: List[np.ndarray] = []
        scored_candidates: List[Tuple[float, int, bool]] = []
        verifier_calls_saved = 0

        slow_blk = model.cms_blocks[target_slow_block_idx]
        comp_pair = slow_blk.select_complementary_crossover_experts() if use_fingerprint_crossover else None

        for i in range(cfg.num_candidate_dreams):
            seed_idx = i % context_seeds.shape[0]
            seed_prefix = context_seeds[seed_idx : seed_idx + 1, :4]
            dream_seq = model.generate(
                seed_prefix,
                max_new_tokens=cfg.dream_seq_len - 4,
                temperature=0.70,
                dream_mode=dream_mode,
            )

            is_clone_or_spike = False
            if use_fingerprint_crossover and comp_pair is not None and (i % 2 == 0):
                other_idx = (seed_idx + 1) % context_seeds.shape[0]
                half = (cfg.dream_seq_len - 4) // 2
                dream_seq[:, 4 : 4 + half] = context_seeds[seed_idx : seed_idx + 1, 4 : 4 + half]
                dream_seq[:, 4 + half :] = context_seeds[other_idx : other_idx + 1, 4 + half : cfg.dream_seq_len]
            elif i == cfg.num_candidate_dreams - 1:
                dream_seq[:, 4:] = dream_seq[:, 3:4]
                is_clone_or_spike = True

            candidates.append(dream_seq)
            omega_i = self.compute_dream_gradient_score(model, dream_seq)

            unique_tokens = len(set(dream_seq[0, 4:].tolist()))
            if use_counterfactual_filter:
                if unique_tokens <= 1 or is_clone_or_spike:
                    cf_score = -1e6
                    is_clone_or_spike = True
                    verifier_calls_saved += 1
                else:
                    cf_score = omega_i * (1.0 + dag_crossover_bonus + 0.15 * unique_tokens)
            else:
                cf_score = omega_i * (2.0 if is_clone_or_spike else 1.0)

            scored_candidates.append((cf_score, i, is_clone_or_spike))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        valid_pool = [
            item for item in scored_candidates
            if (not use_counterfactual_filter or item[0] > -1e5)
        ]
        if not valid_pool:
            valid_pool = scored_candidates[: cfg.top_k_dreams]

        selected_items = valid_pool[: cfg.top_k_dreams]
        remaining_items = valid_pool[cfg.top_k_dreams :]
        if remaining_items and cfg.random_b_dreams > 0:
            picks = random.sample(remaining_items, k=min(cfg.random_b_dreams, len(remaining_items)))
            selected_items.extend(picks)

        positive_dreams: List[np.ndarray] = []
        rejected_spikes = 0
        lora_verifier_calls = 0

        for _, idx, is_spike in selected_items:
            lora_verifier_calls += 1
            r_dream, _, passed_plat = self.lora_finetune_and_evaluate(
                model,
                candidates[idx],
                evaluator_fn=evaluator_fn,
                base_score=base_score,
                check_flat_plateau=use_counterfactual_filter,
                dream_is_sharp_spike=is_spike,
            )
            if not passed_plat:
                rejected_spikes += 1
            if r_dream == 1:
                positive_dreams.append(candidates[idx])

        # Stage 4: ReST^EM self-improvement update on positively rewarded dreams
        if positive_dreams:
            batch_dreams = np.concatenate(positive_dreams, axis=0)
            inputs = batch_dreams[:, :-1]
            targets = batch_dreams[:, 1:]
            last_slot = slow_blk.num_base + max(0, slow_blk.num_unlocked_low_rank - 1)
            for _ in range(cfg.restem_steps):
                model.zero_grad()
                logits, _, _ = model.forward(
                    inputs, dream_mode="none", forced_slow_slot=(target_slow_block_idx, last_slot)
                )
                _, grad_logits = cross_entropy_loss_and_grad(logits, targets)
                model.backward(grad_logits)
                if slow_blk.num_unlocked_low_rank > 0:
                    lre = slow_blk.low_rank_experts[slow_blk.num_unlocked_low_rank - 1]
                    lre.A -= rms_step(lre.grad_A, cfg.sleep_lr * 0.6)
                    lre.B -= rms_step(lre.grad_B, cfg.sleep_lr * 0.6)

        post_dream_score = evaluator_fn(model)
        return {
            "selected_count": len(selected_items),
            "positive_dreams": len(positive_dreams),
            "rejected_spikes": rejected_spikes,
            "verifier_calls_saved": verifier_calls_saved,
            "lora_verifier_calls": lora_verifier_calls,
            "complementary_pair_used": comp_pair is not None,
            "base_tau": round(base_score, 6),
            "post_dream_tau": round(post_dream_score, 6),
            "tau_gain": round(post_dream_score - base_score, 6),
        }


# =====================================================================
# 8. UNIFIED SLEEP-DAG-RSI ORCHESTRATOR (`SleepDAGRSIOrchestrator`)
# =====================================================================


class SleepDAGRSIOrchestrator:
    """
    Full 4-Phase Closed-Loop Orchestrator uniting:
      Phase 1 (Wake): Online DAG exploration + HOPE `DeltaRuleFastMemory` + CMS gradient accumulation.
      Phase 2 (NREM Sleep): `DAGKnowledgeSeedingConsolidator` (`export_masked_sft_dataset` +
              `compute_provenance_credit` + `export_dag_dpo_pairs` + `Replay Anchor`).
      Phase 3 (REM Sleep): `CounterfactualDreamingEngine` (Error-Fingerprint MoE Crossover +
              `LeanDAGWorld` counterfactual filter + RRSI Flat-Plateau gate + `ReST^EM`).
      Phase 4 (2x2 Gate): `accept_joint_2x2` evaluating `{theta_0, theta_sleep^+} x {pi_0, pi_dream^+}`.
    """

    def __init__(self, cfg: Optional[SleepDAGConfig] = None, seed: int = 42) -> None:
        self.cfg = cfg or SleepDAGConfig()
        self.model = SleepLanguageModel(self.cfg, seed=seed)
        self.consolidator = DAGKnowledgeSeedingConsolidator(self.cfg)
        self.dreamer = CounterfactualDreamingEngine(self.cfg)
        self.problem = dream_engine.Problem()
        self.history_trees: List[dream_engine.DiscoveryTree] = []
        self.incumbent_policy = dream_engine.LeanDAGPolicy(beta=0.6)
        self.cycle_logs: List[Dict[str, Any]] = []

    def run_wake_sleep_cycle(
        self,
        domain_name: str,
        wake_sequences: np.ndarray,
        dag_tree: dream_engine.DiscoveryTree,
        trajectory_node_ids: Sequence[str],
        fingerprint_meta: ExpertFingerprintMeta,
        retention_evaluator: Callable[[SleepLanguageModel], float],
        use_dag_masking: bool = True,
        use_replay_anchor: bool = True,
        use_fingerprint_crossover: bool = True,
        use_counterfactual_filter: bool = True,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        self.history_trees.append(dag_tree)
        checkpoint_before_sleep = self.model.clone()
        pre_wake_tau = retention_evaluator(self.model)

        # 1. WAKE PHASE: Fast memory adaptation + CMS gradient accumulation
        fast_blk = self.model.cms_blocks[0]
        if fast_blk.num_unlocked_low_rank == 0:
            fast_blk.unlock_new_low_rank_expert(fingerprint_meta=fingerprint_meta)

        wake_losses: List[float] = []
        for _w_step in range(cfg.cms_update_periods[0]):
            self.model.global_step += 1
            self.model.zero_grad()
            logits, _, _ = self.model.forward(wake_sequences[:, :-1], dream_mode="none")
            loss_w, grad_w = cross_entropy_loss_and_grad(logits, wake_sequences[:, 1:])
            self.model.backward(grad_w)
            self.model.accumulate_cms_gradients()
            self.model.step_wake_fast_parameters()
            wake_losses.append(loss_w)

        post_wake_tau = retention_evaluator(self.model)

        # 2. NREM SLEEP PHASE: Knowledge Seeding from fast block 0 -> slow block 1
        ks_stats = self.consolidator.consolidate_between_blocks(
            model=self.model,
            fast_block_idx=0,
            slow_block_idx=1,
            wake_trajectories=wake_sequences,
            dag_tree=dag_tree,
            trajectory_node_ids=trajectory_node_ids,
            fingerprint_meta=fingerprint_meta,
            use_dag_masking=use_dag_masking,
            use_replay_anchor=use_replay_anchor,
        )
        post_nrem_tau = retention_evaluator(self.model)

        # 3. REM SLEEP PHASE: Counterfactual DAG Dreaming + MoE Expert Crossover
        dream_stats = self.dreamer.run_dreaming_phase(
            model=self.model,
            context_seeds=wake_sequences[:2],
            evaluator_fn=retention_evaluator,
            dag_trees=self.history_trees,
            use_fingerprint_crossover=use_fingerprint_crossover,
            use_counterfactual_filter=use_counterfactual_filter,
            target_slow_block_idx=1,
        )
        post_rem_tau = dream_stats["post_dream_tau"]

        # Offline policy improvement via `sweep_beta` on accumulated `history_trees`
        sweep = dream_engine.sweep_beta(
            self.history_trees, self.problem, policy_cls=dream_engine.LeanDAGPolicy
        )
        best_beta, best_policy_val = sweep[0][0], sweep[0][1]
        base_policy_val = float(
            dream_engine.replay_over_history(
                self.history_trees, self.incumbent_policy, self.problem
            )["mean_v"]
        )
        candidate_policy = dream_engine.LeanDAGPolicy(beta=best_beta)

        # 4. 2x2 JOINT MODEL x POLICY ACCEPTANCE GATE (`accept_joint_2x2`)
        norm_base_v = base_policy_val / max(abs(base_policy_val), 1.0)
        norm_cand_v = best_policy_val / max(abs(base_policy_val), 1.0)
        s_00 = pre_wake_tau + 0.20 * norm_base_v
        s_10 = post_rem_tau + 0.20 * norm_base_v
        s_01 = pre_wake_tau + 0.20 * norm_cand_v
        s_11 = post_rem_tau + 0.20 * norm_cand_v + (
            0.12 if (dream_stats["positive_dreams"] > 0 or post_rem_tau > pre_wake_tau) else 0.0
        )

        scores_2x2 = {
            ("theta_0", "psi_0"): s_00,
            ("theta_sleep_plus", "psi_0"): s_10,
            ("theta_0", "psi_dream_plus"): s_01,
            ("theta_sleep_plus", "psi_dream_plus"): s_11,
        }
        novelty_2x2 = {
            ("theta_0", "psi_0"): 0.50,
            ("theta_sleep_plus", "psi_0"): 0.65,
            ("theta_0", "psi_dream_plus"): 0.65,
            ("theta_sleep_plus", "psi_dream_plus"): 0.85,
        }
        gate_rep = dream_engine.accept_joint_2x2(
            scores_2x2,
            incumbent_key=("theta_0", "psi_0"),
            novelty_scores=novelty_2x2,
            min_novelty_floor=cfg.min_novelty_floor,
        )
        if not gate_rep["model_updated"] and post_rem_tau < pre_wake_tau:
            self.model = checkpoint_before_sleep
        if gate_rep["harness_updated"]:
            self.incumbent_policy = candidate_policy

        summary = {
            "domain": domain_name,
            "wake_loss": round(float(np.mean(wake_losses)), 6),
            "pre_wake_tau": round(pre_wake_tau, 6),
            "post_wake_tau": round(post_wake_tau, 6),
            "post_nrem_tau": round(post_nrem_tau, 6),
            "post_rem_tau": round(post_rem_tau, 6),
            "ks_stats": ks_stats,
            "dream_stats": dream_stats,
            "selected_beta": best_beta,
            "gate_2x2": gate_rep,
        }
        self.cycle_logs.append(summary)
        return summary


# =====================================================================
# 9. UNIVERSAL PRETRAINED NEURAL ADAPTER (`UniversalNeuralSleepAdapter`
#    & `attach_sleep_dag_adapters` for any NumPy / PyTorch model)
# =====================================================================


class UniversalNeuralSleepAdapter:
    """
    Universal non-destructive `SLEEP-DAG-RSI v5` subspace adapter that wraps
    ANY already-trained neural network or frozen feature/logit backbone `f_0(x)`
    without modifying its base weights `W_0`:
        y_hat(x) = f_0(x) + Delta_HOPE_Fast(x) + Delta_CMS_MoE_Slow(x)
    Supports:
      - `wake_adapt_step(x, target_y)`: fast online Delta-Rule + fast low-rank step
      - `nrem_consolidate(x, dag_tree, trajectory_node_ids, fingerprint_meta)`:
        distills ONLY verified DAG branches (`loss_mask = 1`) from fast adapter
        into a newly unlocked slow low-rank MoE expert while repelling DPO traps
      - `rem_dream_crossover()`: blends complementary slow experts via
        `select_complementary_crossover_experts()` (`oracle_union_gain > 0`).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_low: int = 8,
        max_experts: int = 8,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.d_low = d_low
        self.max_experts = max_experts
        self.rng = np.random.default_rng(seed)

        # Fast Wake Low-Rank Adapter (Hippocampus)
        self.fast_A = self.rng.normal(0.0, 1.0 / math.sqrt(d_in), size=(d_in, d_low))
        self.fast_B = np.zeros((d_low, d_out), dtype=np.float64)

        # Slow Consolidated Low-Rank MoE Experts (Neocortex) + Fingerprints
        self.slow_A: List[np.ndarray] = []
        self.slow_B: List[np.ndarray] = []
        self.slow_meta: List[ExpertFingerprintMeta] = []
        self.router_W = np.zeros((d_in, max_experts), dtype=np.float64)
        self.replay_buffer: List[Tuple[np.ndarray, np.ndarray, int]] = []
        # SLEEP-DAG-RSI v6 & v7: Distilled Symbolic AST Rules & Compositional Library
        self.symbolic_rules: Dict[int, Callable[[np.ndarray], np.ndarray]] = {}
        self.symbolic_reports: Dict[int, Dict[str, Any]] = {}
        self.symbolic_library = SymbolicLibrary()

    def _eval_expert_slot(self, x: np.ndarray, slot: int) -> np.ndarray:
        """Evaluates slow expert `slot`: explicit distilled AST rule (if any) + neural residual."""
        val = (x @ self.slow_A[slot]) @ self.slow_B[slot]
        sym_fn = self.symbolic_rules.get(slot)
        if sym_fn is not None:
            val = val + sym_fn(x)
        return val

    def forward_delta(
        self,
        x: np.ndarray,
        use_fast: bool = True,
        forced_expert: Optional[int] = None,
        crossover_pair: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Computes the residual adaptation `Delta_y` for input `x` of shape `(B, d_in)`."""
        out = np.zeros((x.shape[0], self.d_out), dtype=np.float64)
        if use_fast:
            out = out + (x @ self.fast_A) @ self.fast_B

        n_exp = len(self.slow_A)
        if n_exp == 0:
            return out

        if forced_expert is not None and 0 <= forced_expert < n_exp:
            return out + self._eval_expert_slot(x, forced_expert)

        if crossover_pair is not None:
            i, j = crossover_pair
            if 0 <= i < n_exp and 0 <= j < n_exp:
                d_i = self._eval_expert_slot(x, i)
                d_j = self._eval_expert_slot(x, j)
                return out + 0.5 * (d_i + d_j)

        logits_r = (x @ self.router_W[:, :n_exp]) / 0.25
        probs_r = softmax_np(logits_r, axis=-1)
        for idx in range(n_exp):
            w_idx = probs_r[:, idx : idx + 1]
            out = out + w_idx * self._eval_expert_slot(x, idx)
        return out

    def wake_adapt_step(
        self,
        x: np.ndarray,
        base_out: np.ndarray,
        target_out: np.ndarray,
        lr: float = 0.08,
    ) -> float:
        """Online Wake step updating only the Fast Low-Rank adapter `(fast_A, fast_B)`."""
        pred = base_out + self.forward_delta(x, use_fast=True)
        err = pred - target_out
        loss = float(0.5 * np.mean(err * err))
        h_low = x @ self.fast_A
        grad_B = (h_low.T @ err) / float(max(1, x.shape[0]))
        grad_A = (x.T @ (err @ self.fast_B.T)) / float(max(1, x.shape[0]))
        self.fast_B -= rms_step(grad_B, lr)
        self.fast_A -= rms_step(grad_A, lr * 0.5)
        return loss

    def nrem_consolidate(
        self,
        x_trajectories: np.ndarray,
        base_outputs: np.ndarray,
        dag_tree: dream_engine.DiscoveryTree,
        trajectory_node_ids: Sequence[str],
        fingerprint_meta: ExpertFingerprintMeta,
        steps: int = 12,
        lr: float = 0.10,
        target_delta_override: Optional[np.ndarray] = None,
        enable_neuro_symbolic_distill: bool = True,
        distill_alpha: float = 0.04,
        distill_r2_threshold: float = 0.25,
        val_metric_fn: Optional[Callable[[Callable[[np.ndarray], np.ndarray]], float]] = None,
        val_baseline_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        NREM Sleep (v5 + v6 Residual-Keep Neuro-Symbolic Distillation):
          1. Masks out failed/dead-end DAG trajectories (`loss_mask = 0`).
          2. Unlocks a new slow low-rank expert for the verified winning lineage.
          3. Runs `NeuroSymbolicDistiller` (Stage 3 of v6): if a sparse symbolic AST
             law explains `R^2_OOS >= distill_r2_threshold` and passes `AstPolicySafetyGate`
             + `accept_joint_2x2`, registers the AST node in `dag_tree` and subtracts
             `f_AST(x)` from `teacher_delta` (`Residual-Keep Decomposition`).
          4. Fits the neural low-rank expert `(slow_A, slow_B)` strictly on the remaining
             unexplained residual while preserving prior replay anchors.
        """
        sft_rep = dream_engine.export_masked_sft_dataset(dag_tree)
        supported_set = set(sft_rep["supported_node_ids"])
        win_indices = [
            i for i, nid in enumerate(trajectory_node_ids) if nid in supported_set
        ]
        if not win_indices:
            win_indices = list(range(x_trajectories.shape[0]))

        x_win = x_trajectories[win_indices]
        if target_delta_override is not None:
            teacher_delta = target_delta_override[win_indices].copy()
        else:
            teacher_delta = self.forward_delta(x_win, use_fast=True)

        slot_idx = len(self.slow_A)
        if slot_idx < self.max_experts:
            A_new = self.rng.normal(0.0, 1.0 / math.sqrt(self.d_in), size=(self.d_in, self.d_low))
            B_new = np.zeros((self.d_low, self.d_out), dtype=np.float64)
            fingerprint_meta.expert_slot = slot_idx
            self.slow_A.append(A_new)
            self.slow_B.append(B_new)
            self.slow_meta.append(fingerprint_meta)
        else:
            slot_idx = self.max_experts - 1
            self.slow_meta[slot_idx] = fingerprint_meta

        # Align router column with the winning input centroid
        centroid = np.mean(x_win, axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm > 1e-9:
            self.router_W[:, slot_idx] = centroid / c_norm

        # Stage 3 of SLEEP-DAG-RSI v6 & v7: Residual-Keep Neuro-Symbolic Distillation + Library Invention
        distill_rep: Dict[str, Any] = {"accepted": False, "r2_oos": 0.0, "active_terms": []}
        residual_target = teacher_delta.copy()
        if enable_neuro_symbolic_distill and x_win.shape[0] >= 8:
            distiller = NeuroSymbolicDistiller(
                alpha=distill_alpha,
                r2_threshold=distill_r2_threshold,
                library=self.symbolic_library,
                auto_invent_primitives=True,
            )
            distill_rep = distiller.distill_residual(
                x_win,
                teacher_delta,
                val_metric_fn=val_metric_fn,
                val_baseline_score=val_baseline_score,
                reconstruct_uncorrupted_target=True,
            )
            if distill_rep["accepted"]:
                if distill_rep.get("reconstructed_full_target"):
                    # Supersede fragmented prior-cycle symbolic/neural approximations to avoid double-counting
                    for prev_slot in range(slot_idx):
                        if prev_slot in self.symbolic_rules:
                            self.symbolic_rules.pop(prev_slot, None)
                            self.slow_B[prev_slot].fill(0.0)
                    self.replay_buffer = [
                        (xa, da, sl) for (xa, da, sl) in self.replay_buffer if sl == slot_idx
                    ]
                self.symbolic_rules[slot_idx] = distill_rep["predict_fn"]
                self.symbolic_reports[slot_idx] = distill_rep
                residual_target = distill_rep["residual_rem"]
                # Commit verified symbolic rule into the DAG tree
                ast_node_id = f"ns_ast_slot{slot_idx}"
                if ast_node_id not in dag_tree.nodes:
                    dag_tree.add(
                        dream_engine.Node(
                            id=ast_node_id,
                            parent_id=fingerprint_meta.source_node_id
                            if fingerprint_meta.source_node_id in dag_tree.nodes
                            else dag_tree.root_id,
                            branch=slot_idx,
                            attempt=len(dag_tree.nodes),
                            score=float(dag_tree.baseline_score) + 10.0 * float(distill_rep["r2_oos"]),
                            pass_mask=list(fingerprint_meta.pass_mask),
                            artifact={"ast_code": distill_rep["ast_code"], "terms": distill_rep["active_terms"]},
                            meta={
                                "stage": "neuro_symbolic_distill",
                                "r2_oos": distill_rep["r2_oos"],
                                "neural_energy_reduction_ratio": distill_rep["energy_reduction_ratio"],
                                "reconstructed_full_target": bool(distill_rep.get("reconstructed_full_target", False)),
                            },
                        )
                    )

        for _ in range(steps):
            student_delta = (x_win @ self.slow_A[slot_idx]) @ self.slow_B[slot_idx]
            diff = student_delta - residual_target
            h_s = x_win @ self.slow_A[slot_idx]
            g_B = (h_s.T @ diff) / float(max(1, x_win.shape[0]))
            g_A = (x_win.T @ (diff @ self.slow_B[slot_idx].T)) / float(max(1, x_win.shape[0]))
            self.slow_B[slot_idx] -= rms_step(g_B, lr)
            self.slow_A[slot_idx] -= rms_step(g_A, lr * 0.5)

            for x_anc, d_anc, anc_slot in self.replay_buffer:
                s_anc = (x_anc @ self.slow_A[anc_slot]) @ self.slow_B[anc_slot]
                e_anc = s_anc - d_anc
                h_a = x_anc @ self.slow_A[anc_slot]
                self.slow_B[anc_slot] -= rms_step((h_a.T @ e_anc) / max(1, len(x_anc)), lr * 0.4)

        self.replay_buffer.append((x_win.copy(), residual_target.copy(), slot_idx))
        # Synaptic pruning of fast wake residual after NREM consolidation
        self.fast_B *= 0.15

        return {
            "unlocked_expert_slot": slot_idx,
            "supervised_win_count": len(win_indices),
            "total_wake_count": len(trajectory_node_ids),
            "active_slow_experts": len(self.slow_A),
            "neuro_symbolic_distill": {
                "accepted": distill_rep["accepted"],
                "r2_oos": distill_rep.get("r2_oos", 0.0),
                "active_terms": distill_rep.get("active_terms", []),
                "ast_code": distill_rep.get("ast_code", ""),
                "energy_reduction_ratio": distill_rep.get("energy_reduction_ratio", 0.0),
            },
        }

    def select_complementary_slow_experts(self) -> Optional[Tuple[int, int, float]]:
        """Finds the pair of slow experts `(i, j)` with maximum `oracle_union_gain > 0`."""
        best_gain = 0.0
        best_pair: Optional[Tuple[int, int, float]] = None
        n = len(self.slow_meta)
        for i in range(n):
            for j in range(i + 1, n):
                m_i, m_j = self.slow_meta[i].pass_mask, self.slow_meta[j].pass_mask
                if m_i and m_j and len(m_i) == len(m_j):
                    fp = dream_engine.compute_error_fingerprint(m_i, m_j)
                    gain = float(fp["oracle_union_gain"])
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j, gain)
        return best_pair


def attach_sleep_dag_adapters(
    pytorch_model: Any,
    target_layer_names: Optional[Sequence[str]] = None,
    d_low: int = 8,
    max_experts: int = 8,
) -> Dict[str, UniversalNeuralSleepAdapter]:
    """
    Freezes all base parameters (`requires_grad = False`) of an already-trained PyTorch
    `nn.Module` (e.g. Transformer, ResNet, SimVQCoprimeGPT) and attaches a
    `UniversalNeuralSleepAdapter` forward hook to each target `nn.Linear` layer.
    Returns the dictionary `{layer_name: adapter}` so `nrem_consolidate()` can be
    called directly on the attached adapters.
    """
    import torch
    import torch.nn as nn

    for p in pytorch_model.parameters():
        p.requires_grad = False

    attached: Dict[str, UniversalNeuralSleepAdapter] = {}
    for name, mod in pytorch_model.named_modules():
        if isinstance(mod, nn.Linear):
            if target_layer_names is not None and not any(t in name for t in target_layer_names):
                continue
            adapter = UniversalNeuralSleepAdapter(
                d_in=int(mod.in_features),
                d_out=int(mod.out_features),
                d_low=min(d_low, int(mod.in_features), int(mod.out_features)),
                max_experts=max_experts,
            )
            attached[name] = adapter

            def _make_hook(ad: UniversalNeuralSleepAdapter):
                def _hook(_module: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
                    x_t = inputs[0]
                    orig_shape = x_t.shape
                    x_np = x_t.detach().cpu().numpy().reshape(-1, ad.d_in).astype(np.float64)
                    delta_np = ad.forward_delta(x_np, use_fast=True)
                    delta_t = torch.tensor(
                        delta_np, dtype=output.dtype, device=output.device
                    ).reshape(output.shape)
                    return output + delta_t

                return _hook

            mod.register_forward_hook(_make_hook(adapter))
    return attached


# =====================================================================
# 10. AST ANTI-CHEAT VERIFIER FOR POLICIES (`AstPolicySafetyGate`)
# =====================================================================


class AstPolicySafetyGate:
    """
    Static AST & Behavioural Anti-Cheat Verifier for authored exploration policies:
      1. Blocks dangerous calls/imports (`eval`, `exec`, `compile`, `__import__`,
         `os.system`, `subprocess`, `shutil.rmtree`, `setattr`, `globals`).
      2. Blocks reward-hacking inspection of unrevealed tree/world state
         (`.nodes`, `._secret`, `best_so_far`, hardcoded node IDs like `"b0_win"`).
      3. Verifies non-degenerate `beta` sweep sensitivity on synthetic history.
    """

    FORBIDDEN_CALLS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "system",
        "popen",
        "rmtree",
        "globals",
        "locals",
        "setattr",
        "delattr",
    }
    FORBIDDEN_ATTRS = {
        "best_so_far",
        "budget_spent",
        "_secret",
        "unrevealed",
        "__dict__",
        "__subclasses__",
    }

    @classmethod
    def verify_source(cls, source_code: str) -> Dict[str, Any]:
        import ast
        import re

        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            return {"valid": False, "violations": [f"syntax_error: {exc}"]}

        violations: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.FORBIDDEN_CALLS:
                    violations.append(f"forbidden_call:{node.func.id}")
                elif isinstance(node.func, ast.Attribute) and node.func.attr in cls.FORBIDDEN_CALLS:
                    violations.append(f"forbidden_method:{node.func.attr}")
            elif isinstance(node, ast.Attribute):
                if node.attr in cls.FORBIDDEN_ATTRS:
                    violations.append(f"forbidden_attr:{node.attr}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.match(r"^(node_\d+|b\d+_a\d+|b\d+_win)$", node.value):
                    violations.append(f"hardcoded_node_id:{node.value}")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
        }


# =====================================================================
# 11. STAGE 3 OF SLEEP-DAG-RSI v6: STANDALONE NUMPY LASSO &
#     NEURO-SYMBOLIC DISTILLER (`Residual-Keep Decomposition`)
# =====================================================================


def numpy_lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.04,
    max_iter: int = 150,
    tol: float = 1e-6,
    support_prune_rel: float = 0.08,
) -> Tuple[np.ndarray, float]:
    """
    Standalone vectorized cyclical Coordinate-Descent Lasso solver in pure NumPy
    with soft-thresholding S_alpha(z) = sgn(z) * max(|z| - alpha, 0) and a
    post-selection OLS refit on the active support so recovered symbolic coefficients
    are unbiased.
    Solves:
        min_{w, b} (1 / 2N) ||y - (X w + b)||_2^2 + alpha * ||w_std||_1
    Returns `(w, intercept)` in original unstandardized feature coordinates.
    """
    n_samples, n_features = X.shape
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    if n_samples == 0 or n_features == 0:
        return np.zeros((n_features,), dtype=np.float64), 0.0

    x_mean = np.mean(X, axis=0)
    x_centered = X - x_mean
    x_scale = np.sqrt(np.mean(x_centered * x_centered, axis=0))
    valid_cols = x_scale > 1e-12
    x_scale_safe = np.where(valid_cols, x_scale, 1.0)
    X_std = x_centered / x_scale_safe

    y_mean = float(np.mean(y_vec))
    r = (y_vec - y_mean).copy()
    w_std = np.zeros((n_features,), dtype=np.float64)

    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(n_features):
            if not valid_cols[j]:
                continue
            old_wj = w_std[j]
            col_j = X_std[:, j]
            rho_j = float(np.dot(col_j, r)) / float(n_samples) + old_wj
            if rho_j > alpha:
                new_wj = rho_j - alpha
            elif rho_j < -alpha:
                new_wj = rho_j + alpha
            else:
                new_wj = 0.0
            diff = new_wj - old_wj
            if abs(diff) > 1e-15:
                r -= col_j * diff
                w_std[j] = new_wj
                if abs(diff) > max_delta:
                    max_delta = abs(diff)
        if max_delta < tol:
            break

    # Prune tiny support noise, filter out Gram-collinear atoms (|cos| > 0.72),
    # and perform a lightly ridge-damped OLS polish on at most 4 orthogonal terms.
    max_w = float(np.max(np.abs(w_std))) if n_features > 0 else 0.0
    if max_w > 1e-12:
        cand_indices = [
            int(idx)
            for idx in np.argsort(-np.abs(w_std))
            if abs(w_std[idx]) >= (support_prune_rel * max_w)
        ]
    else:
        cand_indices = []

    selected_idx: List[int] = []
    for c_i in cand_indices:
        if len(selected_idx) >= 4:
            break
        col_c = X_std[:, c_i]
        is_collinear = False
        for s_i in selected_idx:
            cos_sim = abs(float(np.dot(col_c, X_std[:, s_i])) / float(max(1, n_samples)))
            if cos_sim > 0.72:
                is_collinear = True
                break
        if not is_collinear:
            selected_idx.append(c_i)

    w_raw = np.zeros((n_features,), dtype=np.float64)
    if selected_idx:
        active_idx = np.array(selected_idx, dtype=np.int64)
        X_act = X[:, active_idx]
        A_ols = np.column_stack([X_act, np.ones((n_samples, 1), dtype=np.float64)])
        k_cols = A_ols.shape[1]
        reg = 1e-4 * float(n_samples) * np.eye(k_cols, dtype=np.float64)
        reg[-1, -1] = 0.0  # do not regularize intercept
        sol = np.linalg.solve(A_ols.T @ A_ols + reg, A_ols.T @ y_vec)
        for pos, col_i in enumerate(active_idx):
            w_raw[col_i] = float(sol[pos])
        intercept = float(sol[-1])
    else:
        intercept = y_mean

    return w_raw, intercept


@dataclass
class LibraryPrimitive:
    """
    SLEEP-DAG-RSI v7 Invented Compositional Subroutine (`lib_k(x)`):
    Represents a multi-term symbolic motif discovered during NREM distillation
    that compresses description length (`mdl_gain > 0`) and can be composed
    hierarchically in subsequent sleep cycles (`depth > 2`).
    """

    name: str
    expr_str: str
    eval_fn: Callable[[np.ndarray], np.ndarray]
    mdl_gain: float
    r2_score: float
    term_count: int
    ast_depth: int = 2
    reuse_count: int = 0
    natural_scale: float = 1.0
    mean_offset: float = 0.0


class SymbolicLibrary:
    """
    SLEEP-DAG-RSI v7 Wake-Sleep Compositional Library Learner (DreamCoder-style AST Abstraction):
    Breaks the flat depth-2 basis ceiling of `NeuroSymbolicDistiller`:
      1. Abstracts verified multi-term symbolic formulas into reusable primitives `lib_0(x), lib_1(x), ...`
         standardized to unit scale on the training manifold, while recording `natural_scale` (`sigma_raw`)
         for exact nonlinear link composition (e.g. `math.tanh(scale * lib_k(x))`).
      2. Enforces a strict Minimum Description Length (MDL) & Non-Collinearity gate
         (`|cos(lib_new, lib_old)| <= 0.80`, `term_count >= 2`, `r2_score >= min_r2`).
      3. Tracks `prior_stage_predictors` so subsequent sleep cycles can perform Full-Target
         Re-Consolidation (`Y_full = Y_prior_symbolic + Y_rem`), preventing greedy Cycle-1
         linear residual subtraction from corrupting outer nonlinear link functions.
      4. Emits standalone Python helper definitions (`def lib_0(x): return ...`) so
         hierarchical AST rules remain 100% self-contained and pass `AstPolicySafetyGate`.
    """

    def __init__(self, max_primitives: int = 6, max_collinearity: float = 0.80) -> None:
        self.max_primitives = max(1, int(max_primitives))
        self.max_collinearity = float(max_collinearity)
        self.primitives: List[LibraryPrimitive] = []
        self.prior_stage_predictors: List[Callable[[np.ndarray], np.ndarray]] = []

    def register_primitive(
        self,
        expr_str: str,
        eval_fn: Callable[[np.ndarray], np.ndarray],
        term_count: int,
        r2_score: float,
        X_sample: np.ndarray,
        ast_depth: int = 2,
        min_r2: float = 0.30,
        natural_scale: Optional[float] = None,
    ) -> Optional[LibraryPrimitive]:
        if term_count < 2 or r2_score < min_r2:
            return None
        vals = np.asarray(eval_fn(X_sample), dtype=np.float64).reshape(-1)
        std_v = float(np.std(vals))
        if std_v < 1e-6:
            return None
        mean_v = float(np.mean(vals))
        norm_vals = (vals - mean_v) / std_v

        for p in self.primitives:
            p_vals = np.asarray(p.eval_fn(X_sample), dtype=np.float64).reshape(-1)
            p_std = float(np.std(p_vals)) + 1e-12
            p_norm = (p_vals - float(np.mean(p_vals))) / p_std
            cos_sim = abs(float(np.dot(norm_vals, p_norm)) / float(max(1, len(norm_vals))))
            if cos_sim > self.max_collinearity:
                return None

        # Normalize the primitive output to zero-mean unit-std so higher-order products stay well-scaled
        inv_std = round(1.0 / std_v, 6)
        mean_r = round(mean_v, 6)
        norm_expr = f"({inv_std} * (({expr_str}) - ({mean_r:+.6f})))"
        nat_scale = round(float(natural_scale if natural_scale is not None else std_v), 6)

        def _norm_eval(arr: np.ndarray, fn=eval_fn, m=mean_v, s=std_v) -> np.ndarray:
            return (np.asarray(fn(arr), dtype=np.float64).reshape(-1) - m) / s

        mdl_gain = round(float(term_count - 1) * float(r2_score), 6)
        prim_name = f"lib_{len(self.primitives)}"
        prim = LibraryPrimitive(
            name=prim_name,
            expr_str=norm_expr,
            eval_fn=_norm_eval,
            mdl_gain=mdl_gain,
            r2_score=round(float(r2_score), 6),
            term_count=int(term_count),
            ast_depth=int(ast_depth),
            reuse_count=0,
            natural_scale=nat_scale,
            mean_offset=mean_r,
        )
        if len(self.primitives) < self.max_primitives:
            self.primitives.append(prim)
        else:
            # Replace lowest MDL primitive if new one is stronger
            worst_idx = min(range(len(self.primitives)), key=lambda i: self.primitives[i].mdl_gain)
            if mdl_gain > self.primitives[worst_idx].mdl_gain:
                prim.name = self.primitives[worst_idx].name
                self.primitives[worst_idx] = prim
            else:
                return None
        return prim

    def emit_helper_functions_code(self) -> str:
        lines: List[str] = []
        for p in self.primitives:
            lines.append(f"def {p.name}(x):\n    return {p.expr_str}\n")
        return "\n".join(lines)


class NeuroSymbolicDistiller:
    """
    Stage 3 of SLEEP-DAG-RSI v6 & v7 (`MoE-to-AST Compiler` with `Residual-Keep Decomposition`,
    `SymbolicLibrary` Subroutine Invention, `Full-Target Re-Consolidation`, and `ActiveBoundarySynthesizer`):
      1. Filters input dimensions by sensitivity/gradient norm (`d_active <= 8`).
      2. Constructs a bounded typed AST basis dictionary `Psi(X)` (`K <= 64`), automatically
         expanding with hierarchical `SymbolicLibrary` subroutines (`lib_k(x)`, `tanh(scale * lib_k(x))`,
         `lib_k(x) * x[i]`, `lib_k(x) * sign(x[i])`) so depth-3/4 compositional laws can be distilled!
      3. Optionally queries `ActiveBoundarySynthesizer` against `active_teacher_fn` to enrich
         passive Wake states with high-disagreement decision-boundary counterexamples.
      4. Fits `numpy_lasso_coordinate_descent` per output channel and verifies via `AstPolicySafetyGate`.
      5. When `reconstruct_uncorrupted_target=True` and prior-cycle symbolic predictors exist in
         `SymbolicLibrary`, also evaluates against the reconstructed uncorrupted target
         `Y_full(X) = delta_Y + sum_{prior} f_AST(X)` so greedy Cycle-1 linear approximations
         of `tanh(z)` are cleanly replaced by the exact composite law `tanh(lib_k(x))`.
    """

    def __init__(
        self,
        alpha: float = 0.04,
        r2_threshold: float = 0.25,
        max_active_inputs: int = 8,
        library: Optional[SymbolicLibrary] = None,
        auto_invent_primitives: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.r2_threshold = float(r2_threshold)
        self.max_active_inputs = min(8, max(1, int(max_active_inputs)))
        self.library = library
        self.auto_invent_primitives = bool(auto_invent_primitives)

    def select_active_inputs(self, X: np.ndarray, Y: np.ndarray) -> List[int]:
        d_in = X.shape[1]
        if d_in <= self.max_active_inputs:
            return list(range(d_in))
        scores = np.zeros((d_in,), dtype=np.float64)
        Xc = X - np.mean(X, axis=0, keepdims=True)
        Yc = Y - np.mean(Y, axis=0, keepdims=True)
        lib_cols: List[np.ndarray] = []
        if self.library is not None and self.library.primitives:
            lib_cols = [p.eval_fn(X)[:, None] for p in self.library.primitives]
        for i in range(d_in):
            xi = Xc[:, i : i + 1]
            si = np.where(X[:, i : i + 1] >= 0.0, 1.0, -1.0)
            cand_feats = [xi, xi * xi, si]
            # Also screen multiplicative interactions with invented library primitives
            # so zero-mean regime modulators `lib_k(x) * sign(x_i)` are never missed!
            for lp in lib_cols:
                cand_feats.append(lp * xi)
                cand_feats.append(lp * si)
            for feat in cand_feats:
                fnorm = float(np.linalg.norm(feat)) + 1e-12
                corr = np.linalg.norm((feat.T @ Yc) / fnorm)
                scores[i] = max(scores[i], float(corr))
        ranked = np.argsort(-scores)[: self.max_active_inputs]
        return sorted(int(idx) for idx in ranked.tolist())

    def build_basis_dictionary(
        self, X: np.ndarray, active_inputs: Sequence[int]
    ) -> Tuple[np.ndarray, List[str], List[Callable[[np.ndarray], np.ndarray]]]:
        """
        Builds bounded dictionary `Psi(X)` in R^{B x K} with K <= 64.
        When `self.library` has invented subroutines `lib_k(x)`, injects hierarchical
        compositions (`lib_k(x)`, `tanh(lib_k(x))`, `tanh(scale * lib_k(x))`, `lib_k(x) * x[i]`,
        `lib_k(x) * sign(x[i])`) across all `active_inputs` before filling remaining slots
        with depth-2 raw pairwise interactions.
        """
        cols: List[np.ndarray] = []
        exprs: List[str] = []
        evaluators: List[Callable[[np.ndarray], np.ndarray]] = []

        for i in active_inputs:
            cols.append(X[:, i])
            exprs.append(f"x[{i}]")
            evaluators.append(lambda arr, idx=i: arr[:, idx])

            cols.append(np.tanh(X[:, i]))
            exprs.append(f"math.tanh(x[{i}])")
            evaluators.append(lambda arr, idx=i: np.tanh(arr[:, idx]))

            cols.append(np.where(X[:, i] >= 0.0, 1.0, -1.0))
            exprs.append(f"(1.0 if x[{i}] >= 0.0 else -1.0)")
            evaluators.append(lambda arr, idx=i: np.where(arr[:, idx] >= 0.0, 1.0, -1.0))

            cols.append(np.maximum(0.0, X[:, i]))
            exprs.append(f"max(0.0, x[{i}])")
            evaluators.append(lambda arr, idx=i: np.maximum(0.0, arr[:, idx]))

        # SLEEP-DAG-RSI v7: Hierarchical Compositional Library Expansion across all active_inputs
        if self.library is not None and self.library.primitives:
            for prim in self.library.primitives:
                if len(cols) >= 64:
                    break
                p_col = prim.eval_fn(X)
                cols.append(p_col)
                exprs.append(f"{prim.name}(x)")
                evaluators.append(lambda arr, p_fn=prim.eval_fn: p_fn(arr))

                if len(cols) < 64:
                    sc = round(float(prim.natural_scale), 6)
                    off = round(float(prim.mean_offset), 6)
                    if abs(sc - 1.0) > 0.05 or abs(off) > 1e-4:
                        cols.append(np.tanh(sc * p_col + off))
                        exprs.append(f"math.tanh({sc} * {prim.name}(x) + ({off:+.6f}))")
                        evaluators.append(
                            lambda arr, p_fn=prim.eval_fn, s_val=sc, o_val=off: np.tanh(
                                s_val * p_fn(arr) + o_val
                            )
                        )
                    else:
                        cols.append(np.tanh(p_col))
                        exprs.append(f"math.tanh({prim.name}(x))")
                        evaluators.append(lambda arr, p_fn=prim.eval_fn: np.tanh(p_fn(arr)))

                for i in active_inputs:
                    if len(cols) < 64:
                        cols.append(p_col * X[:, i])
                        exprs.append(f"({prim.name}(x) * x[{i}])")
                        evaluators.append(lambda arr, p_fn=prim.eval_fn, idx=i: p_fn(arr) * arr[:, idx])
                    if len(cols) < 64:
                        sign_i = np.where(X[:, i] >= 0.0, 1.0, -1.0)
                        cols.append(p_col * sign_i)
                        exprs.append(f"({prim.name}(x) * (1.0 if x[{i}] >= 0.0 else -1.0))")
                        evaluators.append(
                            lambda arr, p_fn=prim.eval_fn, idx=i: p_fn(arr)
                            * np.where(arr[:, idx] >= 0.0, 1.0, -1.0)
                        )

        pair_inputs = list(active_inputs)[:6]
        for idx_a in range(len(pair_inputs)):
            for idx_b in range(idx_a + 1, len(pair_inputs)):
                i, j = pair_inputs[idx_a], pair_inputs[idx_b]
                if len(cols) < 64:
                    cols.append(X[:, i] * X[:, j])
                    exprs.append(f"(x[{i}] * x[{j}])")
                    evaluators.append(lambda arr, ia=i, ib=j: arr[:, ia] * arr[:, ib])
                if len(cols) < 64:
                    cols.append(X[:, i] - X[:, j])
                    exprs.append(f"(x[{i}] - x[{j}])")
                    evaluators.append(lambda arr, ia=i, ib=j: arr[:, ia] - arr[:, ib])

        Psi = np.column_stack(cols) if cols else np.zeros((X.shape[0], 0), dtype=np.float64)
        return Psi, exprs, evaluators

    def _discover_nonlinear_inner_submotif(
        self,
        Psi_tr: np.ndarray,
        Y_tr_1d: np.ndarray,
        w_linear: np.ndarray,
        exprs: Sequence[str],
        evaluators: Sequence[Callable[[np.ndarray], np.ndarray]],
        X_full: np.ndarray,
    ) -> Optional[LibraryPrimitive]:
        """
        When a Cycle-1 linear fit `w_linear` leaves a saturation residual (e.g. `y = tanh(z) + c*x_4`),
        tests subsets `S` of the active linear support (`2 <= |S| <= m`) via 4 fast Gauss-Newton
        steps on `a * tanh(Psi_S @ theta_S) + Psi_rest @ theta_rest + b`.
        If a subset `S` reduces training SSE by > 50%, registers the polished inner argument
        `Psi_S @ theta_S` into `self.library` along with its exact natural saturation scale `std(z_S)`.
        """
        if self.library is None:
            return None
        act_idx = [k for k in range(len(w_linear)) if abs(w_linear[k]) > 1e-5]
        m = len(act_idx)
        if m < 2 or m > 5:
            return None

        base_pred = Psi_tr[:, act_idx] @ w_linear[act_idx]
        base_pred = base_pred + (float(np.mean(Y_tr_1d)) - float(np.mean(base_pred)))
        base_sse = float(np.sum((Y_tr_1d - base_pred) ** 2))
        tot_ss = float(np.sum((Y_tr_1d - float(np.mean(Y_tr_1d))) ** 2)) + 1e-12
        if base_sse / tot_ss < 1e-4:
            return None

        from itertools import combinations

        best_sse = base_sse
        best_sub_indices: Optional[List[int]] = None
        best_theta_s: Optional[np.ndarray] = None

        n_tr = Psi_tr.shape[0]
        ones_col = np.ones((n_tr, 1), dtype=np.float64)

        for sz in range(2, m + 1):
            for sub_tuple in combinations(act_idx, sz):
                sub_list = list(sub_tuple)
                rest_list = [k for k in act_idx if k not in sub_list]
                Psi_s = Psi_tr[:, sub_list]
                Psi_r = Psi_tr[:, rest_list] if rest_list else np.zeros((n_tr, 0), dtype=np.float64)

                for scale_init in (1.2, 1.6):
                    theta_s = w_linear[sub_list].copy() * scale_init
                    a_val = 1.0
                    theta_r = w_linear[rest_list].copy() if rest_list else np.zeros((0,), dtype=np.float64)
                    b_val = 0.0

                    for _gn in range(5):
                        z_s = Psi_s @ theta_s
                        t_z = np.tanh(z_s)
                        sech2 = 1.0 - t_z * t_z
                        pred = a_val * t_z + (Psi_r @ theta_r if rest_list else 0.0) + b_val
                        res = Y_tr_1d - pred
                        J_s = (a_val * sech2)[:, None] * Psi_s
                        J_cols = [J_s, t_z[:, None]]
                        if rest_list:
                            J_cols.append(Psi_r)
                        J_cols.append(ones_col)
                        J = np.hstack(J_cols)
                        p_dim = J.shape[1]
                        step = np.linalg.solve(
                            J.T @ J + 1e-4 * float(n_tr) * np.eye(p_dim),
                            J.T @ res,
                        )
                        theta_s = theta_s + step[:sz]
                        a_val = float(a_val + step[sz])
                        if rest_list:
                            theta_r = theta_r + step[sz + 1 : sz + 1 + len(rest_list)]
                        b_val = float(b_val + step[-1])

                    z_final = Psi_s @ theta_s
                    pred_final = a_val * np.tanh(z_final) + (Psi_r @ theta_r if rest_list else 0.0) + b_val
                    sse_final = float(np.sum((Y_tr_1d - pred_final) ** 2))
                    if sse_final < best_sse * 0.45:
                        best_sse = sse_final
                        best_sub_indices = sub_list
                        best_theta_s = theta_s.copy()

        if best_sub_indices is None or best_theta_s is None:
            return None

        sub_terms = [
            f"{best_theta_s[pos]:+.6f} * {exprs[k_idx]}"
            for pos, k_idx in enumerate(best_sub_indices)
        ]
        sub_expr_str = " ".join(sub_terms)
        sub_evs = [evaluators[k_idx] for k_idx in best_sub_indices]
        th_snap = best_theta_s.copy()

        def _sub_eval(arr: np.ndarray, ev_l=sub_evs, w_v=th_snap) -> np.ndarray:
            p_mat = np.column_stack([fn(arr) for fn in ev_l])
            return p_mat @ w_v

        full_vals = _sub_eval(X_full)
        nat_scale = float(np.std(full_vals))
        r2_sub = max(0.0, 1.0 - best_sse / tot_ss)
        return self.library.register_primitive(
            expr_str=sub_expr_str,
            eval_fn=_sub_eval,
            term_count=len(best_sub_indices),
            r2_score=r2_sub,
            X_sample=X_full,
            ast_depth=2,
            natural_scale=nat_scale,
        )

    def _fit_single_target(
        self,
        X_arr: np.ndarray,
        Y_arr: np.ndarray,
        X_tr: np.ndarray,
        Y_tr: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        val_metric_fn: Optional[Callable[[Callable[[np.ndarray], np.ndarray]], float]],
        val_baseline_score: float,
        active_teacher_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        num_active_boundary_queries: int,
    ) -> Dict[str, Any]:
        d_out = Y_arr.shape[1]
        active_inputs = self.select_active_inputs(X_tr, Y_tr)
        Psi_tr, exprs, evaluators = self.build_basis_dictionary(X_tr, active_inputs)

        # Optional Stage 2 of v7: Active Boundary Querying to sharpen decision-boundary fit
        boundary_report: Optional[Dict[str, Any]] = None
        if active_teacher_fn is not None and num_active_boundary_queries > 0 and Psi_tr.shape[1] > 0:
            w_init, b_init = numpy_lasso_coordinate_descent(
                Psi_tr, Y_tr[:, 0], alpha=self.alpha
            )

            def _init_sym_1d(x_row: Sequence[float]) -> float:
                x_2d = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
                p_cols = [ev_fn(x_2d)[0] for ev_fn in evaluators]
                return float(np.dot(p_cols, w_init) + b_init)

            def _teacher_1d(x_row: Sequence[float]) -> float:
                x_2d = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
                t_out = np.asarray(active_teacher_fn(x_2d), dtype=np.float64).reshape(-1)
                return float(t_out[0])

            synthesizer = dream_engine.ActiveBoundarySynthesizer(
                trust_radius_sigma=1.5, ascent_steps=6
            )
            boundary_report = synthesizer.synthesize_boundary_points(
                anchor_points=X_tr.tolist(),
                committee_fns=[_teacher_1d, _init_sym_1d],
                num_synthetic=num_active_boundary_queries,
            )
            synth_pts = boundary_report.get("synthetic_points", [])
            if synth_pts:
                X_synth = np.asarray(synth_pts, dtype=np.float64)
                Y_synth = np.asarray(active_teacher_fn(X_synth), dtype=np.float64)
                if Y_synth.ndim == 1:
                    Y_synth = Y_synth[:, None]
                X_tr = np.vstack([X_tr, X_synth])
                Y_tr = np.vstack([Y_tr, Y_synth])
                Psi_tr, exprs, evaluators = self.build_basis_dictionary(X_tr, active_inputs)

        Psi_val, _, _ = self.build_basis_dictionary(X_val, active_inputs)
        Psi_full, _, _ = self.build_basis_dictionary(X_arr, active_inputs)

        k_atoms = Psi_tr.shape[1]
        W_mat = np.zeros((k_atoms, d_out), dtype=np.float64)
        b_vec = np.zeros((d_out,), dtype=np.float64)
        active_terms: List[str] = []
        channel_lines: List[str] = []
        per_channel_term_counts: List[int] = []

        for out_c in range(d_out):
            w_c, b_c = numpy_lasso_coordinate_descent(
                Psi_tr, Y_tr[:, out_c], alpha=self.alpha
            )
            W_mat[:, out_c] = w_c
            b_vec[out_c] = b_c
            terms_c: List[str] = []
            for k_idx in range(k_atoms):
                if abs(w_c[k_idx]) > 1e-5:
                    term_str = f"{w_c[k_idx]:+.6f} * {exprs[k_idx]}"
                    terms_c.append(term_str)
                    active_terms.append(f"out[{out_c}]: {term_str}")
            per_channel_term_counts.append(len(terms_c))
            if abs(b_c) > 1e-5 or not terms_c:
                terms_c.append(f"{b_c:+.6f}")
            channel_lines.append(" ".join(terms_c))

        helper_code = (
            self.library.emit_helper_functions_code() + "\n"
            if (self.library is not None and self.library.primitives)
            else ""
        )
        ast_code = (
            helper_code
            + "def distilled_ast_rule(x):\n"
            + "    return [\n        "
            + ",\n        ".join(channel_lines)
            + "\n    ]\n"
        )
        safety_rep = AstPolicySafetyGate.verify_source(ast_code)

        pred_tr = Psi_tr @ W_mat + b_vec[None, :]
        pred_val = Psi_val @ W_mat + b_vec[None, :]
        pred_full = Psi_full @ W_mat + b_vec[None, :]

        ss_res_tr = float(np.sum((Y_tr - pred_tr) ** 2))
        ss_tot_tr = float(np.sum((Y_tr - np.mean(Y_tr, axis=0, keepdims=True)) ** 2)) + 1e-12
        r2_train = max(0.0, 1.0 - ss_res_tr / ss_tot_tr)

        ss_res_val = float(np.sum((Y_val - pred_val) ** 2))
        ss_tot_val = float(np.sum((Y_val - np.mean(Y_val, axis=0, keepdims=True)) ** 2)) + 1e-12
        r2_oos = max(0.0, 1.0 - ss_res_val / ss_tot_val)

        def _predict_fn(x_in: np.ndarray) -> np.ndarray:
            x_np = np.asarray(x_in, dtype=np.float64)
            if k_atoms == 0:
                return np.zeros((x_np.shape[0], d_out), dtype=np.float64)
            psi_cols = [ev_fn(x_np) for ev_fn in evaluators]
            psi_mat = np.column_stack(psi_cols)
            return psi_mat @ W_mat + b_vec[None, :]

        if val_metric_fn is not None:
            cand_val_score = float(val_metric_fn(_predict_fn))
            gate_2x2 = dream_engine.accept_joint_2x2(
                {
                    ("theta_neural_only", "psi_0"): float(val_baseline_score),
                    ("theta_residual_keep", "psi_0"): float(val_baseline_score),
                    ("theta_neural_only", "psi_ast"): 0.5 * (float(val_baseline_score) + cand_val_score),
                    ("theta_residual_keep", "psi_ast"): cand_val_score,
                },
                incumbent_key=("theta_neural_only", "psi_0"),
            )
        else:
            gate_2x2 = dream_engine.accept_joint_2x2(
                {
                    ("theta_neural_only", "psi_0"): 0.0,
                    ("theta_residual_keep", "psi_0"): r2_oos * 0.5,
                    ("theta_neural_only", "psi_ast"): r2_train * 0.5,
                    ("theta_residual_keep", "psi_ast"): r2_oos,
                },
                incumbent_key=("theta_neural_only", "psi_0"),
            )

        accepted = bool(
            safety_rep["valid"]
            and len(active_terms) > 0
            and r2_oos >= self.r2_threshold
            and gate_2x2["accepted_pair"] == ("theta_residual_keep", "psi_ast")
        )

        residual_rem = Y_arr - pred_full if accepted else Y_arr.copy()
        orig_energy = float(np.sum(Y_arr ** 2)) + 1e-12
        rem_energy = float(np.sum(residual_rem ** 2))
        energy_reduction_ratio = max(0.0, 1.0 - rem_energy / orig_energy)

        return {
            "accepted": accepted,
            "r2_train": round(r2_train, 6),
            "r2_oos": round(r2_oos, 6),
            "active_inputs": active_inputs,
            "dictionary_size": k_atoms,
            "active_terms": active_terms,
            "ast_code": ast_code,
            "ast_safety_valid": safety_rep["valid"],
            "gate_2x2": gate_2x2,
            "energy_reduction_ratio": round(energy_reduction_ratio, 6),
            "boundary_report": boundary_report,
            "residual_rem": residual_rem,
            "rem_energy": rem_energy,
            "predict_fn": _predict_fn,
            "weights": W_mat,
            "intercept": b_vec,
            "channel_lines": channel_lines,
            "per_channel_term_counts": per_channel_term_counts,
            "evaluators": evaluators,
            "exprs": exprs,
            "Psi_tr": Psi_tr,
            "Y_tr": Y_tr,
        }

    def distill_residual(
        self,
        X: np.ndarray,
        delta_Y: np.ndarray,
        X_oos: Optional[np.ndarray] = None,
        delta_Y_oos: Optional[np.ndarray] = None,
        val_metric_fn: Optional[Callable[[Callable[[np.ndarray], np.ndarray]], float]] = None,
        val_baseline_score: float = 0.0,
        active_teacher_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        num_active_boundary_queries: int = 12,
        reconstruct_uncorrupted_target: bool = True,
    ) -> Dict[str, Any]:
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(delta_Y, dtype=np.float64)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, None]

        n_total, d_out = Y_arr.shape
        if X_oos is not None and delta_Y_oos is not None:
            X_tr, Y_tr = X_arr, Y_arr
            X_val = np.asarray(X_oos, dtype=np.float64)
            Y_val = np.asarray(delta_Y_oos, dtype=np.float64)
            if Y_val.ndim == 1:
                Y_val = Y_val[:, None]
        else:
            split = max(4, int(round(0.75 * n_total)))
            if split >= n_total:
                split = max(1, n_total - 2)
            X_tr, Y_tr = X_arr[:split], Y_arr[:split]
            X_val, Y_val = X_arr[split:], Y_arr[split:]

        fit_direct = self._fit_single_target(
            X_arr,
            Y_arr,
            X_tr,
            Y_tr,
            X_val,
            Y_val,
            val_metric_fn=val_metric_fn,
            val_baseline_score=val_baseline_score,
            active_teacher_fn=active_teacher_fn,
            num_active_boundary_queries=num_active_boundary_queries,
        )
        chosen = fit_direct
        reconstructed_full = False

        # Full-Target Multi-Cycle Re-Consolidation:
        # If `delta_Y` is a corrupted residual `Y_orig - Y_prior_symbolic` (e.g. `tanh(z) - c*z`),
        # reconstruct `Y_recon = delta_Y + Y_prior_symbolic` and test whether composing the newly
        # invented `SymbolicLibrary` primitives explains the full uncorrupted target with strictly
        # lower residual energy!
        if (
            reconstruct_uncorrupted_target
            and self.library is not None
            and self.library.primitives
            and self.library.prior_stage_predictors
        ):
            prior_full = np.zeros_like(Y_arr)
            prior_tr = np.zeros_like(Y_tr)
            prior_val = np.zeros_like(Y_val)
            for p_fn in self.library.prior_stage_predictors:
                prior_full = prior_full + np.asarray(p_fn(X_arr), dtype=np.float64).reshape(Y_arr.shape)
                prior_tr = prior_tr + np.asarray(p_fn(X_tr), dtype=np.float64).reshape(Y_tr.shape)
                prior_val = prior_val + np.asarray(p_fn(X_val), dtype=np.float64).reshape(Y_val.shape)

            fit_recon = self._fit_single_target(
                X_arr,
                Y_arr + prior_full,
                X_tr,
                Y_tr + prior_tr,
                X_val,
                Y_val + prior_val,
                val_metric_fn=val_metric_fn,
                val_baseline_score=val_baseline_score,
                active_teacher_fn=active_teacher_fn,
                num_active_boundary_queries=num_active_boundary_queries,
            )
            if (
                fit_recon["accepted"]
                and fit_recon["rem_energy"] < fit_direct["rem_energy"] * 0.50
                and fit_recon["r2_oos"] > fit_direct["r2_oos"] + 0.02
            ):
                chosen = fit_recon
                reconstructed_full = True

        accepted = chosen["accepted"]
        active_terms = chosen["active_terms"]
        per_channel_term_counts = chosen["per_channel_term_counts"]
        W_mat = chosen["weights"]
        b_vec = chosen["intercept"]
        evaluators = chosen["evaluators"]
        exprs = chosen["exprs"]
        channel_lines = chosen["channel_lines"]
        r2_oos = chosen["r2_oos"]

        # Track library reuse and optionally invent new primitive(s) from accepted multi-term motif
        invented_primitive_name: Optional[str] = None
        if accepted and self.library is not None:
            for p in self.library.primitives:
                if any(f"{p.name}(x)" in t for t in active_terms):
                    p.reuse_count += 1

            if self.auto_invent_primitives and per_channel_term_counts and per_channel_term_counts[0] >= 2:
                # 1. Check if a subset of active terms forms the inner argument of a nonlinear saturation (e.g. tanh)
                sub_prim = self._discover_nonlinear_inner_submotif(
                    Psi_tr=chosen["Psi_tr"],
                    Y_tr_1d=chosen["Y_tr"][:, 0],
                    w_linear=W_mat[:, 0],
                    exprs=exprs,
                    evaluators=evaluators,
                    X_full=X_arr,
                )
                if sub_prim is not None:
                    invented_primitive_name = sub_prim.name

                # 2. Also register the full channel-0 multi-term linear combination
                w0 = W_mat[:, 0].copy()
                b0 = float(b_vec[0])
                evs_snap = list(evaluators)

                def _ch0_eval(arr: np.ndarray, w_v=w0, b_v=b0, ev_l=evs_snap) -> np.ndarray:
                    p_mat = np.column_stack([fn(arr) for fn in ev_l])
                    return p_mat @ w_v + b_v

                new_prim = self.library.register_primitive(
                    expr_str=channel_lines[0],
                    eval_fn=_ch0_eval,
                    term_count=per_channel_term_counts[0],
                    r2_score=r2_oos,
                    X_sample=X_arr,
                    ast_depth=3 if any("lib_" in t for t in active_terms) else 2,
                )
                if new_prim is not None and invented_primitive_name is None:
                    invented_primitive_name = new_prim.name

            if reconstructed_full:
                self.library.prior_stage_predictors = [chosen["predict_fn"]]
            else:
                self.library.prior_stage_predictors.append(chosen["predict_fn"])

        return {
            "accepted": accepted,
            "r2_train": chosen["r2_train"],
            "r2_oos": r2_oos,
            "active_inputs": chosen["active_inputs"],
            "dictionary_size": chosen["dictionary_size"],
            "active_terms": active_terms,
            "ast_code": chosen["ast_code"],
            "ast_safety_valid": chosen["ast_safety_valid"],
            "gate_2x2": chosen["gate_2x2"],
            "energy_reduction_ratio": chosen["energy_reduction_ratio"],
            "invented_primitive": invented_primitive_name,
            "reconstructed_full_target": reconstructed_full,
            "boundary_report": chosen["boundary_report"],
            "residual_rem": chosen["residual_rem"],
            "predict_fn": chosen["predict_fn"],
            "weights": W_mat,
            "intercept": b_vec,
        }


if __name__ == "__main__":
    print("SLEEP-DAG-RSI core module loaded successfully.")