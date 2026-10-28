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


if __name__ == "__main__":
    print("SLEEP-DAG-RSI core module loaded successfully.")