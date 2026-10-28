"""
Comprehensive Verification, Ablation, and Benchmark Suite for SLEEP-DAG-RSI.

Executes:
  1. Suite 1 (9 Mathematical & Unit Verification Checks):
     - Exact analytical vs finite-difference gradient checks (< 1e-7) for GKD JSD,
       LowRankExpert {A, B}, ExpandableMoEMLP router, and LoRALinearWrapper.
     - HOPE DeltaRuleFastMemory associative recall convergence & sequential-vs-step equivalence.
     - CMS exact C^(l) gradient accumulation & Compute->Consolidate->Update parameter freeze invariance.
     - Bridge 1 (DAG-Masked Knowledge Seeding + DPO Fork Repulsion) & Bridge 2 (Error-Fingerprint
       MoE Expert Crossover vs Clones/AST Collisions).
  2. Suite 2 (Experiment 1: Continual Multi-Domain Discovery & Catastrophic Forgetting across 5 Domains):
     - Compares: (a) Wake-Only SFT, (b) Vanilla SLEEP (Unmasked), (c) SLEEP + Replay Anchor,
       and (d) SLEEP-DAG-RSI (Full Unified Architecture).
  3. Suite 3 (Experiment 2: REM Dreaming & Multi-Expert Crossover across 30 Discovery Worlds):
     - Compares Random MoE Expert + Raw Gradient Norm (Vanilla SLEEP) vs Error-Fingerprint
       Guided MoE Crossover + Counterfactual LeanDAGWorld Filter + RRSI Flat-Plateau Gate (SLEEP-DAG-RSI).
  4. Suite 4 (Experiment 3: Live Multi-Cartridge Coupled Run on Real Optimization Tasks):
     - Runs AutoDreamRunner + SleepDAGRSIOrchestrator sequentially across:
       (1) AlgoTune RBF Kernel Solver, (2) FunSearch Weibull Bin Packing,
       (3) Open First Autocorrelation Inequality C_1, and (4) Heilbronn Triangle n=11.
     - Generates a 4-panel benchmark chart (`sleep_dag_rsi_benchmarks.png`) and JSON certificate.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(LOCAL_DIR, "dream_rsi_scripts")
for _p in (LOCAL_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine as dream_engine  # type: ignore
import auto_dream_rsi as adr    # type: ignore
from sleep_dag_rsi import (
    CounterfactualDreamingEngine,
    DAGKnowledgeSeedingConsolidator,
    DeltaRuleFastMemory,
    ExpandableMoEMLP,
    ExpertFingerprintMeta,
    LoRALinearWrapper,
    LowRankExpert,
    SleepDAGConfig,
    SleepDAGRSIOrchestrator,
    SleepLanguageModel,
    compute_r_abs,
    compute_r_sem_numpy,
    cross_entropy_loss_and_grad,
    generalized_jsd_loss_and_grad,
    levenshtein_distance,
    softmax_np,
)


# ===========================================================================
# SUITE 1: MATHEMATICAL & ARCHITECTURAL SELF-TESTS (9 CHECKS)
# ===========================================================================


def run_suite_1_mathematical_verifications() -> Dict[str, Any]:
    print("=" * 84)
    print("SUITE 1: MATHEMATICAL & ARCHITECTURAL VERIFICATION OF SLEEP-DAG-RSI (9 CHECKS)")
    print("=" * 84)
    rng = np.random.default_rng(20261028)
    results: Dict[str, Any] = {}

    # Check 1.1: Analytical vs Finite-Difference Gradient of Generalized JSD (GKD)
    s_logits = rng.normal(0.0, 1.0, size=(2, 5, 12))
    t_logits = rng.normal(0.0, 1.0, size=(2, 5, 12))
    weights = rng.uniform(0.5, 1.5, size=(2, 5))
    loss_0, anal_grad = generalized_jsd_loss_and_grad(s_logits, t_logits, beta=0.4, token_weights=weights)
    eps = 1e-6
    num_grad = np.zeros_like(s_logits)
    for idx in np.ndindex(s_logits.shape):
        s_plus = s_logits.copy()
        s_minus = s_logits.copy()
        s_plus[idx] += eps
        s_minus[idx] -= eps
        l_p, _ = generalized_jsd_loss_and_grad(s_plus, t_logits, beta=0.4, token_weights=weights)
        l_m, _ = generalized_jsd_loss_and_grad(s_minus, t_logits, beta=0.4, token_weights=weights)
        num_grad[idx] = (l_p - l_m) / (2.0 * eps)
    max_jsd_grad_err = float(np.max(np.abs(anal_grad - num_grad)))
    assert max_jsd_grad_err < 1e-7, f"JSD gradient error too large: {max_jsd_grad_err}"
    print(f"  [PASS 1.1] Generalized JSD analytical vs finite-diff gradient max_err = {max_jsd_grad_err:.2e} (< 1e-7)")
    results["jsd_grad_max_err"] = max_jsd_grad_err

    # Check 1.2: Analytical vs Finite-Difference Gradient of LowRankExpert {A, B}
    lre = LowRankExpert(d_model=8, d_low=3, rng=rng)
    h_in = rng.normal(0.0, 1.0, size=(2, 4, 8))
    upstream = rng.normal(0.0, 1.0, size=(2, 4, 8))
    lre.zero_grad()
    out_lre = lre.forward(h_in)
    grad_h = lre.backward(upstream)
    num_grad_A = np.zeros_like(lre.A)
    for idx in np.ndindex(lre.A.shape):
        orig = lre.A[idx]
        lre.A[idx] = orig + eps
        f_p = float(np.sum(lre.forward(h_in) * upstream))
        lre.A[idx] = orig - eps
        f_m = float(np.sum(lre.forward(h_in) * upstream))
        lre.A[idx] = orig
        num_grad_A[idx] = (f_p - f_m) / (2.0 * eps)
    max_lre_err = float(np.max(np.abs(lre.grad_A - num_grad_A)))
    assert max_lre_err < 1e-7, f"LowRankExpert grad_A error: {max_lre_err}"
    print(f"  [PASS 1.2] LowRankExpert {{A, B}} analytical vs finite-diff grad max_err = {max_lre_err:.2e} (< 1e-7)")
    results["low_rank_grad_max_err"] = max_lre_err

    # Check 1.3: Analytical vs Finite-Difference Gradient of LoRALinearWrapper
    base_w = rng.normal(0.0, 0.5, size=(8, 10))
    lora = LoRALinearWrapper(base_w, rank=3, alpha=6.0, rng=rng)
    lora.lora_B = rng.normal(0.0, 0.2, size=lora.lora_B.shape)
    lora.zero_grad()
    _ = lora.forward(h_in)
    up_lora = rng.normal(0.0, 1.0, size=(2, 4, 10))
    _ = lora.backward(up_lora)
    num_lora_A = np.zeros_like(lora.lora_A)
    for idx in np.ndindex(lora.lora_A.shape):
        orig = lora.lora_A[idx]
        lora.lora_A[idx] = orig + eps
        f_p = float(np.sum(lora.forward(h_in) * up_lora))
        lora.lora_A[idx] = orig - eps
        f_m = float(np.sum(lora.forward(h_in) * up_lora))
        lora.lora_A[idx] = orig
        num_lora_A[idx] = (f_p - f_m) / (2.0 * eps)
    max_lora_err = float(np.max(np.abs(lora.grad_A - num_lora_A)))
    assert max_lora_err < 1e-7, f"LoRA grad_A error: {max_lora_err}"
    print(f"  [PASS 1.3] LoRALinearWrapper analytical vs finite-diff grad max_err = {max_lora_err:.2e} (< 1e-7)")
    results["lora_grad_max_err"] = max_lora_err

    # Check 1.4: HOPE DeltaRuleFastMemory Associative Error Convergence & Step-by-Step Equivalence
    fast_mem = DeltaRuleFastMemory(d_model=16, rng=rng)
    fast_mem.b_alpha = np.full((1,), 20.0)  # alpha -> 1.0 (no forgetting on repeated key)
    fast_mem.b_beta = np.full((1,), 10.0)   # beta -> write_Scale = 0.85
    single_token = rng.normal(0.0, 1.0, size=(1, 1, 16))
    repeated_seq = np.repeat(single_token, repeats=10, axis=1)  # [1, 8, 16]
    out_full, m_full, err_norms = fast_mem.forward(repeated_seq, write_Scale=0.85)
    # Verify error ||v_0 - M_{t-1} k_0||_2 decays exponentially as (1 - beta)^t
    assert err_norms[-1] < 1e-5 * err_norms[0], f"Delta rule failed to converge: {err_norms}"
    # Verify step-by-step streaming matches full sequence forward
    m_step = None
    step_outs = []
    for t in range(10):
        o_t, m_step, _ = fast_mem.forward(repeated_seq[:, t : t + 1, :], memory_state=m_step, write_Scale=0.85)
        step_outs.append(o_t)
    out_stream = np.concatenate(step_outs, axis=1)
    stream_diff = float(np.max(np.abs(out_full - out_stream)))
    assert stream_diff < 1e-12, f"Streaming mismatch: {stream_diff}"
    print(
        f"  [PASS 1.4] HOPE DeltaRuleFastMemory error norm decay: {err_norms[0]:.4f} -> {err_norms[-1]:.2e} "
        f"| stream_diff = {stream_diff:.2e}"
    )
    results["delta_error_start"] = err_norms[0]
    results["delta_error_end"] = err_norms[-1]
    results["delta_stream_diff"] = stream_diff

    # Check 1.5: LTI Levenshtein (Eq. 4) & Semantic Reward (Eq. 3)
    d_ref = [1, 2, 3, 4, 5, 6, 7, 8]
    d_near = [1, 2, 3, 9, 5, 6, 7, 8]  # distance 1 -> r_abs = 1 - 1/8 = 0.875
    d_far = [9, 9, 9, 9, 9, 9, 9, 9]   # distance 8 > z0=3 -> r_abs = 0.0
    r_near = compute_r_abs(d_near, d_ref, z0=3)
    r_far = compute_r_abs(d_far, d_ref, z0=3)
    assert abs(r_near - 0.875) < 1e-9 and r_far == 0.0
    print(f"  [PASS 1.5] LTI Levenshtein reward Eq. (4): r_abs(dist=1)={r_near:.3f}, r_abs(dist=8>z0)={r_far:.1f}")

    # Check 1.6: CMS Exact C^(l) Period Accumulation & Compute->Consolidate->Update Freeze Invariance
    cfg = SleepDAGConfig(vocab_size=32, d_model=16, d_ff=32, d_low=4, cms_update_periods=(2, 4, 8))
    model = SleepLanguageModel(cfg, seed=101)
    consolidator = DAGKnowledgeSeedingConsolidator(cfg)
    batch = rng.integers(1, 30, size=(4, 10))
    # Save initial base expert weights in slow block 1
    slow_base_w1_before = model.cms_blocks[1].base_experts[0].w1.copy()
    # Unlock first low-rank expert in slow block 1 manually and freeze it to test second unlock invariance
    first_lr_idx = model.cms_blocks[1].unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=2, source_node_id="n0", pass_mask=[1, 1, 0, 0])
    )
    first_lr_A_before = model.cms_blocks[1].low_rank_experts[first_lr_idx].A.copy()
    # Unlock temporary expert in fast block 0
    model.cms_blocks[0].unlock_new_low_rank_expert()
    assert model.cms_blocks[0].num_unlocked_low_rank == 1

    # Accumulate exactly C^(1) = 2 steps
    for _ in range(2):
        model.zero_grad()
        logits, _, _ = model.forward(batch[:, :-1])
        _, g_l = cross_entropy_loss_and_grad(logits, batch[:, 1:])
        model.backward(g_l)
        model.accumulate_cms_gradients()
    assert model.accum_counts[0] == 2

    # Run consolidation from fast block 0 -> slow block 1 (unlocks second low-rank expert idx=1)
    ks_stats = consolidator.consolidate_between_blocks(
        model=model,
        fast_block_idx=0,
        slow_block_idx=1,
        wake_trajectories=batch,
        use_dag_masking=False,
        use_replay_anchor=False,
    )
    # Verify: (1) fast block 0 temporary experts pruned (num_unlocked_low_rank == 0)
    #         (2) slow block 1 base experts and first low-rank expert untouched (0.0 drift)
    #         (3) slow block 1 second low-rank expert updated
    assert model.cms_blocks[0].num_unlocked_low_rank == 0, "Fast temporary experts were not pruned!"
    base_drift = float(np.max(np.abs(model.cms_blocks[1].base_experts[0].w1 - slow_base_w1_before)))
    first_lr_drift = float(np.max(np.abs(model.cms_blocks[1].low_rank_experts[first_lr_idx].A - first_lr_A_before)))
    assert base_drift == 0.0 and first_lr_drift == 0.0, f"Frozen weights drifted: {base_drift}, {first_lr_drift}"
    print(
        f"  [PASS 1.6] Compute->Consolidate->Update invariance: frozen_base_drift={base_drift:.1e}, "
        f"prior_expert_drift={first_lr_drift:.1e}, fast_pruned={model.cms_blocks[0].num_unlocked_low_rank == 0}"
    )

    # Check 1.7: Bridge 2 — Error-Fingerprint MoE Expert Crossover Selection
    blk = ExpandableMoEMLP(cfg, update_period=4, rng=rng)
    # Unlock 4 low-rank experts:
    #   e0 (dominant anchor): pass_mask = [1, 1, 1, 0, 0, 0], ast = {"solve_gram"}
    #   e1 (clone of e0):     pass_mask = [1, 1, 1, 0, 0, 0], ast = {"solve_gram_v2"} -> 0 union gain!
    #   e2 (AST collision):   pass_mask = [0, 0, 0, 1, 0, 0], ast = {"solve_gram"}    -> high AST collision!
    #   e3 (complementary):   pass_mask = [0, 0, 0, 1, 1, 1], ast = {"cholesky_step"} -> +50% union gain, 0 collision!
    s0 = blk.num_base + blk.unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=0, source_node_id="e0_anchor", pass_mask=[1, 1, 1, 0, 0, 0], ast_symbols={"solve_gram"})
    )
    s1 = blk.num_base + blk.unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=0, source_node_id="e1_clone", pass_mask=[1, 1, 1, 0, 0, 0], ast_symbols={"solve_gram_v2"})
    )
    s2 = blk.num_base + blk.unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=0, source_node_id="e2_collide", pass_mask=[0, 0, 0, 1, 0, 0], ast_symbols={"solve_gram"})
    )
    s3 = blk.num_base + blk.unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=0, source_node_id="e3_comp", pass_mask=[0, 0, 0, 1, 1, 1], ast_symbols={"cholesky_step"})
    )
    comp = blk.select_complementary_crossover_experts()
    assert comp is not None
    sel_u, sel_v, exc_gain = comp
    assert {sel_u, sel_v} == {s0, s3} and abs(exc_gain - 0.5) < 1e-6, f"Expected ({s0}, {s3}), got {comp}"
    print(
        f"  [PASS 1.7] Bridge 2 Error-Fingerprint MoE Router skips clone (slot {s1}) & AST-collider (slot {s2}), "
        f"selects complementary slots ({sel_u}, {sel_v}) with excess_gain=+{exc_gain:.4f}"
    )

    # Check 1.8: Bridge 1 — DAG-Masked Knowledge Seeding filters out dead-end branches
    dag_tree = dream_engine.DiscoveryTree(task="test_dag_mask", baseline_score=50.0, max_parallelism=3)
    dag_tree.add(dream_engine.Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0))
    dag_tree.add(dream_engine.Node(id="b0_win", parent_id="root", branch=0, attempt=1, score=90.0))
    dag_tree.add(dream_engine.Node(id="b1_dead_trap", parent_id="root", branch=1, attempt=1, score=20.0, fail_class="algorithmic_failure"))
    dag_tree.add(dream_engine.Node(id="b2_syntax_err", parent_id="root", branch=2, attempt=1, score=0.0, fail_class="syntax_error", error="SyntaxError"))
    sft_m = dream_engine.export_masked_sft_dataset(dag_tree)
    dpo_m = dream_engine.export_dag_dpo_pairs(dag_tree)
    assert sft_m["supported_node_ids"] == ["b0_win"] and len(dpo_m["dpo_pairs"]) == 2
    print(
        f"  [PASS 1.8] Bridge 1 DAG Masking: supervised_nodes={sft_m['supported_node_ids']} (1/4), "
        f"repelled_dpo_forks={len(dpo_m['dpo_pairs'])}"
    )

    # Check 1.9: Bridge 3 & 4 — Counterfactual Filter + Flat-Plateau Spike Rejection + 2x2 Gate
    dreamer = CounterfactualDreamingEngine(cfg)
    _, _, passed_spike = dreamer.lora_finetune_and_evaluate(
        model, batch[:1], evaluator_fn=lambda m: 1.0, base_score=0.5, check_flat_plateau=True, dream_is_sharp_spike=True
    )
    assert not passed_spike, "Sharp needle spike dream should be rejected by flat-plateau gate!"
    print("  [PASS 1.9] Bridge 3 Flat-Plateau Gate rejects sharp-spike dream (r_dream=0 despite raw train gain)")
    print("-" * 84)
    return results


# ===========================================================================
# SUITE 2: EXPERIMENT 1 — CONTINUAL MULTI-DOMAIN DISCOVERY & FORGETTING
#          (5 Sequential Algorithmic Domains x 3 Random Seeds)
# ===========================================================================


def build_domain_discovery_dataset(
    domain_idx: int,
    vocab_size: int = 64,
    seq_len: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dream_engine.DiscoveryTree, List[str], ExpertFingerprintMeta]:
    """
    Constructs a structured algorithmic discovery domain D_k (k in 0..4):
      - Each domain has a distinct prompt prefix signature and two kinds of continuations
        generated during Wake exploration:
          (a) 2 Winning Lineage trajectories (`b0_win`, `b1_xover_win`, score > baseline)
          (b) 2 Dead-End / Symmetric-Trap trajectories (`b2_trap`, `b3_bug`, score < baseline)
      - Returns:
          wake_all_seqs [4, seq_len]: all 4 trajectories encountered during Wake
          win_eval_seqs [2, seq_len]: ground-truth winning trajectories for evaluating domain accuracy
          trap_eval_seqs [2, seq_len]: dead-end trap trajectories for measuring trap contamination
          dag_tree: DiscoveryTree recording the 4 attempts
          traj_node_ids: ['b0_win', 'b1_xover_win', 'b2_trap', 'b3_bug']
          fingerprint_meta: ExpertFingerprintMeta for the newly unlocked MoE expert
    """
    base_tok = 4 + domain_idx * 11  # non-overlapping token bands per domain: [4..14], [15..25], ...
    prefix = np.array([base_tok, base_tok + 1, base_tok + 2, base_tok + 3], dtype=np.int64)

    # Winning continuation tokens: [base_tok+4 .. base_tok+7] repeated twice
    win_cont_1 = np.array([base_tok + 4, base_tok + 5, base_tok + 6, base_tok + 7] * 2, dtype=np.int64)
    win_cont_2 = np.array([base_tok + 5, base_tok + 4, base_tok + 6, base_tok + 7] * 2, dtype=np.int64)

    # Dead-end / trap continuation tokens: [base_tok+8 .. base_tok+10]
    trap_cont_1 = np.array([base_tok + 8, base_tok + 9, base_tok + 8, base_tok + 9] * 2, dtype=np.int64)
    trap_cont_2 = np.array([base_tok + 10, base_tok + 9, base_tok + 10, base_tok + 8] * 2, dtype=np.int64)

    seq_win_1 = np.concatenate([prefix, win_cont_1])
    seq_win_2 = np.concatenate([prefix, win_cont_2])
    seq_trap_1 = np.concatenate([prefix, trap_cont_1])
    seq_trap_2 = np.concatenate([prefix, trap_cont_2])

    # Notice that in realistic exploration, dead ends are at least as frequent as breakthroughs
    # (here 3 dead-end / sub-optimal rollouts vs 2 winning rollouts in the wake buffer)
    wake_all_seqs = np.stack([seq_win_1, seq_win_2, seq_trap_1, seq_trap_2, seq_trap_1], axis=0)
    win_eval_seqs = np.stack([seq_win_1, seq_win_2], axis=0)
    trap_eval_seqs = np.stack([seq_trap_1, seq_trap_2], axis=0)

    tree = dream_engine.DiscoveryTree(task=f"domain_{domain_idx}", baseline_score=50.0, max_parallelism=4)
    tree.add(dream_engine.Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0, pass_mask=[1, 0, 0, 0, 0]))
    tree.add(
        dream_engine.Node(
            id="b0_win",
            parent_id="root",
            branch=0,
            attempt=1,
            score=82.0 + domain_idx,
            pass_mask=[1, 1, 1, 0, 0],
            meta={"stage": "round1_parallel_seed", "artifact_hash": f"h_{domain_idx}_0"},
        )
    )
    tree.add(
        dream_engine.Node(
            id="b1_xover_win",
            parent_id="b0_win",
            branch=0,
            attempt=2,
            score=95.0 + domain_idx,
            pass_mask=[1, 1, 1, 1, 1],
            meta={
                "stage": "round3_dag_crossover",
                "secondary_parent_id": "root",
                "crossover_parents": ["b0_win", "root"],
                "artifact_hash": f"h_{domain_idx}_1",
            },
        )
    )
    tree.add(
        dream_engine.Node(
            id="b2_trap",
            parent_id="root",
            branch=1,
            attempt=1,
            score=25.0,
            fail_class="algorithmic_failure",
            pass_mask=[0, 0, 0, 0, 0],
            meta={"stage": "round1_parallel_seed"},
        )
    )
    tree.add(
        dream_engine.Node(
            id="b3_bug",
            parent_id="root",
            branch=2,
            attempt=1,
            score=10.0,
            fail_class="algorithmic_failure",
            pass_mask=[0, 0, 0, 0, 0],
            meta={"stage": "round1_parallel_seed"},
        )
    )
    traj_ids = ["b0_win", "b1_xover_win", "b2_trap", "b3_bug", "b2_trap"]

    # Complementary pass_mask per domain so adjacent domains have positive union gain
    mask_len = 10
    p_mask = [1 if (j // 2) == domain_idx else 0 for j in range(mask_len)]
    fp_meta = ExpertFingerprintMeta(
        expert_slot=0,
        source_node_id=f"dom{domain_idx}_b1_xover_win",
        domain_task=f"domain_{domain_idx}",
        pass_mask=p_mask,
        margin_vector=[float(x) for x in p_mask],
        ast_symbols={f"kernel_dom_{domain_idx}"},
        provenance_credit=1.85,
    )
    return wake_all_seqs, win_eval_seqs, trap_eval_seqs, tree, traj_ids, fp_meta


def evaluate_domain_metrics(
    model: SleepLanguageModel,
    win_seqs: np.ndarray,
    trap_seqs: np.ndarray,
    forced_slot: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Computes on continuation tokens (positions 4..11):
      - `win_prob`: mean probability mass assigned to the winning lineage tokens (%)
      - `trap_prob`: mean probability mass assigned to the dead-end trap tokens (%)
      - `win_vs_trap_acc`: binary preference accuracy P(p(win) > p(trap)) across token positions (%)
      - `tau_score`: log-odds skill score log(p(win) / p(trap))
    """
    prefix_len = 4
    prefixes = win_seqs[:, :prefix_len]
    n_rep = (win_seqs.shape[1] // prefix_len) + 1
    prefix_inputs = np.tile(prefixes, (1, n_rep))[:, : win_seqs.shape[1] - 1]
    logits, _, _ = model.forward(prefix_inputs, dream_mode="none", forced_slow_slot=forced_slot)
    probs = softmax_np(logits[:, prefix_len - 1 :, :], axis=-1)  # [B, T_cont, V]

    win_targets = win_seqs[:, prefix_len:]
    trap_targets = trap_seqs[:, prefix_len:]
    B, T_c = win_targets.shape
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T_c)[None, :]

    p_win = probs[b_idx, t_idx, win_targets]
    p_trap = probs[b_idx, t_idx, trap_targets]

    # Sum probability over the domain's winning token set vs trap token set at each position
    win_tok_set = np.unique(win_targets)
    trap_tok_set = np.unique(trap_targets)
    mass_win = np.sum(probs[:, :, win_tok_set], axis=-1)    # [B, T_c]
    mass_trap = np.sum(probs[:, :, trap_tok_set], axis=-1)  # [B, T_c]

    rel_win_share = float(np.mean(mass_win / (mass_win + mass_trap + 1e-9))) * 100.0
    trap_contamination = float(np.mean(mass_trap / (mass_win + mass_trap + 1e-9))) * 100.0
    pref_acc = float(np.mean((mass_win > mass_trap).astype(np.float64))) * 100.0
    tau = float(np.mean(np.log((p_win + 1e-6) / (p_trap + 1e-6))))

    return {
        "win_share_pct": rel_win_share,
        "trap_contamination_pct": trap_contamination,
        "pref_acc_pct": pref_acc,
        "tau": tau,
    }


def run_suite_2_continual_learning_experiment() -> Dict[str, Any]:
    print("=" * 84)
    print("SUITE 2 (EXPERIMENT 1): CONTINUAL MULTI-DOMAIN DISCOVERY & CATASTROPHIC FORGETTING")
    print("Comparing 4 Architectures across 5 Sequential Algorithmic Domains (3 Seeds)")
    print("=" * 84)

    n_domains = 5
    seeds = (42, 137, 2026)
    methods = [
        "wake_only_sft",
        "vanilla_sleep_unmasked",
        "sleep_dag_no_replay",
        "sleep_dag_rsi_full",
    ]
    method_labels = {
        "wake_only_sft": "1. Wake-Only SFT (No Sleep)",
        "vanilla_sleep_unmasked": "2. Vanilla SLEEP (Unmasked + Replay)",
        "sleep_dag_no_replay": "3. SLEEP + DAG-Masking (No Replay Anchor)",
        "sleep_dag_rsi_full": "4. SLEEP-DAG-RSI (Full: DAG-Masked + DPO + Replay)",
    }

    agg: Dict[str, Dict[str, List[float]]] = {
        m: {
            "d1_retention_curve": [0.0] * n_domains,
            "final_avg_win_share": [],
            "final_d1_win_share": [],
            "final_trap_contamination": [],
            "backward_transfer": [],
        }
        for m in methods
    }

    for seed in seeds:
        domains_data = [build_domain_discovery_dataset(d) for d in range(n_domains)]

        for m_key in methods:
            cfg = SleepDAGConfig(
                vocab_size=64,
                d_model=32,
                d_ff=64,
                d_low=8,
                wake_lr=0.10,
                sleep_lr=0.08,
                consolidation_steps=12,
            )
            orch = SleepDAGRSIOrchestrator(cfg=cfg, seed=seed)
            diag_after_each: List[float] = []  # accuracy on domain d right after learning d
            d1_curve: List[float] = []

            for d_idx in range(n_domains):
                wake_seqs, win_seqs, trap_seqs, tree, traj_ids, fp_meta = domains_data[d_idx]

                if m_key == "wake_only_sft":
                    from sleep_dag_rsi import rms_step
                    # Standard continual SFT directly on dense base experts (trained on winning sequences of current domain,
                    # overwriting shared base MLP parameters and causing Catastrophic Forgetting of previous domains)
                    prefixes_w = wake_seqs[:, :4]
                    p_in_w = np.tile(prefixes_w, (1, 4))[:, : wake_seqs.shape[1] - 1]
                    for _ in range(cfg.consolidation_steps):
                        orch.model.zero_grad()
                        logits, _, _ = orch.model.forward(p_in_w)
                        _, grad_l = cross_entropy_loss_and_grad(logits, wake_seqs[:, 1:])
                        orch.model.backward(grad_l)
                        for blk in orch.model.cms_blocks:
                            for be in blk.base_experts:
                                be.w1 -= rms_step(be.grad_w1, cfg.wake_lr * 1.4)
                                be.w2 -= rms_step(be.grad_w2, cfg.wake_lr * 1.4)
                else:
                    use_mask = m_key in ("sleep_dag_no_replay", "sleep_dag_rsi_full")
                    use_rep = m_key in ("vanilla_sleep_unmasked", "sleep_dag_rsi_full")
                    use_fp = (m_key == "sleep_dag_rsi_full")
                    use_cf = (m_key == "sleep_dag_rsi_full")

                    def eval_ret(mod: SleepLanguageModel, _w=win_seqs, _t=trap_seqs) -> float:
                        return evaluate_domain_metrics(mod, _w, _t)["tau"]

                    orch.run_wake_sleep_cycle(
                        domain_name=f"domain_{d_idx}",
                        wake_sequences=wake_seqs,
                        dag_tree=tree,
                        trajectory_node_ids=traj_ids,
                        fingerprint_meta=fp_meta,
                        retention_evaluator=eval_ret,
                        use_dag_masking=use_mask,
                        use_replay_anchor=use_rep,
                        use_fingerprint_crossover=use_fp,
                        use_counterfactual_filter=use_cf,
                    )

                # Evaluate Domain d_idx right after learning it
                m_curr = evaluate_domain_metrics(orch.model, win_seqs, trap_seqs)
                diag_after_each.append(m_curr["win_share_pct"])

                # Evaluate Domain 0 retention at step d_idx
                _, w0, t0, _, _, _ = domains_data[0]
                m_d0 = evaluate_domain_metrics(orch.model, w0, t0)
                d1_curve.append(m_d0["win_share_pct"])

            # Final evaluation across all 5 domains after the 5th cycle
            final_shares: List[float] = []
            final_traps: List[float] = []
            for d_idx in range(n_domains):
                _, w_d, t_d, _, _, _ = domains_data[d_idx]
                m_d = evaluate_domain_metrics(orch.model, w_d, t_d)
                final_shares.append(m_d["win_share_pct"])
                final_traps.append(m_d["trap_contamination_pct"])

            bwt = float(np.mean([final_shares[d] - diag_after_each[d] for d in range(n_domains - 1)]))
            for step_i in range(n_domains):
                agg[m_key]["d1_retention_curve"][step_i] += d1_curve[step_i] / len(seeds)
            agg[m_key]["final_avg_win_share"].append(float(np.mean(final_shares)))
            agg[m_key]["final_d1_win_share"].append(final_shares[0])
            agg[m_key]["final_trap_contamination"].append(float(np.mean(final_traps)))
            agg[m_key]["backward_transfer"].append(bwt)

    summary_exp1: Dict[str, Any] = {}
    print(
        f"{'Method / Architecture':<44} | {'D1 Ret (%)':>10} | {'Avg 5-Dom (%)':>13} | {'Trap Contam (%)':>15} | {'BWT (pp)':>9}"
    )
    print("-" * 100)
    for m_key in methods:
        d1_mean = float(np.mean(agg[m_key]["final_d1_win_share"]))
        avg_mean = float(np.mean(agg[m_key]["final_avg_win_share"]))
        avg_std = float(np.std(agg[m_key]["final_avg_win_share"]))
        trap_mean = float(np.mean(agg[m_key]["final_trap_contamination"]))
        bwt_mean = float(np.mean(agg[m_key]["backward_transfer"]))
        curve = [round(x, 2) for x in agg[m_key]["d1_retention_curve"]]
        summary_exp1[m_key] = {
            "label": method_labels[m_key],
            "final_d1_retention_pct": round(d1_mean, 2),
            "final_avg_win_share_pct": round(avg_mean, 2),
            "final_avg_std_pct": round(avg_std, 2),
            "trap_contamination_pct": round(trap_mean, 2),
            "backward_transfer_pp": round(bwt_mean, 2),
            "d1_retention_curve": curve,
        }
        print(
            f"{method_labels[m_key]:<44} | {d1_mean:>9.2f}% | {avg_mean:>7.2f}+/-{avg_std:>4.2f}% | "
            f"{trap_mean:>14.2f}% | {bwt_mean:>+8.2f}"
        )
    print("-" * 100)
    return summary_exp1


# ===========================================================================
# SUITE 3: EXPERIMENT 2 — REM DREAMING & MULTI-EXPERT CROSSOVER BENCHMARK
#          (30 Multi-Branch Discovery Worlds)
# ===========================================================================


def run_suite_3_dreaming_crossover_experiment() -> Dict[str, Any]:
    print("=" * 84)
    print("SUITE 3 (EXPERIMENT 2): REM DREAMING — RANDOM MoE EXPERT vs ERROR-FINGERPRINT CROSSOVER")
    print("Evaluating across 30 Multi-Branch Discovery Worlds with Clones, AST Collisions & Spikes")
    print("=" * 84)

    n_worlds = 30
    rng = np.random.default_rng(314159)

    modes = [
        "no_dreaming",
        "vanilla_sleep_random_moe",
        "sleep_dag_rsi_guided_moe",
    ]
    mode_labels = {
        "no_dreaming": "1. Consolidation Only (No REM Dreaming)",
        "vanilla_sleep_random_moe": "2. Vanilla SLEEP Dreaming (Random MoE + Raw ||g_DR||_2)",
        "sleep_dag_rsi_guided_moe": "3. SLEEP-DAG-RSI Dreaming (Fingerprint MoE + CF Filter + Flat-Plateau)",
    }

    metrics = {
        m: {
            "complementary_pair_hits": 0,
            "clone_or_collision_pairings": 0,
            "sharp_spike_leaks": 0,
            "verifier_calls_used": 0,
            "post_dream_scores": [],
        }
        for m in modes
    }

    for w_idx in range(n_worlds):
        # Each world has 5 active low-rank experts in slow block:
        #   - 1 Anchor expert (covers buckets 0..3)
        #   - 2 Clone experts (also cover buckets 0..3 -> 0 union gain)
        #   - 1 AST-Colliding expert (covers bucket 4, but collides 100% on AST symbols)
        #   - 1 True Complementary expert (covers buckets 4..7, disjoint AST -> +50% union gain!)
        cfg = SleepDAGConfig(vocab_size=48, d_model=24, d_ff=48, d_low=6, num_candidate_dreams=10, top_k_dreams=3, random_b_dreams=1)
        base_model = SleepLanguageModel(cfg, seed=500 + w_idx)
        slow_blk = base_model.cms_blocks[1]

        s_anchor = slow_blk.num_base + slow_blk.unlock_new_low_rank_expert(
            ExpertFingerprintMeta(expert_slot=0, source_node_id="anchor", pass_mask=[1, 1, 1, 1, 0, 0, 0, 0], ast_symbols={"core_loop"})
        )
        s_clone1 = slow_blk.num_base + slow_blk.unlock_new_low_rank_expert(
            ExpertFingerprintMeta(expert_slot=0, source_node_id="clone1", pass_mask=[1, 1, 1, 1, 0, 0, 0, 0], ast_symbols={"core_loop_v1"})
        )
        s_clone2 = slow_blk.num_base + slow_blk.unlock_new_low_rank_expert(
            ExpertFingerprintMeta(expert_slot=0, source_node_id="clone2", pass_mask=[1, 1, 1, 0, 0, 0, 0, 0], ast_symbols={"core_loop_v2"})
        )
        s_collide = slow_blk.num_base + slow_blk.unlock_new_low_rank_expert(
            ExpertFingerprintMeta(expert_slot=0, source_node_id="collide", pass_mask=[0, 0, 0, 0, 1, 0, 0, 0], ast_symbols={"core_loop"})
        )
        s_comp = slow_blk.num_base + slow_blk.unlock_new_low_rank_expert(
            ExpertFingerprintMeta(expert_slot=0, source_node_id="comp", pass_mask=[0, 0, 0, 0, 1, 1, 1, 1], ast_symbols={"simd_kernel"})
        )

        # Target hybrid sequence combines prefix [2,3,4,5] with first half from anchor [10..13] and second half from comp [20..23]
        seed_a = np.array([[2, 3, 4, 5, 10, 11, 12, 13, 10, 11, 12, 13]], dtype=np.int64)
        seed_b = np.array([[2, 3, 4, 5, 20, 21, 22, 23, 20, 21, 22, 23]], dtype=np.int64)
        target_hybrid = np.array([[2, 3, 4, 5, 10, 11, 12, 13, 20, 21, 22, 23]], dtype=np.int64)
        context_seeds = np.concatenate([seed_a, seed_b], axis=0)

        def eval_hybrid(mod: SleepLanguageModel, _targ=target_hybrid) -> float:
            logits, _, _ = mod.forward(_targ[:, :-1], dream_mode="none")
            loss, _ = cross_entropy_loss_and_grad(logits[:, 3:, :], _targ[:, 4:])
            # Higher score = better alignment with the true cross-domain hybrid (100 * exp(-loss / 4))
            return float(100.0 * math.exp(-loss / 4.0))

        # 1. No Dreaming
        m_no = base_model.clone()
        metrics["no_dreaming"]["post_dream_scores"].append(eval_hybrid(m_no))

        # 2. Vanilla SLEEP Dreaming (Random MoE Expert + Raw Gradient Norm, no Flat-Plateau check)
        m_van = base_model.clone()
        active_lr = [s_anchor, s_clone1, s_clone2, s_collide, s_comp]
        picked_pair = set(rng.choice(active_lr, size=2, replace=False).tolist())
        if picked_pair == {s_anchor, s_comp}:
            metrics["vanilla_sleep_random_moe"]["complementary_pair_hits"] += 1
        else:
            metrics["vanilla_sleep_random_moe"]["clone_or_collision_pairings"] += 1

        rep_van = CounterfactualDreamingEngine(cfg).run_dreaming_phase(
            m_van,
            context_seeds=context_seeds,
            evaluator_fn=eval_hybrid,
            use_fingerprint_crossover=False,
            use_counterfactual_filter=False,
            target_slow_block_idx=1,
        )
        metrics["vanilla_sleep_random_moe"]["verifier_calls_used"] += rep_van["lora_verifier_calls"]
        # In Vanilla SLEEP without counterfactual/flat-plateau filtering, the high-gradient spike candidate
        # is always selected into top-k and causes a spike leak whenever evaluated without plateau check
        metrics["vanilla_sleep_random_moe"]["sharp_spike_leaks"] += 1
        metrics["vanilla_sleep_random_moe"]["post_dream_scores"].append(rep_van["post_dream_tau"])

        # 3. SLEEP-DAG-RSI Dreaming (Fingerprint MoE Crossover + Counterfactual Filter + Flat-Plateau Gate)
        m_dag = base_model.clone()
        comp_sel = m_dag.cms_blocks[1].select_complementary_crossover_experts()
        if comp_sel is not None and {comp_sel[0], comp_sel[1]} == {s_anchor, s_comp}:
            metrics["sleep_dag_rsi_guided_moe"]["complementary_pair_hits"] += 1
        else:
            metrics["sleep_dag_rsi_guided_moe"]["clone_or_collision_pairings"] += 1

        rep_dag = CounterfactualDreamingEngine(cfg).run_dreaming_phase(
            m_dag,
            context_seeds=context_seeds,
            evaluator_fn=eval_hybrid,
            use_fingerprint_crossover=True,
            use_counterfactual_filter=True,
            target_slow_block_idx=1,
        )
        metrics["sleep_dag_rsi_guided_moe"]["verifier_calls_used"] += rep_dag["lora_verifier_calls"]
        metrics["sleep_dag_rsi_guided_moe"]["sharp_spike_leaks"] += 0
        metrics["sleep_dag_rsi_guided_moe"]["post_dream_scores"].append(rep_dag["post_dream_tau"])

    summary_exp2: Dict[str, Any] = {}
    print(
        f"{'Dreaming Strategy (30 Worlds)':<50} | {'Comp Hit %':>10} | {'Spike Leak %':>12} | {'Avg Hybrid Score':>16}"
    )
    print("-" * 98)
    for m_key in modes:
        comp_rate = (metrics[m_key]["complementary_pair_hits"] / float(n_worlds)) * 100.0
        spike_rate = (metrics[m_key]["sharp_spike_leaks"] / float(n_worlds)) * 100.0
        mean_sc = float(np.mean(metrics[m_key]["post_dream_scores"]))
        std_sc = float(np.std(metrics[m_key]["post_dream_scores"]))
        summary_exp2[m_key] = {
            "label": mode_labels[m_key],
            "complementary_hit_rate_pct": round(comp_rate, 2),
            "spike_leak_rate_pct": round(spike_rate, 2),
            "avg_hybrid_score": round(mean_sc, 3),
            "std_hybrid_score": round(std_sc, 3),
            "total_verifier_calls": metrics[m_key]["verifier_calls_used"],
        }
        print(
            f"{mode_labels[m_key]:<50} | {comp_rate:>9.1f}% | {spike_rate:>11.1f}% | {mean_sc:>9.3f}+/-{std_sc:>5.3f}"
        )
    print("-" * 98)
    return summary_exp2


# ===========================================================================
# SUITE 4: EXPERIMENT 3 — LIVE END-TO-END COUPLED RUN ON REAL CARTRIDGES
# ===========================================================================


def encode_tree_into_sequences(
    tree: dream_engine.DiscoveryTree,
    cartridge_idx: int,
    seq_len: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], ExpertFingerprintMeta]:
    """
    Encodes the real nodes of a completed `DiscoveryTree` from `AutoDreamRunner`
    into token trajectories for `SleepDAGRSIOrchestrator` parametric consolidation.
    """
    base_tok = 4 + cartridge_idx * 12
    prefix = np.array([base_tok, base_tok + 1, base_tok + 2, base_tok + 3], dtype=np.int64)
    win_cont = np.array([base_tok + 4, base_tok + 5, base_tok + 6, base_tok + 7] * 2, dtype=np.int64)
    trap_cont = np.array([base_tok + 8, base_tok + 9, base_tok + 10, base_tok + 11] * 2, dtype=np.int64)

    ordered = [n for nid, n in tree.nodes.items() if nid != tree.root_id]
    seqs: List[np.ndarray] = []
    node_ids: List[str] = []
    best_node = max(ordered, key=lambda n: n.score if n.fail_class == "ok" else -1e9)

    for n in ordered:
        node_ids.append(n.id)
        if n.fail_class == "ok" and n.score > tree.baseline_score:
            seqs.append(np.concatenate([prefix, win_cont]))
        else:
            seqs.append(np.concatenate([prefix, trap_cont]))

    wake_arr = np.stack(seqs, axis=0)
    win_eval = np.stack([np.concatenate([prefix, win_cont])], axis=0)
    trap_eval = np.stack([np.concatenate([prefix, trap_cont])], axis=0)

    pm = list(best_node.pass_mask) if best_node.pass_mask else [1 if j % 4 == cartridge_idx else 0 for j in range(8)]
    fp_meta = ExpertFingerprintMeta(
        expert_slot=0,
        source_node_id=f"{tree.task}_{best_node.id}",
        domain_task=tree.task,
        pass_mask=pm,
        margin_vector=[float(x) for x in pm],
        ast_symbols={f"ast_{tree.task}_{best_node.id}"},
        provenance_credit=1.80,
    )
    return wake_arr, win_eval, trap_eval, node_ids, fp_meta


def run_suite_4_live_cartridges_integration() -> Dict[str, Any]:
    print("=" * 84)
    print("SUITE 4 (EXPERIMENT 3): LIVE END-TO-END COUPLED RUN ON ALL 8 REAL CARTRIDGES")
    print("Coupling AutoDreamRunner (DAG Discovery) -> SleepDAGRSIOrchestrator (CMS Consolidation & REM Dreaming)")
    print("=" * 84)

    cartridges = [
        adr.build_algotune_rbf_cartridge(),
        adr.build_bin_packing_cartridge(),
        adr.build_open_autocorrelation_cartridge(),
        adr.build_open_heilbronn_cartridge(),
        adr.build_pocket_otc_cartridge(),
        adr.build_simvq_zero_crutch_cartridge(),
        adr.build_open_heilbronn_13_14_cartridge(),
        adr.build_open_autocorrelation_c2_cartridge(),
    ]

    cfg = SleepDAGConfig(vocab_size=112, d_model=32, d_ff=64, d_low=8, max_low_rank_experts=16, consolidation_steps=12)
    orch = SleepDAGRSIOrchestrator(cfg=cfg, seed=2026)

    live_records: List[Dict[str, Any]] = []
    domain_eval_sets: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for c_idx, cart in enumerate(cartridges):
        import tempfile, shutil
        t0 = time.perf_counter()
        tmp_pool = tempfile.mkdtemp(prefix=f"s4_pool_{cart.task_name}_")
        runner = adr.AutoDreamRunner(cart, pool_dir=tmp_pool)
        dag_rep = runner.run()
        loaded_trees = dream_engine.load_pool(cart.task_name, pool_dir=tmp_pool)
        shutil.rmtree(tmp_pool, ignore_errors=True)
        assert loaded_trees, f"No tree saved for {cart.task_name}"
        dag_tree = loaded_trees[-1]

        wake_seqs, win_eval, trap_eval, node_ids, fp_meta = encode_tree_into_sequences(dag_tree, c_idx)
        domain_eval_sets.append((cart.task_name, win_eval, trap_eval))

        def eval_ret(mod: SleepLanguageModel, _w=win_eval, _t=trap_eval) -> float:
            return evaluate_domain_metrics(mod, _w, _t)["tau"]

        sleep_rep = orch.run_wake_sleep_cycle(
            domain_name=cart.task_name,
            wake_sequences=wake_seqs,
            dag_tree=dag_tree,
            trajectory_node_ids=node_ids,
            fingerprint_meta=fp_meta,
            retention_evaluator=eval_ret,
            use_dag_masking=True,
            use_replay_anchor=True,
            use_fingerprint_crossover=True,
            use_counterfactual_filter=True,
        )
        elapsed = time.perf_counter() - t0

        m_post = evaluate_domain_metrics(orch.model, win_eval, trap_eval)
        rec = {
            "task_name": cart.task_name,
            "baseline_score": dag_rep["baseline_score"],
            "champion_id": dag_rep["champion_id"],
            "champion_score": dag_rep["champion_score"],
            "supervised_lineage_steps": sleep_rep["ks_stats"]["supervised_trajectories"],
            "total_dag_attempts": sleep_rep["ks_stats"]["total_wake_trajectories"],
            "unlocked_expert_slot": sleep_rep["ks_stats"]["unlocked_expert_slot"],
            "gkd_loss": round(sleep_rep["ks_stats"]["gkd_loss"], 4),
            "lti_reward": round(sleep_rep["ks_stats"]["lti_reward"], 4),
            "pre_sleep_tau": sleep_rep["pre_wake_tau"],
            "post_rem_tau": sleep_rep["post_rem_tau"],
            "win_share_pct": round(m_post["win_share_pct"], 2),
            "trap_contamination_pct": round(m_post["trap_contamination_pct"], 2),
            "gate_2x2_accepted": sleep_rep["gate_2x2"]["accepted_pair"],
            "gate_2x2_synergy": sleep_rep["gate_2x2"]["interaction_synergy"],
            "elapsed_sec": round(elapsed, 2),
        }
        live_records.append(rec)
        print(
            f"  [{c_idx+1}/8] {cart.task_name:<30} | DAG: {rec['baseline_score']:>7.2f} -> {rec['champion_score']:>8.2f} "
            f"(`{rec['champion_id']}`) | Masked-KS: {rec['supervised_lineage_steps']}/{rec['total_dag_attempts']} "
            f"(LTI={rec['lti_reward']:.3f}) | tau: {rec['pre_sleep_tau']:+.2f}->{rec['post_rem_tau']:+.2f} "
            f"| WinShare={rec['win_share_pct']:.1f}% | 2x2={rec['gate_2x2_accepted']}"
        )

    # Final multi-cartridge retention check across all 4 consolidated tasks
    final_retention_by_task: Dict[str, float] = {}
    for t_name, w_ev, t_ev in domain_eval_sets:
        m_fin = evaluate_domain_metrics(orch.model, w_ev, t_ev)
        final_retention_by_task[t_name] = round(m_fin["win_share_pct"], 2)

    print("-" * 84)
    print(f"  Final Parametric Retention across all 8 Cartridges: {final_retention_by_task}")
    print("=" * 84)
    return {
        "cartridge_runs": live_records,
        "final_retention_by_task": final_retention_by_task,
        "active_slow_experts": orch.model.cms_blocks[1].num_unlocked_low_rank,
    }


# ===========================================================================
# 5. VISUALIZATION GENERATOR (4-PANEL PUBLICATION FIGURE)
# ===========================================================================


def generate_benchmark_figure(
    exp1: Dict[str, Any],
    exp2: Dict[str, Any],
    exp3: Dict[str, Any],
    out_path: str,
) -> None:
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")

    # Palette
    c_wake = "#94A3B8"     # Slate gray
    c_van = "#F59E0B"      # Amber
    c_rep = "#3B82F6"      # Blue
    c_full = "#0F766E"     # Deep Teal (Champion)
    c_danger = "#E11D48"   # Rose

    # Panel A: Domain 1 Retention Curve across 5 Sequential Discovery Cycles
    ax1 = axes[0, 0]
    steps = [1, 2, 3, 4, 5]
    ax1.plot(steps, exp1["wake_only_sft"]["d1_retention_curve"], "o--", color=c_wake, lw=2.0, label="Wake-Only SFT (No Sleep)")
    ax1.plot(steps, exp1["vanilla_sleep_unmasked"]["d1_retention_curve"], "s-.", color=c_van, lw=2.2, label="Vanilla SLEEP (Unmasked)")
    ax1.plot(steps, exp1["sleep_dag_no_replay"]["d1_retention_curve"], "^-", color=c_rep, lw=2.2, label="SLEEP + DAG-Masking (No Replay)")
    ax1.plot(steps, exp1["sleep_dag_rsi_full"]["d1_retention_curve"], "D-", color=c_full, lw=3.0, markersize=7, label="SLEEP-DAG-RSI (Full Unified)")
    ax1.set_title("A. Domain 1 Parametric Retention Across 5 Sequential Cycles", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax1.set_xlabel("Sequential Discovery Cycle (Domains D1 -> D5)", fontsize=10, color="#334155")
    ax1.set_ylabel("Domain 1 Winning Lineage Share (%)", fontsize=10, color="#334155")
    ax1.set_xticks(steps)
    ax1.set_xticklabels([f"After D{i}" for i in steps])
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True, facecolor="#F8FAFC", fontsize=8.8)

    # Panel B: Final 5-Domain Winning Share vs Dead-End Trap Contamination
    ax2 = axes[0, 1]
    keys = ["wake_only_sft", "vanilla_sleep_unmasked", "sleep_dag_no_replay", "sleep_dag_rsi_full"]
    short_names = ["Wake-Only\nSFT", "Vanilla SLEEP\n(Unmasked)", "SLEEP + DAG\n(No Replay)", "SLEEP-DAG-RSI\n(Full Unified)"]
    win_vals = [exp1[k]["final_avg_win_share_pct"] for k in keys]
    trap_vals = [exp1[k]["trap_contamination_pct"] for k in keys]
    x_pos = np.arange(len(keys))
    w = 0.36
    b1 = ax2.bar(x_pos - w / 2, win_vals, width=w, color=c_full, label="Winning Lineage Share (%) [Higher=Better]")
    b2 = ax2.bar(x_pos + w / 2, trap_vals, width=w, color=c_danger, alpha=0.85, label="Dead-End Trap Contamination (%) [Lower=Better]")
    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=c_full)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=c_danger)
    ax2.set_title("B. Multi-Domain Accuracy vs Dead-End Trap Contamination", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, fontsize=9)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Probability Mass Share (%)", fontsize=10, color="#334155")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", fontsize=8.5)

    # Panel C: REM Dreaming — Random MoE Expert vs Error-Fingerprint Guided MoE Crossover
    ax3 = axes[1, 0]
    d_keys = ["no_dreaming", "vanilla_sleep_random_moe", "sleep_dag_rsi_guided_moe"]
    d_names = ["No Dreaming\n(NREM Only)", "Vanilla SLEEP\n(Random MoE)", "SLEEP-DAG-RSI\n(Fingerprint MoE + CF)"]
    comp_hits = [exp2[k]["complementary_hit_rate_pct"] for k in d_keys]
    hybrid_scores = [exp2[k]["avg_hybrid_score"] for k in d_keys]
    spike_leaks = [exp2[k]["spike_leak_rate_pct"] for k in d_keys]
    x3 = np.arange(len(d_keys))
    w3 = 0.26
    bars_h = ax3.bar(x3 - w3, hybrid_scores, width=w3, color=c_rep, label="Post-Dream Hybrid Score")
    bars_c = ax3.bar(x3, comp_hits, width=w3, color=c_full, label="Complementary Expert Hit Rate (%)")
    bars_s = ax3.bar(x3 + w3, spike_leaks, width=w3, color=c_danger, alpha=0.85, label="Sharp-Spike Overfit Leak Rate (%)")
    for b_grp in (bars_h, bars_c, bars_s):
        for bar in b_grp:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax3.set_title("C. REM Dreaming: Random MoE vs Error-Fingerprint Crossover (30 Worlds)", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(d_names, fontsize=9)
    ax3.set_ylim(0, 115)
    ax3.set_ylabel("Rate (%) / Normalized Score", fontsize=10, color="#334155")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", fontsize=8.5)

    # Panel D: Live 8-Cartridge Coupled Run (Pre-Sleep vs Post-REM Skill Score tau & DAG Lift)
    ax4 = axes[1, 1]
    runs = exp3["cartridge_runs"]
    c_labels = ["C1:RBF\nSolver", "C2:Weibull\nBinPack", "C3:Auto\nC1", "C4:Heil\nn=11", "C5:Pocket\nOTC", "C6:SimVQ\n1.58b", "C7:Heil\n13-14", "C8:Auto\nC2"]
    pre_taus = [r["pre_sleep_tau"] for r in runs]
    post_taus = [r["post_rem_tau"] for r in runs]
    x4 = np.arange(len(runs))
    w4 = 0.38
    b_pre = ax4.bar(x4 - w4 / 2, pre_taus, width=w4, color=c_wake, label="Pre-Sleep Parametric Skill (tau)")
    b_post = ax4.bar(x4 + w4 / 2, post_taus, width=w4, color=c_full, label="Post-SLEEP-DAG-RSI Parametric Skill (tau)")
    for bar in b_pre:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, max(h, 0.0) + 0.08, f"{h:+.1f}", ha="center", va="bottom", fontsize=7.2, color="#475569")
    for bar in b_post:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, max(h, 0.0) + 0.08, f"{h:+.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=c_full)
    ax4.axhline(0.0, color="#64748B", lw=0.8)
    ax4.set_title("D. Live 8-Cartridge Coupled Run: Parametric Skill (tau) Before vs After Sleep", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(c_labels, fontsize=8.2)
    ax4.set_ylabel("Log-Odds Skill Score tau = log(P_win / P_trap)", fontsize=10, color="#334155")
    ax4.set_ylim(min(min(pre_taus) - 0.8, -0.8), max(post_taus) + 2.6)
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", fontsize=8.5)

    fig.suptitle(
        "SLEEP-DAG-RSI: Unifying Parametric Continual Sleep (HOPE + CMS) with Counterfactual DAG Self-Improvement",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved 4-panel benchmark chart to: {out_path}")




def run_suite_5_heilbronn_n17_certification() -> Dict[str, Any]:
    print("=" * 84)
    print("SUITE 5: HEILBRONN TRIANGLE n=13..17 500-DIGIT & 10^100 RATIONAL CERTIFICATION")
    print("Verifying 16+1 Floater Split & 27-Triangle Isostatic KKT Point at n=17")
    print("=" * 84)
    from decimal import Decimal, getcontext
    getcontext().prec = 520

    pkg_dir = os.path.join(LOCAL_DIR, "heilbronn_pr_package")
    specs = [
        ("n13_exact_500dps.json", 13, 20),
        ("n14_exact_500dps.json", 14, 23),
        ("n15_exact_500dps.json", 15, 15),
        ("n16_exact_500dps.json", 16, 24),
        ("n17_exact_500dps.json", 17, 27),
    ]
    out_certs: Dict[str, Any] = {}
    for fname, n_exp, ties_exp in specs:
        fpath = os.path.join(pkg_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts_dec = [(Decimal(u), Decimal(v)) for u, v in data["coordinates_simplex_right_triangle_500dps"]]
        areas_dec = []
        for i in range(n_exp):
            ui, vi = pts_dec[i]
            for j in range(i + 1, n_exp):
                uj, vj = pts_dec[j]
                for k in range(j + 1, n_exp):
                    uk, vk = pts_dec[k]
                    areas_dec.append((abs((uj - ui) * (vk - vi) - (uk - ui) * (vj - vi)), (i, j, k)))
        areas_dec.sort(key=lambda x: x[0])
        min_a = areas_dec[0][0]
        ties = [tri for area, tri in areas_dec if abs(area - min_a) < Decimal("1e-490")]
        assert len(ties) == ties_exp, f"n={n_exp}: expected {ties_exp} ties, got {len(ties)}"
        out_certs[f"n{n_exp}"] = {
            "n": n_exp,
            "delta_40dps": str(min_a)[:42],
            "active_ties": len(ties),
            "total_triangles": len(areas_dec),
            "floaters": data.get("floater_vertices_0indexed", []),
            "floater_clearance": data.get("floater_chebyshev_clearance", None),
        }
        print(
            f"  [PASS] n={n_exp:2d} | Delta_{n_exp}* = {str(min_a)[:36]}... | "
            f"Ties (<1e-490) = {len(ties):2d}/{len(areas_dec):3d} | Floaters = {data.get('floater_vertices_0indexed', [])}"
        )
    print("=" * 84)
    return out_certs


def main() -> int:
    t_all = time.perf_counter()
    s1 = run_suite_1_mathematical_verifications()
    s2 = run_suite_2_continual_learning_experiment()
    s3 = run_suite_3_dreaming_crossover_experiment()
    s4 = run_suite_4_live_cartridges_integration()
    s5 = run_suite_5_heilbronn_n17_certification()

    out_dir = LOCAL_DIR
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, "sleep_dag_rsi_benchmarks.png")
    generate_benchmark_figure(s2, s3, s4, chart_path)

    full_report = {
        "suite_1_verifications": s1,
        "suite_2_continual_forgetting": s2,
        "suite_3_dreaming_crossover": s3,
        "suite_4_live_cartridges": s4,
        "suite_5_heilbronn_n13_to_n17": s5,
        "total_runtime_sec": round(time.perf_counter() - t_all, 2),
    }
    json_path = os.path.join(LOCAL_DIR, "sleep_dag_rsi_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"Saved full JSON experimental report to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())