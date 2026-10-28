"""
Backward-compatible shim for `simulator.py` that delegates to the canonical
`engine.py` module so there is only a single source of truth in `dream-rsi`.
"""

from __future__ import annotations

from engine import (  # noqa: F401
    BranchTrajectory,
    DiscoveryTree,
    DynamicPortfolio as DynamicPortfolioPolicy,
    GridPlan,
    GridPlanningContext,
    Node,
    OptimalPolicy,
    Policy as BaseExplorationPolicy,
    Problem,
    World as ReplayWorld,
    pareto_sweep,
    reconstruct_trajectories,
    replay as run_policy_replay,
)
