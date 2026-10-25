"""Masking diagnostics M1–M5 and masking gap computation.

All diagnostics are computed from pre-run attack result dictionaries and
written to a summary table. No re-running of attacks here.

Diagnostics:
  M1: PGD (multi-step) robust_acc <= single-step robust_acc + tolerance.
  M2: At eps = 5*d_med, robust_acc approaches 0.
  M3: Black-box success should not greatly exceed white-box success.
  M4: Log distribution of ||grad f||; flag if median near zero.
  M5: Restarts help — with-restart robust_acc <= no-restart robust_acc + tol.
  Masking gap: robust_acc(weakest_attack) - robust_acc(strongest_attack).
"""
import numpy as np
from typing import Optional


def check_m1_iterative_stronger(
    robust_acc_multistep: float,
    robust_acc_singlestep: float,
    tolerance: float = 0.02,
) -> dict:
    """M1: Multi-step attack should be at least as strong as single-step."""
    passed = robust_acc_multistep <= robust_acc_singlestep + tolerance
    return {
        "diagnostic": "M1",
        "passed": passed,
        "robust_acc_multistep": robust_acc_multistep,
        "robust_acc_singlestep": robust_acc_singlestep,
        "flag": None if passed else "MASKED_GRADIENT: single-step >= multi-step",
    }


def check_m2_large_eps_crushes(
    robust_acc_large_eps: float,
    threshold: float = 0.1,
) -> dict:
    """M2: At very large eps (5*d_med), robust_acc should approach 0."""
    passed = robust_acc_large_eps <= threshold
    return {
        "diagnostic": "M2",
        "passed": passed,
        "robust_acc_large_eps": robust_acc_large_eps,
        "flag": None if passed else f"BROKEN_ATTACK: robust_acc={robust_acc_large_eps:.3f} at large eps",
    }


def check_m3_blackbox_not_better(
    robust_acc_whitebox: float,
    robust_acc_blackbox: float,
    margin: float = 0.10,
) -> dict:
    """M3: Black-box success should not greatly exceed white-box success."""
    # Attack success rate = 1 - robust_acc
    asr_wb = 1.0 - robust_acc_whitebox
    asr_bb = 1.0 - robust_acc_blackbox
    # Flag if bb significantly exceeds wb (gradient masking)
    passed = asr_bb <= asr_wb + margin
    return {
        "diagnostic": "M3",
        "passed": passed,
        "asr_whitebox": asr_wb,
        "asr_blackbox": asr_bb,
        "flag": None if passed else f"MASKED_GRADIENT: bb_asr={asr_bb:.3f} >> wb_asr={asr_wb:.3f}",
    }


def check_m4_grad_norms(
    grad_norms: np.ndarray,
    zero_threshold: float = 1e-6,
) -> dict:
    """M4: Log distribution of ||grad f||; flag if median is near zero."""
    if len(grad_norms) == 0:
        return {"diagnostic": "M4", "passed": False, "flag": "No gradient norms provided"}
    median_norm = float(np.median(grad_norms))
    pct_near_zero = float(np.mean(grad_norms < zero_threshold))
    passed = median_norm > zero_threshold
    return {
        "diagnostic": "M4",
        "passed": passed,
        "median_grad_norm": median_norm,
        "pct_near_zero": pct_near_zero,
        "p10": float(np.percentile(grad_norms, 10)),
        "p50": float(np.percentile(grad_norms, 50)),
        "p90": float(np.percentile(grad_norms, 90)),
        "flag": None if passed else f"VANISHING_GRADIENTS: median_norm={median_norm:.2e}",
    }


def check_m5_restarts_help(
    robust_acc_with_restarts: float,
    robust_acc_no_restarts: float,
    tolerance: float = 0.02,
) -> dict:
    """M5: Multiple restarts should not yield worse results than single restart."""
    passed = robust_acc_with_restarts <= robust_acc_no_restarts + tolerance
    return {
        "diagnostic": "M5",
        "passed": passed,
        "robust_acc_with_restarts": robust_acc_with_restarts,
        "robust_acc_no_restarts": robust_acc_no_restarts,
        "flag": None if passed else "RESTART_ANOMALY: restarts made attack weaker",
    }


def compute_masking_gap(
    robust_acc_weakest: float,
    robust_acc_strongest: float,
) -> dict:
    """Masking gap = robust_acc(weakest) - robust_acc(strongest)."""
    gap = robust_acc_weakest - robust_acc_strongest
    return {
        "masking_gap": gap,
        "robust_acc_weakest": robust_acc_weakest,
        "robust_acc_strongest": robust_acc_strongest,
    }


def run_masking_diagnostics(
    robust_acc_a1: float,
    robust_acc_a2: float,
    robust_acc_a2_singlestep: float,
    robust_acc_a2_no_restarts: float,
    robust_acc_large_eps: float,
    grad_norms: np.ndarray,
    robust_acc_blackbox: Optional[float] = None,
) -> list[dict]:
    """Run all masking diagnostics and return a list of result dicts."""
    results = []
    results.append(check_m1_iterative_stronger(robust_acc_a2, robust_acc_a2_singlestep))
    results.append(check_m2_large_eps_crushes(robust_acc_large_eps))
    if robust_acc_blackbox is not None:
        results.append(check_m3_blackbox_not_better(robust_acc_a2, robust_acc_blackbox))
    results.append(check_m4_grad_norms(grad_norms))
    results.append(check_m5_restarts_help(robust_acc_a2, robust_acc_a2_no_restarts))
    results.append(compute_masking_gap(robust_acc_a1, robust_acc_a2))
    return results
