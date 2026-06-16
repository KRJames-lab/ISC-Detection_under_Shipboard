"""
Test set evaluation — Window-level metrics + Run-level detection delay.
Supports supervised models (logit output).
"""
import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from ai.config import BATCH_SIZE, T_ONSET, DEADLINE_FACTOR, SCENARIO_NAMES


def evaluate_supervised(model, test_ds, model_name: str, device: str = "cpu"):
    """Evaluate a supervised model on test set with full confusion matrix."""
    model.to(device)
    model.eval()
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    labels = np.array(all_labels, dtype=int)
    preds = np.array(all_preds, dtype=int)
    metrics = _compute_metrics(labels, preds)
    _print_report(model_name, metrics)
    return metrics


def _compute_metrics(labels, preds):
    """Compute full confusion matrix and derived metrics."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy, "far": far,
    }


def _print_report(model_name, metrics):
    """Print formatted evaluation report with confusion matrix."""
    print(f"\n  [{model_name}] Test Results")
    print(f"  {'─' * 40}")

    # Confusion matrix
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    print(f"  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Normal   ISC")
    print(f"  Actual Normal   {tn:5d}   {fp:5d}")
    print(f"  Actual ISC      {fn:5d}   {tp:5d}")
    print()
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  FAR:        {metrics['far']:.4f}")

    if "threshold" in metrics:
        print(f"  Threshold:  {metrics['threshold']:.4f}")
        print(f"  Mean Error (Normal): {metrics['mean_error_normal']:.4f}")
        print(f"  Mean Error (ISC):    {metrics['mean_error_isc']:.4f}")


# ──────────────────────────────────────────────────────────────
# Run-level Detection Delay
# ──────────────────────────────────────────────────────────────

def evaluate_detection_delay(model, test_sets, model_name, device="cpu"):
    """
    Run-level detection delay evaluation.

    For each test run (WindowSet):
      1. Predict all windows in temporal order
      2. Find first ISC prediction after T_ONSET
      3. Calculate detection delay vs deadline (R_ISC ≤ 10Ω)

    Returns list of per-run result dicts.
    """
    model.to(device)
    model.eval()

    results = []
    for ws in test_sets:
        # Predict all windows in temporal order
        with torch.no_grad():
            x = torch.FloatTensor(ws.windows).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

        t_ends = ws.t_ends
        labels = ws.labels

        # Label onset: first window labeled ISC=1
        isc_mask = labels == 1
        t_label_onset = float(t_ends[isc_mask][0]) if isc_mask.any() else None

        # First detection: first prediction=1 after T_ONSET
        post_onset = t_ends > T_ONSET
        post_preds = preds[post_onset]
        post_times = t_ends[post_onset]

        detect_idx = np.where(post_preds == 1)[0]
        if len(detect_idx) > 0:
            t_detect = float(post_times[detect_idx[0]])
            delay_from_onset = t_detect - T_ONSET
        else:
            t_detect = None
            delay_from_onset = None

        # Deadline
        deadline = DEADLINE_FACTOR * ws.tau
        t_deadline = T_ONSET + deadline

        if delay_from_onset is not None:
            passed = delay_from_onset <= deadline
            margin = deadline - delay_from_onset
            # Delay relative to label onset (reaction time after signal emergence)
            delay_from_label = (t_detect - t_label_onset) if t_label_onset else None
        else:
            passed = False
            margin = None
            delay_from_label = None

        # False alarms before ISC onset
        pre_onset = t_ends <= T_ONSET
        n_pre_fa = int(preds[pre_onset].sum()) if pre_onset.any() else 0

        results.append({
            "run_id": ws.run_id,
            "scenario": ws.scenario,
            "scenario_name": SCENARIO_NAMES.get(ws.scenario, str(ws.scenario)),
            "k_R_mOhm": ws.k_R_mOhm,
            "tau": ws.tau,
            "t_label_onset": t_label_onset,
            "t_detect": t_detect,
            "delay_from_onset": delay_from_onset,
            "delay_from_label": delay_from_label,
            "deadline": deadline,
            "t_deadline": t_deadline,
            "passed": passed,
            "margin": margin,
            "n_pre_fa": n_pre_fa,
        })

    _print_delay_report(model_name, results)
    return results


def _print_delay_report(model_name, results):
    """Print formatted detection delay report."""
    if not results:
        return

    tau = results[0]["tau"]
    deadline = results[0]["deadline"]
    t_deadline = results[0]["t_deadline"]
    t_label = results[0]["t_label_onset"]

    print(f"\n  [{model_name}] Detection Delay Analysis")
    print(f"  {'=' * 80}")
    print(f"  Test: {len(results)} runs, tau={tau}s")
    print(f"  Deadline: {deadline:.0f}s after onset  (R_ISC <= 10 Ohm at t={t_deadline:.0f}s)")
    if t_label:
        label_delay = t_label - T_ONSET
        print(f"  Signal emergence: {label_delay:.0f}s after onset  "
              f"(R_ISC <= 50 Ohm at t={t_label:.0f}s)")
        print(f"  Available window: {deadline - label_delay:.0f}s "
              f"(from signal emergence to deadline)")

    # Per-run table
    print(f"\n  {'Run':<24} {'k_R':>5} {'Scenario':<9} "
          f"{'t_detect':>8} {'Delay':>8} {'Reaction':>8} {'Margin':>8} {'Result':>6}")
    print(f"  {'─' * 80}")

    sorted_results = sorted(results, key=lambda r: (r["k_R_mOhm"], r["scenario"]))
    for r in sorted_results:
        td = f"{r['t_detect']:.0f}" if r["t_detect"] else "  N/A"
        dl = f"{r['delay_from_onset']:.0f}" if r["delay_from_onset"] is not None else "  N/A"
        rx = f"{r['delay_from_label']:.0f}" if r["delay_from_label"] is not None else "  N/A"
        mg = f"{r['margin']:.0f}" if r["margin"] is not None else "  N/A"
        rs = "PASS" if r["passed"] else "FAIL"
        fa = f" FA:{r['n_pre_fa']}" if r["n_pre_fa"] > 0 else ""
        print(f"  {r['run_id']:<24} {r['k_R_mOhm']:>5.2f} {r['scenario_name']:<9} "
              f"{td:>8}s {dl:>7}s {rx:>7}s {mg:>7}s {rs:>6}{fa}")

    # Summary
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    detected = [r for r in results if r["delay_from_onset"] is not None]

    print(f"\n  Summary")
    print(f"  {'─' * 40}")
    print(f"  PASS: {n_pass}/{n_total} ({n_pass/n_total*100:.0f}%)")

    if detected:
        delays = [r["delay_from_onset"] for r in detected]
        reactions = [r["delay_from_label"] for r in detected
                     if r["delay_from_label"] is not None]
        margins = [r["margin"] for r in detected]

        print(f"  Detected: {len(detected)}/{n_total}")
        print(f"  Avg delay from onset: {np.mean(delays):.0f}s")
        if reactions:
            print(f"  Avg reaction time: {np.mean(reactions):.0f}s  "
                  f"(from signal emergence)")
        print(f"  Avg margin: {np.mean(margins):.0f}s")

        best = min(detected, key=lambda r: r["delay_from_onset"])
        worst = max(detected, key=lambda r: r["delay_from_onset"])
        print(f"  Fastest: {best['delay_from_onset']:.0f}s "
              f"(k_R={best['k_R_mOhm']}, {best['scenario_name']})")
        print(f"  Slowest: {worst['delay_from_onset']:.0f}s "
              f"(k_R={worst['k_R_mOhm']}, {worst['scenario_name']})")

    # By k_R
    print(f"\n  By k_R (mOhm/g)")
    print(f"  {'k_R':>6}  {'PASS':>8}  {'Avg Delay':>10}  {'Avg Reaction':>12}  {'Avg Margin':>10}")
    print(f"  {'─' * 55}")
    for kr in sorted(set(r["k_R_mOhm"] for r in results)):
        sub = [r for r in results if r["k_R_mOhm"] == kr]
        n_p = sum(1 for r in sub if r["passed"])
        det = [r for r in sub if r["delay_from_onset"] is not None]
        avg_d = np.mean([r["delay_from_onset"] for r in det]) if det else float("nan")
        avg_r = np.mean([r["delay_from_label"] for r in det
                         if r["delay_from_label"] is not None]) if det else float("nan")
        avg_m = np.mean([r["margin"] for r in det]) if det else float("nan")
        print(f"  {kr:>6.2f}  {n_p}/{len(sub):>5}  {avg_d:>9.0f}s  {avg_r:>11.0f}s  {avg_m:>9.0f}s")

    # By scenario
    print(f"\n  By Scenario")
    print(f"  {'Scenario':<10}  {'PASS':>8}  {'Avg Delay':>10}  {'Avg Reaction':>12}  {'Avg Margin':>10}")
    print(f"  {'─' * 55}")
    for sc in sorted(set(r["scenario"] for r in results)):
        sub = [r for r in results if r["scenario"] == sc]
        n_p = sum(1 for r in sub if r["passed"])
        det = [r for r in sub if r["delay_from_onset"] is not None]
        avg_d = np.mean([r["delay_from_onset"] for r in det]) if det else float("nan")
        avg_r = np.mean([r["delay_from_label"] for r in det
                         if r["delay_from_label"] is not None]) if det else float("nan")
        avg_m = np.mean([r["margin"] for r in det]) if det else float("nan")
        nm = SCENARIO_NAMES.get(sc, str(sc))
        print(f"  {nm:<10}  {n_p}/{len(sub):>5}  {avg_d:>9.0f}s  {avg_r:>11.0f}s  {avg_m:>9.0f}s")
