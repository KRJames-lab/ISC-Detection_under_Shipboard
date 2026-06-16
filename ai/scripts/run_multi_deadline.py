"""
Multi-R_deadline evaluation: EKF / ModernTCN / LITE 의 PASS/FAIL·margin 을
R_deadline in {10, 15, 25, 50, 100} Ohm 에 대해 재평가.

동일 ISC 궤적 R_ISC(t) = 5 + 495 exp(-(t - t_onset)/tau) 위에서
각 R_deadline 은 서로 다른 cutoff 시각 t_deadline = t_onset + k(R) * tau 만 준다.
  k(R) = ln(495 / (R - 5))
  R=10 -> 4.595   R=15 -> 3.902   R=25 -> 3.209   R=50 -> 2.398   R=100 -> 1.651

재평가 원칙:
  * Simulink 재실행 없음, 모델 재학습 없음
  * detection_time (delay_from_onset) 은 ISC 궤적에 대한 방법의 반응 시점이므로
    deadline 정의와 무관 -> 그대로 사용
  * 각 run 을 각 R_deadline 기준으로 다시 PASS/FAIL 판정, margin 재계산

실행: python -m ai.scripts.run_multi_deadline
"""
import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from ai.config import (
    RANDOM_SEED, RESULTS_DIR, RESAMPLE_HZ,
    EVAL_TAUS, TEST_TAU, T_ONSET,
    R_INIT, R_FINAL,
    SCENARIOS, SCENARIO_NAMES, ABLATION_V_ONLY, ABLATION_NOVIB_TRAIN,
)
from ai.data.loader import load_all_runs, load_eval_runs, SimRun
from ai.data.preprocess import preprocess_all, resample_run
from ai.data.windowing import extract_all_windows
from ai.data.splits import split_window_sets
from ai.models.modern_tcn import ModernTCN
from ai.models.lite import LITE
from ai.models.npu_conv2d import NPUConv2D
from ai.training.evaluate import evaluate_detection_delay
from ai.training.evaluate_ekf import calibrate_ekf, evaluate_ekf_detection_delay


R_DEADLINES = [10, 15, 25, 50]
# NPU-Conv2D excluded in V-only ablation (kept as V+T baseline; checkpoint shape mismatch under 1-channel input)
METHODS = ["EKF", "ModernTCN", "LITE"] if ABLATION_V_ONLY else ["EKF", "ModernTCN", "LITE", "NPU-Conv2D"]

def k_d_factor(R_deadline):
    """deadline 시간계수 k_d: t_deadline - t_onset = k_d(R) * tau"""
    return math.log((R_INIT - R_FINAL) / (R_deadline - R_FINAL))


def resample_runs_no_normalize(runs):
    return [resample_run(r) for r in runs]


def _rescore(results, R_deadline):
    """주어진 results 리스트의 각 run 을 R_deadline 으로 재판정."""
    k = k_d_factor(R_deadline)
    rescored = []
    for r in results:
        tau = r["tau"]
        deadline_new = k * tau
        delay = r["delay_from_onset"]
        if delay is not None:
            passed = delay <= deadline_new
            margin = deadline_new - delay
        else:
            passed = False
            margin = None
        rescored.append({
            **r,
            "R_deadline": R_deadline,
            "deadline": deadline_new,
            "passed": passed,
            "margin": margin,
        })
    return rescored


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0 / RESAMPLE_HZ

    print(f"Device: {device}")
    print(f"Sampling: {RESAMPLE_HZ}Hz, dt={dt}s")
    print(f"R_deadlines: {R_DEADLINES} Ohm")
    print(f"k factors: {[round(k_d_factor(R), 3) for R in R_DEADLINES]}")

    # ── 1. Load data ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Loading data...")

    all_runs_raw = load_all_runs()
    all_runs_ai = preprocess_all([SimRun(**r.__dict__) for r in all_runs_raw])
    window_sets = extract_all_windows(all_runs_ai)
    _, _, test_sets_1800 = split_window_sets(window_sets)

    all_runs_ekf = resample_runs_no_normalize(all_runs_raw)
    ekf_runs_1800 = [r for r in all_runs_ekf if r.tau == 1800]

    eval_runs_raw = load_eval_runs()
    eval_runs_ai = preprocess_all([SimRun(**r.__dict__) for r in eval_runs_raw])
    eval_window_sets = extract_all_windows(eval_runs_ai)
    eval_runs_ekf = resample_runs_no_normalize(eval_runs_raw)

    ai_sets_by_tau = {}
    for tau in EVAL_TAUS:
        ai_sets_by_tau[tau] = [ws for ws in eval_window_sets if ws.tau == tau]
    ai_sets_by_tau[TEST_TAU] = test_sets_1800

    ekf_runs_by_tau = {}
    for tau in EVAL_TAUS:
        ekf_runs_by_tau[tau] = [r for r in eval_runs_ekf if r.tau == tau]
    ekf_runs_by_tau[TEST_TAU] = ekf_runs_1800

    # ── 2. Load AI models ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Loading AI models...")
    suffix = ""
    if ABLATION_V_ONLY:
        suffix += "_v_only"
    if ABLATION_NOVIB_TRAIN:
        suffix += "_novib_train"
    tcn = ModernTCN()
    tcn.load_state_dict(torch.load(
        RESULTS_DIR / f"moderntcn{suffix}_best.pt", map_location=device, weights_only=True))
    print(f"  ModernTCN: {tcn.count_params()} params")

    lite = LITE()
    lite.load_state_dict(torch.load(
        RESULTS_DIR / f"lite{suffix}_best.pt", map_location=device, weights_only=True))
    print(f"  LITE: {lite.count_params()} params")

    npu = None
    if not ABLATION_V_ONLY:
        npu = NPUConv2D()
        npu.load_state_dict(torch.load(
            RESULTS_DIR / "npu_conv2d_best.pt", map_location=device, weights_only=True))
        print(f"  NPU-Conv2D: {npu.count_params()} params")

    # ── 3. EKF calibration (per-scenario) ───────────────────────────
    print("\n" + "=" * 80)
    print("EKF Calibration (per-scenario)")
    all_ekf_runs = all_runs_ekf + eval_runs_ekf
    ekf_calibs = {}
    for sc in SCENARIOS:
        sc_runs = [r for r in all_ekf_runs if r.scenario == sc]
        print(f"\n  --- {SCENARIO_NAMES[sc]} ({len(sc_runs)} runs) ---")
        ekf_calibs[sc] = calibrate_ekf(sc_runs, dt)

    # ── 4. Get detection_time once (baseline R=10) ──────────────────
    print("\n" + "=" * 80)
    print("Running detection once per method x tau ...")
    raw_results = {}   # raw_results[tau][method] = [run_dicts]
    eval_taus = sorted(ai_sets_by_tau.keys())

    for tau in eval_taus:
        print(f"\n  tau={tau}s")
        raw_results[tau] = {}

        ekf_runs = ekf_runs_by_tau[tau]
        ekf_all = []
        for sc in SCENARIOS:
            sc_runs = [r for r in ekf_runs if r.scenario == sc]
            if sc_runs:
                ekf_all.extend(evaluate_ekf_detection_delay(
                    sc_runs, ekf_calibs[sc], dt))
        raw_results[tau]["EKF"] = ekf_all

        ai_sets = ai_sets_by_tau[tau]
        raw_results[tau]["ModernTCN"] = evaluate_detection_delay(
            tcn, ai_sets, "ModernTCN", device)
        raw_results[tau]["LITE"] = evaluate_detection_delay(
            lite, ai_sets, "LITE", device)
        if npu is not None:
            raw_results[tau]["NPU-Conv2D"] = evaluate_detection_delay(
                npu, ai_sets, "NPU-Conv2D", device)

    # ── 5. Rescore across R_deadlines ───────────────────────────────
    print("\n" + "=" * 80)
    print("Rescoring across R_deadlines")
    rescored_all = {}   # [R][tau][method] = runs
    for R in R_DEADLINES:
        rescored_all[R] = {}
        for tau in eval_taus:
            rescored_all[R][tau] = {}
            for method in METHODS:
                rescored_all[R][tau][method] = _rescore(
                    raw_results[tau][method], R)

    # ── 6. Print summary tables ─────────────────────────────────────
    _print_master_pass_rate(rescored_all, eval_taus)
    _print_margin_table(rescored_all, eval_taus)
    _print_per_method_matrix(rescored_all, eval_taus)
    _print_fail_cases(rescored_all, eval_taus)
    _print_vibration_impact(rescored_all, eval_taus)
    _print_vibration_fa(raw_results, eval_taus)

    # ── 7. Save JSON ────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent.parent / "data"
    out_path = out_dir / f"multi_deadline_results{suffix}.json"

    payload = {
        "config": {
            "R_init": R_INIT, "R_final": R_FINAL, "t_onset": T_ONSET,
            "R_deadlines": R_DEADLINES,
            "k_d_factors": {R: k_d_factor(R) for R in R_DEADLINES},
            "taus": eval_taus,
            "methods": METHODS,
        },
        "results": {},
    }
    for R in R_DEADLINES:
        payload["results"][str(R)] = {}
        for tau in eval_taus:
            payload["results"][str(R)][str(tau)] = {}
            for method in METHODS:
                runs = rescored_all[R][tau][method]
                payload["results"][str(R)][str(tau)][method] = [
                    {
                        "run_id": r["run_id"],
                        "scenario_name": r["scenario_name"],
                        "k_R_mOhm": r["k_R_mOhm"],
                        "tau": r["tau"],
                        "delay_from_onset": r["delay_from_onset"],
                        "deadline": r["deadline"],
                        "passed": r["passed"],
                        "margin": r["margin"],
                        "n_pre_fa": r["n_pre_fa"],
                    } for r in runs
                ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


def _print_master_pass_rate(rescored_all, eval_taus):
    print(f"\n{'=' * 80}")
    print("TABLE 1. PASS rate (n_pass / n_total)  -  rows: R_deadline, cols: tau x method")
    print("=" * 80)
    header = f"  {'R (Ohm)':>8}  {'k_d':>6}"
    for tau in eval_taus:
        for m in METHODS:
            header += f"  {f'tau={tau}|{m[:4]}':>12}"
    header += f"  {'ALL':>14}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for R in R_DEADLINES:
        k_d = k_d_factor(R)
        line = f"  {R:>8d}  {k_d:>6.3f}"
        total_p = 0; total_n = 0
        for tau in eval_taus:
            for m in METHODS:
                runs = rescored_all[R][tau][m]
                n_p = sum(1 for r in runs if r["passed"])
                n_t = len(runs)
                total_p += n_p; total_n += n_t
                line += f"  {n_p:>3}/{n_t:<3}({n_p/n_t*100:>3.0f}%)"
        pct = total_p / total_n * 100 if total_n else 0
        line += f"  {total_p:>4}/{total_n:<4}({pct:>3.0f}%)"
        print(line)


def _print_margin_table(rescored_all, eval_taus):
    print(f"\n{'=' * 80}")
    print("TABLE 2. Avg margin (deadline - delay, sec)  -  negative -> FAIL")
    print("=" * 80)
    header = f"  {'R (Ohm)':>8}"
    for tau in eval_taus:
        for m in METHODS:
            header += f"  {f'tau={tau}|{m[:4]}':>14}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for R in R_DEADLINES:
        line = f"  {R:>8d}"
        for tau in eval_taus:
            for m in METHODS:
                runs = rescored_all[R][tau][m]
                det = [r for r in runs if r["margin"] is not None]
                if det:
                    avg = np.mean([r["margin"] for r in det])
                    line += f"  {avg:>+14.1f}"
                else:
                    line += f"  {'N/A':>14}"
        print(line)


def _print_per_method_matrix(rescored_all, eval_taus):
    """각 방법마다 R_deadline x tau PASS rate 매트릭스."""
    print(f"\n{'=' * 80}")
    print("TABLE 3. Per-method PASS rate matrix (R_deadline rows x tau cols)")
    print("=" * 80)
    for m in METHODS:
        print(f"\n  [{m}]")
        header = f"  {'R (Ohm)':>8}  {'k_d':>6}"
        for tau in eval_taus:
            header += f"  {f'tau={tau}':>12}"
        header += f"  {'ALL':>10}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for R in R_DEADLINES:
            k_d = k_d_factor(R)
            line = f"  {R:>8d}  {k_d:>6.3f}"
            t_p = 0; t_n = 0
            for tau in eval_taus:
                runs = rescored_all[R][tau][m]
                n_p = sum(1 for r in runs if r["passed"])
                n_t = len(runs)
                t_p += n_p; t_n += n_t
                line += f"  {n_p:>3}/{n_t:<3}({n_p/n_t*100:>3.0f}%)"
            pct = t_p / t_n * 100 if t_n else 0
            line += f"  {t_p:>3}/{t_n:<3}({pct:>3.0f}%)"
            print(line)


def _print_fail_cases(rescored_all, eval_taus):
    """R 을 조여갈 때 어느 R 까지 FAIL 인지 run 별 기록."""
    print(f"\n{'=' * 80}")
    print("TABLE 4. Toughest R at which each run still fails (last_fail_R)")
    print("=" * 80)
    print(f"  (R 를 100 -> 10 조이는 순서에서 마지막까지 FAIL 인 R.)")
    fail_log = {}
    for m in METHODS:
        fail_log[m] = []
        for tau in eval_taus:
            runs_by_R = {R: rescored_all[R][tau][m] for R in R_DEADLINES}
            n_runs = len(runs_by_R[R_DEADLINES[0]])
            for i in range(n_runs):
                last_fail_R = None
                fail_run = None
                for R in sorted(R_DEADLINES, reverse=True):  # 100 -> 10
                    r = runs_by_R[R][i]
                    if not r["passed"]:
                        last_fail_R = R
                        fail_run = r
                if last_fail_R is not None:
                    fail_log[m].append({
                        "tau": tau,
                        "scenario": fail_run["scenario_name"],
                        "k_R": fail_run["k_R_mOhm"],
                        "delay": fail_run["delay_from_onset"],
                        "last_fail_R": last_fail_R,
                    })
    for m in METHODS:
        print(f"\n  [{m}]  total_fails_any_R = {len(fail_log[m])}")
        if not fail_log[m]:
            print(f"    all PASS across R in {R_DEADLINES}")
            continue
        print(f"    {'tau':>5}  {'scenario':<9}  {'k_R':>5}  {'delay(s)':>9}  {'last_fail_R':>12}")
        for entry in sorted(fail_log[m], key=lambda x: (-x["last_fail_R"], x["tau"])):
            d = entry["delay"] if entry["delay"] is not None else -1
            print(f"    {entry['tau']:>5}  {entry['scenario']:<9}  "
                  f"{entry['k_R']:>5.2f}  {d:>9.1f}  {entry['last_fail_R']:>12}")


def _print_vibration_impact(rescored_all, eval_taus):
    """TABLE 5. 진동이 감지 margin 에 미치는 영향, 시나리오별 분리."""
    print(f"\n{'=' * 80}")
    print("TABLE 5. Vibration impact  -  Avg margin (s) by scenario")
    print(f"  Positive margin = PASS reserve. Penalty = NoVib_margin - Vib_margin (s).")
    print(f"  Larger penalty => method is more disturbed by vibration.")
    print("=" * 80)

    scenarios = ["NoVib", "MIL-STD", "MSS-Head"]

    for R in R_DEADLINES:
        print(f"\n  --- R_deadline = {R} Ohm  (k_d={k_d_factor(R):.3f}) ---")
        header = (f"  {'method':<12}  {'tau':>5}  "
                  f"{'NoVib':>9}  {'MIL-STD':>9}  {'MSS-Head':>9}  "
                  f"{'PenMIL':>8}  {'PenMSS':>8}")
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for m in METHODS:
            for tau in eval_taus:
                runs = rescored_all[R][tau][m]
                avgs = {}
                for sc in scenarios:
                    sub = [r for r in runs
                           if r["scenario_name"] == sc and r["margin"] is not None]
                    avgs[sc] = float(np.mean([r["margin"] for r in sub])) if sub else None

                def _fmt(v):
                    return f"{v:>+9.1f}" if v is not None else "      N/A"

                no = avgs["NoVib"]; mil = avgs["MIL-STD"]; mss = avgs["MSS-Head"]
                pen_mil = (no - mil) if (no is not None and mil is not None) else None
                pen_mss = (no - mss) if (no is not None and mss is not None) else None

                def _fmt_pen(v):
                    return f"{v:>+8.1f}" if v is not None else "     N/A"

                print(f"  {m:<12}  {tau:>5}  "
                      f"{_fmt(no)}  {_fmt(mil)}  {_fmt(mss)}  "
                      f"{_fmt_pen(pen_mil)}  {_fmt_pen(pen_mss)}")

    # Operational range (R in {10,15,25}) pooled summary
    print(f"\n  === Operational range summary (R in {{10, 15, 25}} Ohm, pooled over tau) ===")
    header = (f"  {'method':<12}  "
              f"{'NoVib':>9}  {'MIL-STD':>9}  {'MSS-Head':>9}  "
              f"{'PenMIL':>8}  {'PenMSS':>8}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    ops_R = [10, 15, 25]
    for m in METHODS:
        agg = {sc: [] for sc in scenarios}
        for R in ops_R:
            for tau in eval_taus:
                for r in rescored_all[R][tau][m]:
                    if r["margin"] is None: continue
                    if r["scenario_name"] in agg:
                        agg[r["scenario_name"]].append(r["margin"])
        def _mean(l): return float(np.mean(l)) if l else None
        no = _mean(agg["NoVib"]); mil = _mean(agg["MIL-STD"]); mss = _mean(agg["MSS-Head"])
        pen_mil = (no - mil) if (no is not None and mil is not None) else None
        pen_mss = (no - mss) if (no is not None and mss is not None) else None
        def _fmt(v): return f"{v:>+9.1f}" if v is not None else "      N/A"
        def _fmt_pen(v): return f"{v:>+8.1f}" if v is not None else "     N/A"
        print(f"  {m:<12}  {_fmt(no)}  {_fmt(mil)}  {_fmt(mss)}  "
              f"{_fmt_pen(pen_mil)}  {_fmt_pen(pen_mss)}")


def _print_vibration_fa(raw_results, eval_taus):
    """TABLE 6. 진동이 사전 오경보 (pre-onset FA) 빈도에 미치는 영향."""
    print(f"\n{'=' * 80}")
    print("TABLE 6. Pre-onset False Alarms per scenario (sum of n_pre_fa across all runs)")
    print(f"  onset 이전 윈도우에서 ISC 로 오예측한 횟수의 합. 0 이 이상적.")
    print("=" * 80)
    header = (f"  {'method':<12}  {'tau':>5}  "
              f"{'NoVib':>7}  {'MIL-STD':>7}  {'MSS-Head':>8}  {'total':>7}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for m in METHODS:
        method_total = 0
        m_fa = {"NoVib": 0, "MIL-STD": 0, "MSS-Head": 0}
        for tau in eval_taus:
            runs = raw_results[tau][m]
            fa = {"NoVib": 0, "MIL-STD": 0, "MSS-Head": 0}
            for r in runs:
                if r["scenario_name"] in fa:
                    fa[r["scenario_name"]] += r["n_pre_fa"]
                    m_fa[r["scenario_name"]] += r["n_pre_fa"]
            total = sum(fa.values())
            method_total += total
            print(f"  {m:<12}  {tau:>5}  "
                  f"{fa['NoVib']:>7}  {fa['MIL-STD']:>7}  {fa['MSS-Head']:>8}  {total:>7}")
        print(f"  {m:<12}  {'ALL':>5}  "
              f"{m_fa['NoVib']:>7}  {m_fa['MIL-STD']:>7}  {m_fa['MSS-Head']:>8}  {method_total:>7}")
        print()


if __name__ == "__main__":
    main()
